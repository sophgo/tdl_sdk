#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <map>
#include <memory>
#include <numeric>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include "coco_utils.hpp"
#include "core/core/cvai_core_types.h"
#include "core/object/cvai_object_types.h"
#include "core_utils.hpp"
#include "cvi_comm_vpss.h"
#include "cvi_sys.h"
#include "cviruntime.h"
#include "misc.hpp"
#include "object_detection/mobiledetv2/mobiledetv2.hpp"

#include "core/utils/vpss_helper.h"

static const float STD_R = (255.0 * 0.229);
static const float STD_G = (255.0 * 0.224);
static const float STD_B = (255.0 * 0.225);
static const float MODEL_MEAN_R = 0.485 * 255.0;
static const float MODEL_MEAN_G = 0.456 * 255.0;
static const float MODEL_MEAN_B = 0.406 * 255.0;

#define FACTOR_R (1.0 / STD_R)
#define FACTOR_G (1.0 / STD_G)
#define FACTOR_B (1.0 / STD_B)
#define MEAN_R (MODEL_MEAN_R / STD_R)
#define MEAN_G (MODEL_MEAN_G / STD_G)
#define MEAN_B (MODEL_MEAN_B / STD_B)

using namespace std;

namespace cviai {
using Detections = MobileDetV2::Detections;
using PtrDectRect = MobileDetV2::PtrDectRect;
using MDetV2Config = MobileDetV2::CvimodelInfo;

static vector<size_t> sort_indexes(const Detections &v) {
  // initialize original index locations
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values
  stable_sort(idx.begin(), idx.end(),
              [&v](size_t i1, size_t i2) { return v[i1]->score > v[i2]->score; });
  return idx;
}

static vector<size_t> calculate_area(const Detections &dets) {
  vector<size_t> areas(dets.size());
  for (size_t i = 0; i < dets.size(); i++) {
    areas[i] = (dets[i]->x2 - dets[i]->x1) * (dets[i]->y2 - dets[i]->y1);
  }
  return areas;
}

static Detections nms(const Detections &dets, float iou_threshold) {
  vector<int> keep(dets.size(), 0);
  vector<int> suppressed(dets.size(), 0);

  size_t ndets = dets.size();
  size_t num_to_keep = 0;

  vector<size_t> order = sort_indexes(dets);
  vector<size_t> areas = calculate_area(dets);

  for (size_t _i = 0; _i < ndets; _i++) {
    auto i = order[_i];
    if (suppressed[i] == 1) continue;
    keep[num_to_keep++] = i;
    auto ix1 = dets[i]->x1;
    auto iy1 = dets[i]->y1;
    auto ix2 = dets[i]->x2;
    auto iy2 = dets[i]->y2;
    auto iarea = areas[i];

    for (size_t _j = _i + 1; _j < ndets; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1) continue;
      auto xx1 = std::max(ix1, dets[j]->x1);
      auto yy1 = std::max(iy1, dets[j]->y1);
      auto xx2 = std::min(ix2, dets[j]->x2);
      auto yy2 = std::min(iy2, dets[j]->y2);

      auto w = std::max(0.0f, xx2 - xx1);
      auto h = std::max(0.0f, yy2 - yy1);
      auto inter = w * h;
      float ovr = static_cast<float>(inter) / (iarea + areas[j] - inter);
      if (ovr > iou_threshold && dets[j]->label == dets[i]->label) suppressed[j] = 1;
    }
  }

  Detections final_dets(num_to_keep);
  for (size_t k = 0; k < num_to_keep; k++) {
    final_dets[k] = dets[keep[k]];
  }
  return final_dets;
}

static void convert_det_struct(const Detections &dets, cvai_object_t *out, int im_height,
                               int im_width, meta_rescale_type_e type,
                               const MobileDetV2::CvimodelInfo &config) {
  out->size = dets.size();
  out->info = (cvai_object_info_t *)malloc(sizeof(cvai_object_info_t) * out->size);
  out->height = im_height;
  out->width = im_width;
  out->rescale_type = type;

  memset(out->info, 0, sizeof(cvai_object_info_t) * out->size);
  for (uint32_t i = 0; i < out->size; ++i) {
    out->info[i].bbox.x1 = dets[i]->x1;
    out->info[i].bbox.y1 = dets[i]->y1;
    out->info[i].bbox.x2 = dets[i]->x2;
    out->info[i].bbox.y2 = dets[i]->y2;
    out->info[i].bbox.score = dets[i]->score;
    out->info[i].classes = config.class_id_map(dets[i]->label);
    const string &classname = coco_utils::class_names_91[out->info[i].classes];
    strncpy(out->info[i].name, classname.c_str(), sizeof(out->info[i].name));
  }
}

static void decode_box(const float *const box, const AnchorBox &anchor, const PtrDectRect &det) {
  float ycenter_a = anchor.y + anchor.h / 2;
  float xcenter_a = anchor.x + anchor.w / 2;

  float ty = box[0];
  float tx = box[1];
  float th = box[2];
  float tw = box[3];

  float w = std::exp(tw) * anchor.w;
  float h = std::exp(th) * anchor.h;
  float ycenter = ty * anchor.h + ycenter_a;
  float xcenter = tx * anchor.w + xcenter_a;
  det->x1 = xcenter - w / 2;
  det->y1 = ycenter - h / 2;
  det->x2 = xcenter + w / 2;
  det->y2 = ycenter + h / 2;
}

static void clip_bbox(const size_t image_width, const size_t image_height, const PtrDectRect &box) {
  if (box->x1 < 0) box->x1 = 0;
  if (box->y1 < 0) box->y1 = 0;
  if (box->x2 < 0) box->x2 = 0;
  if (box->y2 < 0) box->y2 = 0;

  if (box->x1 >= image_width) box->x1 = image_width - 1;
  if (box->y1 >= image_height) box->y1 = image_height - 1;
  if (box->x2 >= image_width) box->x2 = image_width - 1;
  if (box->y2 >= image_height) box->y2 = image_height - 1;
}

static std::vector<int8_t> constructInverseThresh(float threshld, std::vector<int> strides,
                                                  std::map<int, float> dequant_thresh) {
  std::vector<int8_t> inverse_threshold;
  float inverse_th = std::log(threshld / (1 - threshld));
  for (int stride : strides) {
    int8_t quant_score_thresh =
        static_cast<int8_t>(round(inverse_th * 128 / dequant_thresh[stride]));

    inverse_threshold.push_back(quant_score_thresh);
  }

  return inverse_threshold;
}

MobileDetV2::MobileDetV2(MobileDetV2::Model model, float iou_thresh)
    : Core(CVI_MEM_DEVICE, true),
      m_model_config(MDetV2Config::create_config(model)),
      m_iou_threshold(iou_thresh) {
  m_model_threshold = m_model_config.default_score_threshold;
  /**
   *  To speedup post-process of MobileDetV2, we apply inverse function of sigmoid to threshold
   *  and compare with logits directly. That improve post-process speed because of skipping
   *  compute sigmoid on whole class logits tensors.
   *  The inverse function of sigmoid is f(x) = ln(y / 1-y)
   */
  m_quant_inverse_score_threshold = constructInverseThresh(
      m_model_threshold, m_model_config.strides, m_model_config.class_dequant_thresh);

  m_filter.set();  // select all classes
}

MobileDetV2::~MobileDetV2() {}

void MobileDetV2::setModelThreshold(float threshold) {
  if (m_model_threshold != threshold) {
    m_model_threshold = threshold;
    m_quant_inverse_score_threshold = constructInverseThresh(
        m_model_threshold, m_model_config.strides, m_model_config.class_dequant_thresh);
  }
}

int MobileDetV2::setupInputPreprocess(std::vector<InputPreprecessSetup> *data) {
  if (data->size() != 1) {
    LOGE("Mobiledetv2 only has 1 input.\n");
    return CVI_FAILURE;
  }
  (*data)[0].factor[0] = static_cast<float>(FACTOR_R);
  (*data)[0].factor[1] = static_cast<float>(FACTOR_G);
  (*data)[0].factor[2] = static_cast<float>(FACTOR_B);
  (*data)[0].mean[0] = static_cast<float>(MEAN_R);
  (*data)[0].mean[1] = static_cast<float>(MEAN_G);
  (*data)[0].mean[2] = static_cast<float>(MEAN_B);
  (*data)[0].use_quantize_scale = true;
  (*data)[0].rescale_type = RESCALE_RB;
  (*data)[0].resize_method = VPSS_SCALE_COEF_OPENCV_BILINEAR;
  return CVI_SUCCESS;
}

int MobileDetV2::vpssPreprocess(const std::vector<VIDEO_FRAME_INFO_S *> &srcFrames,
                                std::vector<std::shared_ptr<VIDEO_FRAME_INFO_S>> *dstFrames) {
  auto *srcFrame = srcFrames[0];
  auto *dstFrame = (*dstFrames)[0].get();
  auto &vpssChnAttr = m_vpss_config[0].chn_attr;
  auto &factor = vpssChnAttr.stNormalize.factor;
  auto &mean = vpssChnAttr.stNormalize.mean;
  VPSS_CHN_SQ_RB_HELPER(&vpssChnAttr, srcFrame->stVFrame.u32Width, srcFrame->stVFrame.u32Height,
                        vpssChnAttr.u32Width, vpssChnAttr.u32Height, PIXEL_FORMAT_RGB_888_PLANAR,
                        factor, mean, false);
  int ret = mp_vpss_inst->sendFrame(srcFrame, &vpssChnAttr, &m_vpss_config[0].chn_coeff, 1);
  if (ret != CVI_SUCCESS) {
    LOGE("Send frame failed with %#x!\n", ret);
    return ret;
  }
  return mp_vpss_inst->getFrame(dstFrame, 0, m_vpss_timeout);
}

int MobileDetV2::onModelOpened() {
  CVI_SHAPE input_shape = getInputShape(0);
  m_model_config.image_height = input_shape.dim[2];
  m_model_config.image_width = input_shape.dim[3];

  RetinaNetAnchorGenerator generator = RetinaNetAnchorGenerator(
      m_model_config.min_level, m_model_config.max_level, m_model_config.num_scales,
      m_model_config.aspect_ratios, m_model_config.anchor_scale, m_model_config.image_width,
      m_model_config.image_height);
  m_anchors = generator.get_anchor_boxes();
  return CVI_SUCCESS;
}

void MobileDetV2::generate_dets_for_tensor(Detections *det_vec, float class_dequant_thresh,
                                           float bbox_dequant_thresh, int8_t quant_thresh,
                                           const int8_t *logits, const int8_t *objectness,
                                           int8_t *bboxes, size_t class_tensor_size,
                                           const vector<AnchorBox> &anchors) {
  for (size_t obj_index = 0; obj_index < class_tensor_size; obj_index++) {
    if (unlikely(*(objectness + obj_index) >= quant_thresh)) {
      // create detection if any object exists in this grid
      size_t score_index = obj_index * m_model_config.num_classes;
      size_t end = score_index + m_model_config.num_classes;

      // find objects in this grid
      for (size_t class_idx = score_index; class_idx < end; class_idx++) {
        if (logits[class_idx] >= quant_thresh) {
          size_t anchor_index = class_idx / m_model_config.num_classes;
          size_t box_index = anchor_index * 4;
          PtrDectRect det = make_shared<object_detect_rect_t>();
          det->label = class_idx - score_index;

          float dequant_logits = logits[class_idx] * class_dequant_thresh / 128.0;
          det->score = 1.0 / (1.0 + std::exp(-dequant_logits));

          float dequant_box[4];
          Dequantize(bboxes + box_index, dequant_box, bbox_dequant_thresh, 4);
          decode_box(dequant_box, anchors[box_index / 4], det);
          clip_bbox(m_model_config.image_width, m_model_config.image_height, det);
          det_vec->push_back(det);
        }
      }
    }
  }
}

void MobileDetV2::generate_dets_for_each_stride(Detections *det_vec) {
  vector<pair<int8_t *, size_t>> cls_raw_out;
  vector<pair<int8_t *, size_t>> objectness_raw_out;
  vector<pair<int8_t *, size_t>> bbox_raw_out;
  get_raw_outputs(&cls_raw_out, &objectness_raw_out, &bbox_raw_out);

  auto class_thresh_iter = m_model_config.class_dequant_thresh.begin();
  auto bbox_thresh_iter = m_model_config.bbox_dequant_thresh.begin();

  for (size_t stride_index = 0; stride_index < cls_raw_out.size(); stride_index++) {
    generate_dets_for_tensor(
        det_vec, class_thresh_iter->second, bbox_thresh_iter->second,
        m_quant_inverse_score_threshold[stride_index], cls_raw_out[stride_index].first,
        objectness_raw_out[stride_index].first, bbox_raw_out[stride_index].first,
        static_cast<int>(objectness_raw_out[stride_index].second), m_anchors[stride_index]);

    class_thresh_iter++;
    bbox_thresh_iter++;
  }
}

void MobileDetV2::get_raw_outputs(std::vector<pair<int8_t *, size_t>> *cls_tensor_ptr,
                                  std::vector<pair<int8_t *, size_t>> *objectness_tensor_ptr,
                                  std::vector<pair<int8_t *, size_t>> *bbox_tensor_ptr) {
  for (auto stride : m_model_config.strides) {
    {
      const TensorInfo &info = getOutputTensorInfo(m_model_config.class_out_names[stride]);
      cls_tensor_ptr->push_back({info.get<int8_t>(), info.tensor_elem});
    }

    {
      const TensorInfo &info = getOutputTensorInfo(m_model_config.bbox_out_names[stride]);
      bbox_tensor_ptr->push_back({info.get<int8_t>(), info.tensor_elem});
    }

    {
      const TensorInfo &info = getOutputTensorInfo(m_model_config.obj_max_names[stride]);
      objectness_tensor_ptr->push_back({info.get<int8_t>(), info.tensor_elem});
    }
  }
}

// TODO: remove old filter
void coco_class_90_filter(Detections &dets, cvai_obj_det_type_e det_type) {
  auto condition = [det_type](const PtrDectRect &det) {
    int label = det->label;
    bool skip_class = (det_type != CVI_DET_TYPE_ALL);
    if ((det_type & CVI_DET_TYPE_VEHICLE) != 0) {
      if ((label >= CVI_AI_DET_TYPE_BICYCLE) && label <= CVI_AI_DET_TYPE_BOAT) skip_class = false;
    }
    if ((det_type & CVI_DET_TYPE_PEOPLE) != 0) {
      if (label == 0) skip_class = false;
    }
    if ((det_type & CVI_DET_TYPE_PET) != 0) {
      if ((label == CVI_AI_DET_TYPE_DOG) || (label == CVI_AI_DET_TYPE_CAT)) skip_class = false;
    }
    return skip_class;
  };
  dets.erase(remove_if(dets.begin(), dets.end(), condition), dets.end());
}

void MobileDetV2::select_classes(const std::vector<uint32_t> &selected_classes) {
  m_filter.reset();
  for (auto c : selected_classes) {
    m_filter.set(c, true);
  }
}

int MobileDetV2::inference(VIDEO_FRAME_INFO_S *frame, cvai_object_t *meta,
                           cvai_obj_det_type_e det_type) {
  int ret = CVI_SUCCESS;
  std::vector<VIDEO_FRAME_INFO_S *> frames = {frame};

  ret = run(frames);
  Detections dets;
  generate_dets_for_each_stride(&dets);

  Detections final_dets = nms(dets, m_iou_threshold);

  if (!m_filter.all()) {  // filter if not all bit are set
    auto condition = [this](const PtrDectRect &det) { return !m_filter.test(det->label); };
    final_dets.erase(remove_if(final_dets.begin(), final_dets.end(), condition), final_dets.end());
  } else if (det_type != CVI_DET_TYPE_ALL) {
    // TODO: remove old filter
    coco_class_90_filter(final_dets, det_type);
  }

  CVI_SHAPE shape = getInputShape(0);

  convert_det_struct(final_dets, meta, shape.dim[2], shape.dim[3], m_vpss_config[0].rescale_type,
                     m_model_config);

  if (!hasSkippedVpssPreprocess()) {
    for (uint32_t i = 0; i < meta->size; ++i) {
      meta->info[i].bbox =
          box_rescale(frame->stVFrame.u32Width, frame->stVFrame.u32Height, meta->width,
                      meta->height, meta->info[i].bbox, meta_rescale_type_e::RESCALE_RB);
    }
    meta->width = frame->stVFrame.u32Width;
    meta->height = frame->stVFrame.u32Height;
  }

  return ret;
}

MDetV2Config MDetV2Config::create_config(MobileDetV2::Model model) {
  MDetV2Config config;
  config.min_level = 3;
  config.max_level = 7;
  config.num_scales = 3;
  config.aspect_ratios = {{1.0, 1.0}, {1.4, 0.7}, {0.7, 1.4}};
  config.anchor_scale = 4.0;
  config.strides = {8, 16, 32, 64, 128};
  config.num_classes = 90;
  config.class_out_names = {{8, "class_stride_8"},
                            {16, "class_stride_16"},
                            {32, "class_stride_32"},
                            {64, "class_stride_64"},
                            {128, "class_stride_128"}};

  config.obj_max_names = {{8, "class_stride_8_obj_max"},
                          {16, "class_stride_16_obj_max"},
                          {32, "class_stride_32_obj_max"},
                          {64, "class_stride_64_obj_max"},
                          {128, "class_stride_128_obj_max"}};

  config.bbox_out_names = {{8, "box_stride_8"},
                           {16, "box_stride_16"},
                           {32, "box_stride_32"},
                           {64, "box_stride_64"},
                           {128, "box_stride_128"}};

  config.class_id_map = [](int orig_id) { return orig_id; };

  switch (model) {
    case Model::d0:
      config.class_dequant_thresh = {{8, 17.464866638183594},
                                     {16, 15.640022277832031},
                                     {32, 14.935582160949707},
                                     {64, 16.390363693237305},
                                     {128, 15.530133247375488}};

      config.bbox_dequant_thresh = {{8, 2.0975120067596436},
                                    {16, 2.7213516235351562},
                                    {32, 2.6431756019592285},
                                    {64, 2.9647181034088135},
                                    {128, 12.42608642578125}};
      config.default_score_threshold = 0.4;
      break;
    case Model::d1:
      config.class_dequant_thresh = {{8, 14.168907165527344},
                                     {16, 15.155534744262695},
                                     {32, 16.111759185791016},
                                     {64, 15.438838958740234},
                                     {128, 14.437296867370605}};

      config.bbox_dequant_thresh = {{8, 2.5651912689208984},
                                    {16, 2.8889713287353516},
                                    {32, 2.650554656982422},
                                    {64, 2.727674961090088},
                                    {128, 2.260598659515381}};
      config.default_score_threshold = 0.3;
      break;
    case Model::d2:
      config.class_dequant_thresh = {{8, 11.755056381225586},
                                     {16, 12.826586723327637},
                                     {32, 13.664835929870605},
                                     {64, 14.224205017089844},
                                     {128, 12.782066345214844}};

      config.bbox_dequant_thresh = {{8, 3.082498550415039},
                                    {16, 2.491678476333618},
                                    {32, 2.8060667514801025},
                                    {64, 2.563389301300049},
                                    {128, 2.2213821411132812}};
      config.default_score_threshold = 0.3;
      break;
    case Model::lite:
      config.num_classes = 9;
      config.class_dequant_thresh = {{8, 12.48561954498291},
                                     {16, 11.416889190673828},
                                     {32, 13.258634567260742},
                                     {64, 13.312114715576172},
                                     {128, 9.732277870178223}};

      config.bbox_dequant_thresh = {{8, 2.7503433227539062},
                                    {16, 2.9586639404296875},
                                    {32, 2.537200927734375},
                                    {64, 2.3935775756835938},
                                    {128, 2.543354034423828}};
      config.default_score_threshold = 0.3;
      break;
    case Model::vehicle_d0:
      config.num_classes = 3;
      config.class_dequant_thresh = {{8, 8.599571228027344},
                                     {16, 8.887327194213867},
                                     {32, 9.710219383239746},
                                     {64, 10.174678802490234},
                                     {128, 10.896987915039062}};

      config.bbox_dequant_thresh = {{8, 2.2055277824401855},
                                    {16, 2.6225955486297607},
                                    {32, 2.434586763381958},
                                    {64, 4.202951431274414},
                                    {128, 4.039170742034912}};
      config.default_score_threshold = 0.3;
      config.class_id_map = [](int orig_id) {
        if (orig_id == 0) return static_cast<int>(CVI_AI_DET_TYPE_CAR);
        if (orig_id == 1) return static_cast<int>(CVI_AI_DET_TYPE_TRUCK);
        if (orig_id == 2) return static_cast<int>(CVI_AI_DET_TYPE_MOTORBIKE);
        return static_cast<int>(CVI_AI_DET_TYPE_END);
      };
      break;
    case Model::pedestrian_d0:
      config.num_classes = 1;
      config.class_dequant_thresh = {{8, 7.848998546600342},
                                     {16, 8.222152709960938},
                                     {32, 8.384308815002441},
                                     {64, 9.911425590515137},
                                     {128, 8.275640487670898}};

      config.bbox_dequant_thresh = {{8, 2.3595869541168213},
                                    {16, 2.3849966526031494},
                                    {32, 2.362071990966797},
                                    {64, 2.2918550968170166},
                                    {128, 2.743276596069336}};
      config.default_score_threshold = 0.3;
      config.class_id_map = [](int orig_id) {
        if (orig_id == 0) return static_cast<int>(CVI_AI_DET_TYPE_PERSON);
        return static_cast<int>(CVI_AI_DET_TYPE_END);
      };
      break;
  }

  return config;
}
}  // namespace cviai
