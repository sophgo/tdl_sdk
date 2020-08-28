#ifdef __ARM_ARCH
#include <arm_neon.h>
#else
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include "neon2sse/NEON_2_SSE.h"
#pragma clang diagnostic pop
#endif
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
#include "core_utils.hpp"
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
static const float quant_thresh = 2.641289710998535;

#define FACTOR_R (128.0 / (STD_R * quant_thresh))
#define FACTOR_G (128.0 / (STD_G * quant_thresh))
#define FACTOR_B (128.0 / (STD_B * quant_thresh))
#define MEAN_R ((128.0 * MODEL_MEAN_R) / (STD_R * quant_thresh))
#define MEAN_G ((128.0 * MODEL_MEAN_G) / (STD_G * quant_thresh))
#define MEAN_B ((128.0 * MODEL_MEAN_B) / (STD_B * quant_thresh))

using namespace std;

namespace cviai {
using Detections = MobileDetV2::Detections;
using PtrDectRect = MobileDetV2::PtrDectRect;
using MDetV2Config = MobileDetV2::ModelConfig;

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
      if (ovr > iou_threshold) suppressed[j] = 1;
    }
  }

  Detections final_dets(num_to_keep);
  size_t index = 0;
  for (size_t k = 0; k < num_to_keep; k++) {
    final_dets[index++] = dets[keep[k]];
  }
  return final_dets;
}

static void convert_det_struct(const Detections &dets, cvai_object_t *out, int im_height,
                               int im_width) {
  out->size = dets.size();
  out->info = (cvai_object_info_t *)malloc(sizeof(cvai_object_info_t) * out->size);
  out->height = im_height;
  out->width = im_width;

  memset(out->info, 0, sizeof(cvai_object_info_t) * out->size);
  for (uint32_t i = 0; i < out->size; ++i) {
    out->info[i].bbox.x1 = dets[i]->x1;
    out->info[i].bbox.y1 = dets[i]->y1;
    out->info[i].bbox.x2 = dets[i]->x2;
    out->info[i].bbox.y2 = dets[i]->y2;
    out->info[i].bbox.score = dets[i]->score;
    out->info[i].classes = coco_utils::map_90_class_id_to_80(dets[i]->label);
    const string &classname = coco_utils::class_names_80[out->info[i].classes];
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

static void clip_bbox(const size_t image_size, const PtrDectRect &box) {
  if (box->x1 < 0) box->x1 = 0;
  if (box->y1 < 0) box->y1 = 0;
  if (box->x2 >= image_size) box->x2 = image_size - 1;
  if (box->y2 >= image_size) box->y2 = image_size - 1;
}

MobileDetV2::MobileDetV2(MobileDetV2::Model model, float iou_thresh, float score_thresh)
    : m_model_config(MDetV2Config::create_config(model)),
      m_iou_threshold(iou_thresh),
      m_score_threshold(score_thresh) {
  mp_config = std::make_unique<cviai::ModelConfig>();
  mp_config->skip_postprocess = true;
  mp_config->input_mem_type = CVI_MEM_DEVICE;
  mp_config->skip_preprocess = true;

  RetinaNetAnchorGenerator generator = RetinaNetAnchorGenerator(
      m_model_config.min_level, m_model_config.max_level, m_model_config.num_scales,
      m_model_config.aspect_ratios, m_model_config.anchor_scale, m_model_config.image_size);
  m_anchors = generator.get_anchor_boxes();

  /**
   *  To speedup post-process of MobileDetV2, we apply inverse function of sigmoid to threshold
   *  and compare with logits directly. That improve post-process speed because of skipping
   *  compute sigmoid on whole class logits tensors.
   *  The inverse function of sigmoid is f(x) = ln(y / 1-y)
   */
  float inverse_th = std::log(m_score_threshold / (1 - m_score_threshold));
  for (auto stride : m_model_config.strides) {
    int8_t quant_score_thresh =
        static_cast<int8_t>(round(inverse_th * 128 / m_model_config.class_dequant_thresh[stride]));

    m_quant_inverse_score_threshold.push_back(quant_score_thresh);
  }
}

MobileDetV2::~MobileDetV2() {}

int MobileDetV2::initAfterModelOpened() {
  CVI_TENSOR *input = getInputTensor(0);
  VPSS_CHN_ATTR_S vpssChnAttr;
  const float factor[] = {FACTOR_R, FACTOR_G, FACTOR_B};
  const float mean[] = {MEAN_R, MEAN_G, MEAN_B};
  VPSS_CHN_SQ_HELPER(&vpssChnAttr, input->shape.dim[3], input->shape.dim[2],
                     PIXEL_FORMAT_RGB_888_PLANAR, factor, mean, false);
  m_vpss_chn_attr.push_back(vpssChnAttr);

  return CVI_SUCCESS;
}

#if defined(__arm64__) || defined(__aarch64__)
static inline __attribute__((always_inline)) uint32_t sum_q(int8x16_t v) { return vaddvq_s8(v); }
#else
static inline __attribute__((always_inline)) uint32_t sum_q(int8x16_t v) {
  /**
   * v: [0][1][2][3][4][5][6][7][8][9][10][11][12][13][14][15]
   *
   *             [ 8][ 9][10][11][12][13][14][15]
   * folding:   +[ 0][ 1][ 2][ 3][ 4][ 5][ 6][ 7]
   *           ---------------------------------
   *             [ 0][ 1][ 2][ 3][ 4][ 5][ 6][ 7]
   *               \   /   \   /   \  /    \  /
   * padd_16:      [ 0]    [ 1]    [ 2]    [ 3]
   *                 \      /        \      /
   * padd_32:          [ 0]            [ 1]
   *                    |                |
   * return:           [ 0]     +      [ 1]
   */
  int8x8_t folding = vadd_s8(vget_high_s8(v), vget_low_s8(v));
  int16x4_t padd_16 = vpaddl_s8(folding);
  int32x2_t padd_32 = vpaddl_s16(padd_16);
  return (uint32_t)(vget_lane_s32(padd_32, 0) + vget_lane_s32(padd_32, 1));
}
#endif

static inline __attribute__((always_inline)) uint32_t get_num_object_in_grid(const int8_t *logits,
                                                                             size_t count,
                                                                             int8_t quant_thresh) {
  // calculate how much scores greater than threshold using NEON intrinsics
  // don't record the index here, because it needs if-branches and couple memory write ops.
  // we check index later if there is at least one object.
  int8x16_t thresh_vec = vdupq_n_s8(quant_thresh);
  const size_t rest = 10;  // 90 classs % 16 = 10, use hardcode value because it's easy for
                           // unrolling loop by compilier
  int8x16_t sum_vec = vdupq_n_s8(0);
  for (size_t class_idx = 0; class_idx < count - rest; class_idx += 16) {
    int8x16_t value = vld1q_s8(logits + class_idx);
    uint8x16_t cmp = vcgeq_s8(value, thresh_vec);
    sum_vec = vsubq_s8(sum_vec, (int8x16_t)cmp);
  }

  uint32_t num_objects = sum_q(sum_vec);

  const int8_t *rest_arr = logits + (count - rest);
  for (size_t start = 0; start < rest; start++) {
    if (*(rest_arr + start) >= quant_thresh) {
      num_objects++;
    }
  }
  return num_objects;
}

void MobileDetV2::generate_dets_for_tensor(Detections *det_vec, float class_dequant_thresh,
                                           float bbox_dequant_thresh, int8_t quant_thresh,
                                           const int8_t *logits, int8_t *bboxes,
                                           size_t class_tensor_size,
                                           const vector<AnchorBox> &anchors) {
  for (size_t score_index = 0; score_index < class_tensor_size;
       score_index += m_model_config.num_classes) {
    uint32_t num_objects =
        get_num_object_in_grid(logits + score_index, m_model_config.num_classes, quant_thresh);

    // create detection if any object exists in this grid
    if (unlikely(num_objects)) {
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
          clip_bbox(m_model_config.image_size, det);
          det_vec->push_back(det);
        }
      }
    }
  }
}

void MobileDetV2::generate_dets_for_each_stride(Detections *det_vec) {
  vector<pair<int8_t *, size_t>> cls_raw_out;
  vector<pair<int8_t *, size_t>> bbox_raw_out;
  get_raw_outputs(&cls_raw_out, &bbox_raw_out);

  auto class_thresh_iter = m_model_config.class_dequant_thresh.begin();
  auto bbox_thresh_iter = m_model_config.bbox_dequant_thresh.begin();

  for (size_t stride_index = 0; stride_index < cls_raw_out.size(); stride_index++) {
    generate_dets_for_tensor(det_vec, class_thresh_iter->second, bbox_thresh_iter->second,
                             m_quant_inverse_score_threshold[stride_index],
                             cls_raw_out[stride_index].first, bbox_raw_out[stride_index].first,
                             static_cast<int>(cls_raw_out[stride_index].second),
                             m_anchors[stride_index]);

    class_thresh_iter++;
    bbox_thresh_iter++;
  }
}

void MobileDetV2::get_tensor_ptr_size(const std::string &tname, int8_t **ptr, size_t *size) {
  CVI_TENSOR *tensor = CVI_NN_GetTensorByName(tname.c_str(), mp_output_tensors, m_output_num);
  CVI_SHAPE tensor_shape = CVI_NN_TensorShape(tensor);
  *size = tensor_shape.dim[0] * tensor_shape.dim[1] * tensor_shape.dim[2] * tensor_shape.dim[3];
  *ptr = (int8_t *)CVI_NN_TensorPtr(tensor);
}

void MobileDetV2::get_raw_outputs(std::vector<pair<int8_t *, size_t>> *cls_tensor_ptr,
                                  std::vector<pair<int8_t *, size_t>> *bbox_tensor_ptr) {
  for (auto stride : m_model_config.strides) {
    int8_t *tensor = nullptr;
    size_t tsize = 0;
    get_tensor_ptr_size(m_model_config.class_out_names[stride], &tensor, &tsize);
    cls_tensor_ptr->push_back({tensor, tsize});

    tensor = nullptr;
    tsize = 0;
    get_tensor_ptr_size(m_model_config.bbox_out_names[stride], &tensor, &tsize);
    bbox_tensor_ptr->push_back({tensor, tsize});
  }
}

int MobileDetV2::inference(VIDEO_FRAME_INFO_S *frame, cvai_object_t *meta,
                           cvai_obj_det_type_t det_type) {
  CVI_TENSOR *input = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, mp_input_tensors, m_input_num);

  int ret = CVI_SUCCESS;
  if (m_skip_vpss_preprocess) {
    ret = run(frame);
  } else {
    VIDEO_FRAME_INFO_S stDstFrame;
    mp_vpss_inst->sendFrame(frame, &m_vpss_chn_attr[0], 1);
    ret = mp_vpss_inst->getFrame(&stDstFrame, 0);
    if (ret != CVI_SUCCESS) {
      printf("CVI_VPSS_GetChnFrame failed with %#x\n", ret);
      return ret;
    }
    ret = run(&stDstFrame);

    ret |= mp_vpss_inst->releaseFrame(&stDstFrame, 0);
    if (ret != CVI_SUCCESS) {
      return ret;
    }
  }

  Detections dets;
  generate_dets_for_each_stride(&dets);

  Detections final_dets = nms(dets, m_iou_threshold);

  // remove all detections not belong to COCO 80 classes
  auto condition = [](const PtrDectRect &det) {
    return coco_utils::map_90_class_id_to_80(det->label) == -1;
  };
  final_dets.erase(remove_if(final_dets.begin(), final_dets.end(), condition), final_dets.end());

  convert_det_struct(final_dets, meta, input->shape.dim[2], input->shape.dim[3]);

  if (!m_skip_vpss_preprocess) {
    for (uint32_t i = 0; i < meta->size; ++i) {
      meta->info[i].bbox = box_rescale_c(frame->stVFrame.u32Width, frame->stVFrame.u32Height,
                                         meta->width, meta->height, meta->info[i].bbox);
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

  config.bbox_out_names = {{8, "box_stride_8"},
                           {16, "box_stride_16"},
                           {32, "box_stride_32"},
                           {64, "box_stride_64"},
                           {128, "box_stride_128"}};

  switch (model) {
    case Model::d0:
      config.image_size = 512;
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
      break;
    case Model::d1:
      config.image_size = 640;
      break;
    case Model::d2:
      config.image_size = 768;
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
      break;
  }

  return config;
}
}  // namespace cviai