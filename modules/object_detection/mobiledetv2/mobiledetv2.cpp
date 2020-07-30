#include <arm_neon.h>
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
#include "core/cvai_core_types.h"
#include "core_utils.hpp"
#include "cvi_sys.h"
#include "cviruntime.h"
#include "misc.hpp"
#include "object_detection/mobiledetv2/mobiledetv2.hpp"

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

static float inverse_sigmoid(float y) { return std::log(y / (1 - y)); }

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
  out->objects = (cvai_object_info_t *)malloc(sizeof(cvai_object_info_t) * out->size);
  out->height = im_height;
  out->width = im_width;

  memset(out->objects, 0, sizeof(cvai_object_info_t) * out->size);
  for (int i = 0; i < out->size; ++i) {
    out->objects[i].bbox.x1 = dets[i]->x1;
    out->objects[i].bbox.y1 = dets[i]->y1;
    out->objects[i].bbox.x2 = dets[i]->x2;
    out->objects[i].bbox.y2 = dets[i]->y2;
    out->objects[i].bbox.score = dets[i]->score;
    out->objects[i].classes = coco_utils::map_90_class_id_to_80(dets[i]->label);
    const string &classname = coco_utils::class_names_80[out->objects[i].classes];
    strncpy(out->objects[i].name, classname.c_str(), sizeof(out->objects[i].name));
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

static void clip_bbox(PtrDectRect box, size_t image_size) {
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

  float inverse_th = inverse_sigmoid(m_score_threshold);
  for (auto stride : m_model_config.strides) {
    int8_t quant_score_thresh =
        static_cast<int8_t>(round(inverse_th * 128 / m_model_config.class_dequant_thresh[stride]));

    m_quant_inverse_score_threshold.push_back(quant_score_thresh);
  }
}

MobileDetV2::~MobileDetV2() {}

void MobileDetV2::generate_dets_for_tensor(Detections *det_vec, float class_dequant_thresh,
                                           float bbox_dequant_thresh, int8_t quant_thresh,
                                           int8_t *logits, int8_t *bboxes, size_t size,
                                           const vector<AnchorBox> &anchors) {
  for (size_t score_index = 0; score_index < size; score_index += m_model_config.num_classes) {
#if defined(__arm64__) || defined(__aarch64__)
    // calculate how much scores greater than threshold using NEON intrinsics
    // don't record the index here, because it needs if-branches and couple memory write ops.
    // we check index later if there is at least one object.
    int8x16_t thresh_vec = vdupq_n_s8(quant_thresh);
    size_t end = score_index + m_model_config.num_classes;
    size_t rest = end % 16;
    int8x16_t sum_vec = vdupq_n_s8(0);
    for (size_t class_idx = score_index; class_idx < end - rest; class_idx += 16) {
      int8x16_t value = vld1q_s8(logits + class_idx);
      uint8x16_t cmp = vcgeq_s8(value, thresh_vec);
      sum_vec = vsubq_s8(sum_vec, (int8x16_t)cmp);
    }

    uint32_t num_objects = vaddvq_s8(sum_vec);
    if (likely(num_objects == 0)) {
      for (size_t class_idx = end - rest; class_idx < end; class_idx++) {
        if (logits[class_idx] >= quant_thresh) {
          num_objects++;
        }
      }
    }
#else   // TODO: use int8x8_t to speedup
    uint32_t num_objects = 0;
    for (size_t class_idx = score_index; class_idx < score_index + m_model_config.num_classes;
         class_idx++) {
      if (unlikely(logits[class_idx] >= quant_thresh)) {
        num_objects++;
      }
    }
#endif  // defined(__arm64__) || defined(__aarch64__)
    ////////////////////////////////////////////////////////

    if (unlikely(num_objects)) {
      // create detection if any object exists in this grid
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
          clip_bbox(det, m_model_config.image_size);
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

  int ret = run(frame);

  Detections dets;
  generate_dets_for_each_stride(&dets);

  Detections final_dets = nms(dets, m_iou_threshold);

  // remove all detections not belong to COCO 80 classes
  auto condition = [](const PtrDectRect &det) {
    return coco_utils::map_90_class_id_to_80(det->label) == -1;
  };
  final_dets.erase(remove_if(final_dets.begin(), final_dets.end(), condition), final_dets.end());

  convert_det_struct(final_dets, meta, input->shape.dim[2], input->shape.dim[3]);
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