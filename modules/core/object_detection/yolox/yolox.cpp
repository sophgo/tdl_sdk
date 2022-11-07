#include <algorithm>
#include <cmath>
#include <iterator>

#include "coco_utils.hpp"
#include "core/core/cvai_errno.h"
#include "core/cviai_types_mem.h"
#include "core/cviai_types_mem_internal.h"
#include "core_utils.hpp"
#include "cvi_sys.h"
#include "object_utils.hpp"
#include "yolox.hpp"

#define R_SCALE 1
#define G_SCALE 1
#define B_SCALE 1
#define R_MEAN 0
#define G_MEAN 0
#define B_MEAN 0
#define NUM_CLASSES 80
#define NMS_THRESH 0.55
#define NAME_OUTPUT "output_Transpose_dequant"

namespace cviai {

struct GridAndStride {
  int grid0;
  int grid1;
  int stride;
};

static void convert_det_struct(const Detections &dets, cvai_object_t *out, int im_height,
                               int im_width, meta_rescale_type_e type) {
  CVI_AI_MemAllocInit(dets.size(), out);
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
    out->info[i].classes = dets[i]->label;
    const std::string &classname = coco_utils::class_names_91[out->info[i].classes];
    strncpy(out->info[i].name, classname.c_str(), sizeof(out->info[i].name));
  }
}

static void generate_grids_and_stride(const int target_w, const int target_h,
                                      std::vector<int> &strides,
                                      std::vector<GridAndStride> &grid_strides) {
  for (auto stride : strides) {
    int num_grid_w = target_w / stride;
    int num_grid_h = target_h / stride;
    for (int g1 = 0; g1 < num_grid_h; g1++) {
      for (int g0 = 0; g0 < num_grid_w; g0++) {
        grid_strides.push_back((GridAndStride){g0, g1, stride});
      }
    }
  }
}

static void generate_yolox_proposals(std::vector<GridAndStride> grid_strides, const float *feat_ptr,
                                     float prob_threshold, Detections &detections) {
  const int num_anchors = grid_strides.size();

  for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
    const int grid0 = grid_strides[anchor_idx].grid0;
    const int grid1 = grid_strides[anchor_idx].grid1;
    const int stride = grid_strides[anchor_idx].stride;

    const int basic_pos = anchor_idx * (NUM_CLASSES + 5);

    // yolox/models/yolo_head.py decode logic
    //  outputs[..., :2] = (outputs[..., :2] + grids) * strides
    //  outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
    float x_center = (feat_ptr[basic_pos + 0] + grid0) * stride;
    float y_center = (feat_ptr[basic_pos + 1] + grid1) * stride;
    float w = std::exp(feat_ptr[basic_pos + 2]) * stride;
    float h = std::exp(feat_ptr[basic_pos + 3]) * stride;
    float x0 = x_center - w * 0.5f;
    float y0 = y_center - h * 0.5f;

    float box_objectness = feat_ptr[basic_pos + 4];
    const float *score_start = feat_ptr + basic_pos + 5;
    const float *score_end = score_start + NUM_CLASSES;
    auto iter = std::max_element(score_start, score_end);
    float box_prob = box_objectness * *iter;

    if (box_prob > prob_threshold) {
      PtrDectRect det = std::make_shared<object_detect_rect_t>();
      det->label = coco_utils::class_id_map_80_to_91[std::distance(score_start, iter)];
      det->score = box_prob;
      det->x1 = x0;
      det->y1 = y0;
      det->x2 = x0 + w;
      det->y2 = y0 + h;
      detections.push_back(det);
    }  // class loop
  }    // point anchor loop
}

YoloX::YoloX() : Core(CVI_MEM_DEVICE) {
  m_filter.set();  // select all classes
}

YoloX::~YoloX() {}

void YoloX::select_classes(const std::vector<uint32_t> &selected_classes) {
  m_filter.reset();
  for (auto c : selected_classes) {
    m_filter.set(c, true);
  }
}

int YoloX::setupInputPreprocess(std::vector<InputPreprecessSetup> *data) {
  if (data->size() != 1) {
    LOGE("YoloX only has 1 input.\n");
    return CVIAI_ERR_INVALID_ARGS;
  }

  (*data)[0].factor[0] = R_SCALE;
  (*data)[0].factor[1] = G_SCALE;
  (*data)[0].factor[2] = B_SCALE;
  (*data)[0].mean[0] = R_MEAN;
  (*data)[0].mean[1] = G_MEAN;
  (*data)[0].mean[2] = B_MEAN;
  (*data)[0].format = PIXEL_FORMAT_BGR_888_PLANAR;
  (*data)[0].use_quantize_scale = true;
  return CVIAI_SUCCESS;
}

int YoloX::inference(VIDEO_FRAME_INFO_S *srcFrame, cvai_object_t *obj_meta) {
  std::vector<VIDEO_FRAME_INFO_S *> frames = {srcFrame};
  int ret = run(frames);
  if (ret != CVIAI_SUCCESS) {
    return ret;
  }

  CVI_SHAPE shape = getInputShape(0);
  outputParser(shape.dim[3], shape.dim[2], srcFrame->stVFrame.u32Width,
               srcFrame->stVFrame.u32Height, obj_meta);
  model_timer_.TicToc(3, "post");
  return CVIAI_SUCCESS;
}

void YoloX::outputParser(const int image_width, const int image_height, const int frame_width,
                         const int frame_height, cvai_object_t *obj_meta) {
  float *output_blob = getOutputRawPtr<float>(NAME_OUTPUT);

  std::vector<int> strides = {8, 16, 32};
  std::vector<GridAndStride> grid_strides;
  generate_grids_and_stride(image_width, image_height, strides, grid_strides);

  Detections vec_obj;
  generate_yolox_proposals(grid_strides, output_blob, m_model_threshold, vec_obj);

  // Do nms on output result
  Detections final_dets = nms_multi_class(vec_obj, NMS_THRESH);

  if (!m_filter.all()) {  // filter if not all bit are set
    auto condition = [this](const PtrDectRect &det) { return !m_filter.test(det->label); };
    final_dets.erase(remove_if(final_dets.begin(), final_dets.end(), condition), final_dets.end());
  }

  CVI_SHAPE shape = getInputShape(0);

  convert_det_struct(final_dets, obj_meta, shape.dim[2], shape.dim[3],
                     m_vpss_config[0].rescale_type);

  if (!hasSkippedVpssPreprocess()) {
    for (uint32_t i = 0; i < obj_meta->size; ++i) {
      obj_meta->info[i].bbox =
          box_rescale(frame_width, frame_height, obj_meta->width, obj_meta->height,
                      obj_meta->info[i].bbox, obj_meta->rescale_type);
    }
    obj_meta->width = frame_width;
    obj_meta->height = frame_height;
  }
}

}  // namespace cviai
