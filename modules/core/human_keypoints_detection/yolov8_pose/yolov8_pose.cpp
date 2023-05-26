#include <algorithm>
#include <cmath>
#include <iterator>

#include "coco_utils.hpp"
#include "core/core/cvai_errno.h"
#include "core/cviai_types_mem.h"
#include "core/cviai_types_mem_internal.h"
#include "core_utils.hpp"
#include "cvi_sys.h"
#include "yolov8_pose.hpp"

#define R_SCALE 1 / 255.f
#define G_SCALE 1 / 255.f
#define B_SCALE 1 / 255.f
#define R_MEAN 0
#define G_MEAN 0
#define B_MEAN 0
#define NMS_THRESH 0.7
#define SCORE_THRESH 0.25
#define NUM_KEYPOINTS 17

namespace cviai {

YoloV8Pose::YoloV8Pose() : Core(CVI_MEM_DEVICE) {}

YoloV8Pose::~YoloV8Pose() {}

int YoloV8Pose::setupInputPreprocess(std::vector<InputPreprecessSetup> *data) {
  if (data->size() != 1) {
    LOGE("YoloV8Pose only has 1 input.\n");
    return CVIAI_ERR_INVALID_ARGS;
  }

  (*data)[0].factor[0] = R_SCALE;
  (*data)[0].factor[1] = G_SCALE;
  (*data)[0].factor[2] = B_SCALE;
  (*data)[0].mean[0] = R_MEAN;
  (*data)[0].mean[1] = G_MEAN;
  (*data)[0].mean[2] = B_MEAN;
  (*data)[0].format = PIXEL_FORMAT_RGB_888_PLANAR;
  (*data)[0].use_quantize_scale = true;
  // (*data)[0].rescale_type = RESCALE_RB;
  return CVIAI_SUCCESS;
}

int YoloV8Pose::inference(VIDEO_FRAME_INFO_S *srcFrame, cvai_object_t *obj_meta) {
  std::vector<VIDEO_FRAME_INFO_S *> frames = {srcFrame};
  int ret = run(frames);
  if (ret != CVIAI_SUCCESS) {
    LOGW("YoloV8Pose run inference failed\n");
    return ret;
  }
  size_t output_num = getNumOutputTensor();

  // LOGI("start to outputParser\n");
  if (output_num == 1) {
    CVI_SHAPE output_shape = getOutputShape(0);  //////bug

    outputParser(output_shape.dim[1], output_shape.dim[2], srcFrame->stVFrame.u32Width,
                 srcFrame->stVFrame.u32Height, obj_meta);
  }

  model_timer_.TicToc("post");
  return CVIAI_SUCCESS;
}

void YoloV8Pose::outputParser(const int num_boxes, const int feature_length, const int frame_width,
                              const int frame_height, cvai_object_t *obj_meta) {
  Detections vec_obj;

  CVI_SHAPE shape = getInputShape(0);
  float *output_blob = getOutputRawPtr<float>(0);

  std::vector<int> valild_ids;
  for (int i = 0; i < num_boxes; i++) {
    float score = output_blob[i * feature_length + 4];
    if (score > SCORE_THRESH) {
      valild_ids.push_back(i);

      float cx = output_blob[i * feature_length];
      float cy = output_blob[i * feature_length];
      float w = output_blob[i * feature_length];
      float h = output_blob[i * feature_length];

      PtrDectRect det = std::make_shared<object_detect_rect_t>();
      det->score = score;
      det->x1 = cx - 0.5 * w;
      det->y1 = cy - 0.5 * h;
      det->x2 = cx + 0.5 * w;
      det->y2 = cy + 0.5 * h;
      det->label = 0;
      clip_bbox(shape.dim[3], shape.dim[2], det);

      float box_width = det->x2 - det->x1;
      float box_height = det->y2 - det->y1;
      if (box_width > 1 && box_height > 1) {
        vec_obj.push_back(det);
      }
    }
  }

  postProcess(vec_obj, frame_width, frame_height, obj_meta, valild_ids, output_blob);
}

void YoloV8Pose::postProcess(Detections &dets, int frame_width, int frame_height,
                             cvai_object_t *obj, std::vector<int> &valild_ids, float *data) {
  CVI_SHAPE shape = getInputShape(0);

  std::vector<int> keep(dets.size(), 0);

  Detections final_dets = nms_multi_class_with_ids(dets, NMS_THRESH, keep);

  CVI_AI_MemAllocInit(final_dets.size(), obj);
  obj->height = shape.dim[2];
  obj->width = shape.dim[3];
  memset(obj->info, 0, sizeof(cvai_object_info_t) * obj->size);

  for (uint32_t i = 0; i < final_dets.size(); i++) {
    obj->info[i].bbox.x1 = dets[i]->x1;
    obj->info[i].bbox.y1 = dets[i]->y1;
    obj->info[i].bbox.x2 = dets[i]->x2;
    obj->info[i].bbox.y2 = dets[i]->y2;
    obj->info[i].bbox.score = dets[i]->score;
    obj->info[i].classes = dets[i]->label;

    obj->info[i].pedestrian_properity =
        (cvai_pedestrian_meta *)malloc(sizeof(cvai_pedestrian_meta));

    int final_ids = valild_ids[keep[i]];

    int start = data[i * final_ids + 5];
    for (int j = 0; j < NUM_KEYPOINTS; j++) {
      obj->info[i].pedestrian_properity->pose_17.x[j] = data[start + j * 3];
      obj->info[i].pedestrian_properity->pose_17.y[j] = data[start + j * 3 + 1];
      obj->info[i].pedestrian_properity->pose_17.score[j] = data[start + j * 3 + 2];
    }

    if (!hasSkippedVpssPreprocess()) {
      for (uint32_t i = 0; i < obj->size; ++i) {
        obj->info[i] =
            info_rescale_c(frame_width, frame_height, obj->width, obj->height, obj->info[i]);
      }
      obj->width = frame_width;
      obj->height = frame_height;
    }
  }
}

}  // namespace cviai