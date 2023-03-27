#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <iterator>

#include <core/core/cvai_errno.h>
#include <error_msg.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "coco_utils.hpp"
#include "core/core/cvai_errno.h"
#include "core/cviai_types_mem.h"
#include "core/cviai_types_mem_internal.h"
#include "core/utils/vpss_helper.h"
#include "core_utils.hpp"
#include "cvi_sys.h"
#include "hand_detection.hpp"
#include "object_utils.hpp"

#define R_SCALE 0.003922
#define G_SCALE 0.003922
#define B_SCALE 0.003922
#define R_MEAN 0
#define G_MEAN 0
#define B_MEAN 0
#define NMS_THRESH 0.5

namespace cviai {
static void convert_det_struct(const Detections &dets, cvai_object_t *hand, int im_height,
                               int im_width) {
  CVI_AI_MemAllocInit(dets.size(), hand);
  hand->height = im_height;
  hand->width = im_width;
  memset(hand->info, 0, sizeof(cvai_object_info_t) * hand->size);

  for (uint32_t i = 0; i < hand->size; ++i) {
    hand->info[i].bbox.x1 = dets[i]->x1;
    hand->info[i].bbox.y1 = dets[i]->y1;
    hand->info[i].bbox.x2 = dets[i]->x2;
    hand->info[i].bbox.y2 = dets[i]->y2;
    hand->info[i].bbox.score = dets[i]->score;
  }
}

HandDetection::HandDetection() : Core(CVI_MEM_DEVICE) {}

HandDetection::~HandDetection() {}

int HandDetection::setupInputPreprocess(std::vector<InputPreprecessSetup> *data) {
  if (data->size() != 1) {
    LOGE("HandDetection only has 1 input.\n");
    return CVIAI_ERR_INVALID_ARGS;
  }

  (*data)[0].factor[0] = R_SCALE;
  (*data)[0].factor[1] = G_SCALE;
  (*data)[0].factor[2] = B_SCALE;
  (*data)[0].mean[0] = R_MEAN;
  (*data)[0].mean[1] = G_MEAN;
  (*data)[0].mean[2] = B_MEAN;
  (*data)[0].use_quantize_scale = true;
  (*data)[0].rescale_type = RESCALE_RB;
  return CVIAI_SUCCESS;
}

int HandDetection::vpssPreprocess(VIDEO_FRAME_INFO_S *srcFrame, VIDEO_FRAME_INFO_S *dstFrame,
                                  VPSSConfig &vpss_config) {
  auto &vpssChnAttr = vpss_config.chn_attr;
  auto &factor = vpssChnAttr.stNormalize.factor;
  auto &mean = vpssChnAttr.stNormalize.mean;
  VPSS_CHN_SQ_RB_HELPER(&vpssChnAttr, srcFrame->stVFrame.u32Width, srcFrame->stVFrame.u32Height,
                        vpssChnAttr.u32Width, vpssChnAttr.u32Height, PIXEL_FORMAT_RGB_888_PLANAR,
                        factor, mean, false);
  int ret = mp_vpss_inst->sendFrame(srcFrame, &vpssChnAttr, &vpss_config.chn_coeff, 1);

  if (ret != CVI_SUCCESS) {
    LOGE("Send frame failed: %s!\n", get_vpss_error_msg(ret));
    return CVIAI_ERR_VPSS_SEND_FRAME;
  }

  ret = mp_vpss_inst->getFrame(dstFrame, 0, 2000);
  if (ret != CVI_SUCCESS) {
    LOGE("get frame failed: %s!\n", get_vpss_error_msg(ret));
    return CVIAI_ERR_VPSS_GET_FRAME;
  }

  return CVIAI_SUCCESS;
}

int HandDetection::inference(VIDEO_FRAME_INFO_S *srcFrame, cvai_object_t *obj_meta) {
  std::vector<VIDEO_FRAME_INFO_S *> frames = {srcFrame};
  int ret = run(frames);
  if (ret != CVIAI_SUCCESS) {
    return ret;
  }
  CVI_SHAPE shape = getInputShape(0);

  outputParser(shape.dim[3], shape.dim[2], srcFrame->stVFrame.u32Width,
               srcFrame->stVFrame.u32Height, obj_meta);
  model_timer_.TicToc("post");
  return CVIAI_SUCCESS;
}

void HandDetection::outputParser(const int image_width, const int image_height,
                                 const int frame_width, const int frame_height,
                                 cvai_object_t *obj_meta) {
  TensorInfo oinfo = getOutputTensorInfo(0);
  float *output_blob = getOutputRawPtr<float>(oinfo.tensor_name);

  TensorInfo oinfo_cls = getOutputTensorInfo(1);
  float *output_blob_cls = getOutputRawPtr<float>(oinfo_cls.tensor_name);

  Detections vec_obj;
  CVI_SHAPE output = getOutputShape(0);

  int feat_w = output.dim[2];
  for (int i = 0; i < feat_w; i++) {
    float score = output_blob_cls[i];
    if (score >= m_model_threshold) {
      float x = output_blob[0 * feat_w + i];
      float y = output_blob[1 * feat_w + i];
      float w = output_blob[2 * feat_w + i];
      float h = output_blob[3 * feat_w + i];
      int x1 = int((x - 0.5 * w));
      int y1 = int((y - 0.5 * h));

      int x2 = int((x + 0.5 * w));
      int y2 = int((y + 0.5 * h));

      PtrDectRect det = std::make_shared<object_detect_rect_t>();
      det->score = score;
      det->x1 = x1;
      det->y1 = y1;
      det->x2 = x2;
      det->y2 = y2;
      vec_obj.push_back(det);
    }
  }

  Detections final_dets = nms_multi_class(vec_obj, NMS_THRESH);
  CVI_SHAPE shape = getInputShape(0);

  convert_det_struct(final_dets, obj_meta, shape.dim[2], shape.dim[3]);

  if (!hasSkippedVpssPreprocess()) {
    for (uint32_t i = 0; i < obj_meta->size; ++i) {
      obj_meta->info[i].bbox =
          box_rescale(frame_width, frame_height, obj_meta->width, obj_meta->height,
                      obj_meta->info[i].bbox, meta_rescale_type_e::RESCALE_RB);
    }
    obj_meta->width = frame_width;
    obj_meta->height = frame_height;
  }
}
// namespace cviai
}  // namespace cviai
