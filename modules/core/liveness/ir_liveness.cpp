#include "ir_liveness.hpp"

#include "core/cviai_types_mem.h"
#include "rescale_utils.hpp"

#include <cmath>
#include <iostream>
#include "core/core/cvai_core_types.h"
#include "core/core/cvai_errno.h"
#include "cvi_sys.h"

#define R_SCALE (float)(1 / 255.0)
#define G_SCALE (float)(1 / 255.0)
#define B_SCALE (float)(1 / 255.0)
#define R_MEAN 0
#define G_MEAN 0
#define B_MEAN 0

namespace cviai {

IrLiveness::IrLiveness() : Core(CVI_MEM_DEVICE) {}

IrLiveness::~IrLiveness() {}

int IrLiveness::setupInputPreprocess(std::vector<InputPreprecessSetup> *data) {
  if (data->size() != 1) {
    LOGE("Ir liveness only has 1 input.\n");
    return CVIAI_ERR_INVALID_ARGS;
  }
  (*data)[0].factor[0] = R_SCALE;
  (*data)[0].factor[1] = G_SCALE;
  (*data)[0].factor[2] = B_SCALE;
  (*data)[0].mean[0] = R_MEAN;
  (*data)[0].mean[1] = G_MEAN;
  (*data)[0].mean[2] = B_MEAN;
  (*data)[0].use_quantize_scale = true;
  (*data)[0].rescale_type = RESCALE_NOASPECT;
  (*data)[0].use_crop = true;

  return CVIAI_SUCCESS;
}

int IrLiveness::inference(VIDEO_FRAME_INFO_S *stOutFrame, cvai_face_t *meta) {
  uint32_t img_width = stOutFrame->stVFrame.u32Width;
  uint32_t img_height = stOutFrame->stVFrame.u32Height;
  for (uint32_t i = 0; i < meta->size; i++) {
    cvai_face_info_t face_info = info_rescale_c(img_width, img_height, *meta, i);
    int box_x1 = face_info.bbox.x1;
    int box_y1 = face_info.bbox.y1;
    uint32_t box_w = face_info.bbox.x2 - face_info.bbox.x1;
    uint32_t box_h = face_info.bbox.y2 - face_info.bbox.y1;
    CVI_AI_FreeCpp(&face_info);

    m_vpss_config[0].crop_attr.enCropCoordinate = VPSS_CROP_RATIO_COOR;
    m_vpss_config[0].crop_attr.stCropRect = {box_x1, box_y1, box_w, box_h};

    std::vector<VIDEO_FRAME_INFO_S *> frames = {stOutFrame};
    int ret = run(frames);
    if (ret != CVIAI_SUCCESS) {
      return ret;
    }

    float *out_data = getOutputRawPtr<float>(0);

    meta->info[i].liveness_score = out_data[0];
  }

  return CVIAI_SUCCESS;
}

}  // namespace cviai