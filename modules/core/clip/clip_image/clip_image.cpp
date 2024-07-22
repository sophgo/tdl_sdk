#include "clip_image.hpp"
#include <core/core/cvtdl_errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <error_msg.hpp>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>
#include "coco_utils.hpp"
#include "core/core/cvtdl_errno.h"
#include "core/cvi_tdl_types_mem.h"
#include "core/cvi_tdl_types_mem_internal.h"
#include "core/utils/vpss_helper.h"
#include "core_utils.hpp"
#include "cvi_sys.h"

namespace cvitdl {

Clip_Image::Clip_Image() : Core(CVI_MEM_DEVICE) {}

Clip_Image::~Clip_Image() {}

int Clip_Image::setupInputPreprocess(std::vector<InputPreprecessSetup>* data) {
  if (data->size() != 1) {
    LOGE("CLIP only has 1 input.\n");
    return CVI_TDL_ERR_INVALID_ARGS;
  }
  (*data)[0].factor[0] = 0.0145984266;
  (*data)[0].factor[1] = 0.0150077685;
  (*data)[0].factor[2] = 0.0142200657;
  (*data)[0].mean[0] = 1.7922625;
  (*data)[0].mean[1] = 1.7465649;
  (*data)[0].mean[2] = 1.4802198;
  (*data)[0].use_quantize_scale = true;
  (*data)[0].use_crop = false;
  (*data)[0].keep_aspect_ratio = true;
  std::cout << "setupInputPreprocess done" << std::endl;
  return CVI_TDL_SUCCESS;
}

int Clip_Image::inference(VIDEO_FRAME_INFO_S* frame, cvtdl_clip_feature* clip_feature) {
  std::vector<VIDEO_FRAME_INFO_S*> frames = {frame};
  int ret = run(frames);
  if (ret != CVI_TDL_SUCCESS) {
    std::cout << "cvi_tdl clip inference run is fail!\n";
    return ret;
  }
  float* out_feature = getOutputRawPtr<float>(0);
  CVI_SHAPE output_shape = getOutputShape(0);
  clip_feature->feature_dim = output_shape.dim[1];
  clip_feature->out_feature = (float*)malloc(clip_feature->feature_dim * sizeof(float));
  memcpy(clip_feature->out_feature, out_feature, clip_feature->feature_dim * sizeof(float));
  model_timer_.TicToc("post");
  return CVI_TDL_SUCCESS;
}
}  // namespace cvitdl
