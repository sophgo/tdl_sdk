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

Clip_Image::Clip_Image() : Core(CVI_MEM_SYSTEM) {
  // preprocess_params_[0].factor[0] = 0.0145984266;
  // preprocess_params_[0].factor[1] = 0.0150077685;
  // preprocess_params_[0].factor[2] = 0.0142200657;
  // preprocess_params_[0].mean[0] = 1.7922625;
  // preprocess_params_[0].mean[1] = 1.7465649;
  // preprocess_params_[0].mean[2] = 1.4802198;

  // use fused_precess w4f16 model
  preprocess_params_[0].factor[0] = 1;
  preprocess_params_[0].factor[1] = 1;
  preprocess_params_[0].factor[2] = 1;
  preprocess_params_[0].mean[0] = 0;
  preprocess_params_[0].mean[1] = 0;
  preprocess_params_[0].mean[2] = 0;
  preprocess_params_[0].use_crop = true;
  preprocess_params_[0].keep_aspect_ratio = true;
}

Clip_Image::~Clip_Image() {}

int Clip_Image::inference(VIDEO_FRAME_INFO_S* frame, cvtdl_clip_feature* clip_feature) {
  const TensorInfo& tinfo = getInputTensorInfo(0);

  if (tinfo.data_type == 2) {
    CVI_U32 height = frame->stVFrame.u32Height;
    CVI_U32 width = frame->stVFrame.u32Width;
    int crop_x = 0, crop_y = 0;
    int min_size = std::min(width, height);
    if (width > height) {
      crop_x = (width - height) / 2;
    } else {
      crop_y = (height - width) / 2;
    }
    preprocess_params_[0].use_crop = true;
    preprocess_params_[0].keep_aspect_ratio = true;
    preprocess_params_[0].crop_x = crop_x;
    preprocess_params_[0].crop_y = crop_y;
    preprocess_params_[0].crop_w = min_size;
    preprocess_params_[0].crop_h = min_size;
  } else if (tinfo.data_type == 0) {
    float* input_ptr = tinfo.get<float>();
    float* temp_buffer;
    int h = frame->stVFrame.u32Height;
    int w = frame->stVFrame.u32Width;
    temp_buffer = reinterpret_cast<float*>(frame->stVFrame.pu8VirAddr[0]);
    memcpy(input_ptr, temp_buffer, h * w * 3 * sizeof(float));
    delete[] temp_buffer;
  }

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
