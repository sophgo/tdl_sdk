
#include "blip_cap.hpp"
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

#define R_SCALE (1 / (256.0 * 0.269))
#define G_SCALE (1 / (256.0 * 0.261))
#define B_SCALE (1 / (256.0 * 0.276))
#define R_MEAN (0.481 / 0.269)
#define G_MEAN (0.458 / 0.261)
#define B_MEAN (0.408 / 0.276)

namespace cvitdl {

Blip_Cap::Blip_Cap() : Core(CVI_MEM_DEVICE) {
  m_preprocess_param[0].factor[0] = R_SCALE;
  m_preprocess_param[0].factor[1] = G_SCALE;
  m_preprocess_param[0].factor[2] = B_SCALE;
  m_preprocess_param[0].mean[0] = R_MEAN;
  m_preprocess_param[0].mean[1] = G_MEAN;
  m_preprocess_param[0].mean[2] = B_MEAN;
  m_preprocess_param[0].keep_aspect_ratio = true;
  m_preprocess_param[0].format = PIXEL_FORMAT_FP32_C3_PLANAR;
}

Blip_Cap::~Blip_Cap() {}

int Blip_Cap::inference(VIDEO_FRAME_INFO_S* frame, cvtdl_tokens* tokens_meta) {
  std::vector<VIDEO_FRAME_INFO_S*> frames = {frame};
  int ret = run(frames);
  if (ret != CVI_TDL_SUCCESS) {
    std::cout << "cvi_tdl Blip_Cap inference run is fail!\n";
    return ret;
  }

  CVI_SHAPE output_shape = getOutputShape(0);
  float* out = getOutputRawPtr<float>(0);
  printf("out[0]: %f\n", out[0]);

  CVI_TDL_FreeCpp(tokens_meta);
  tokens_meta->max_length = output_shape.dim[1];
  tokens_meta->sentences_num = 1;
  tokens_meta->input_ids = (int32_t**)malloc(tokens_meta->sentences_num * sizeof(int32_t*));
  tokens_meta->input_ids[0] = (int32_t*)malloc(output_shape.dim[1] * sizeof(int32_t));
  memset(tokens_meta->input_ids[0], 0, output_shape.dim[1] * sizeof(int32_t));

  for (int i = 0; i < output_shape.dim[1]; i++) {
    tokens_meta->input_ids[0][i] = (int32_t)out[i];
    printf("tokens_meta->input_ids[0][%d]: %d\n", i, tokens_meta->input_ids[0][i]);
  }

  model_timer_.TicToc("post");
  return CVI_TDL_SUCCESS;
}
}  // namespace cvitdl
