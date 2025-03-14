
#include "blip_vqa_venc.hpp"
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

Blip_Vqa_Venc::Blip_Vqa_Venc() : Core(CVI_MEM_DEVICE) {
  m_preprocess_param[0].factor[0] = R_SCALE;
  m_preprocess_param[0].factor[1] = G_SCALE;
  m_preprocess_param[0].factor[2] = B_SCALE;
  m_preprocess_param[0].mean[0] = R_MEAN;
  m_preprocess_param[0].mean[1] = G_MEAN;
  m_preprocess_param[0].mean[2] = B_MEAN;
  m_preprocess_param[0].keep_aspect_ratio = true;
  m_preprocess_param[0].format = PIXEL_FORMAT_FP32_C3_PLANAR;
}

Blip_Vqa_Venc::~Blip_Vqa_Venc() {}

int Blip_Vqa_Venc::inference(VIDEO_FRAME_INFO_S* frame, cvtdl_image_embeds* embeds_meta) {
  std::vector<VIDEO_FRAME_INFO_S*> frames = {frame};
  int ret = run(frames);
  if (ret != CVI_TDL_SUCCESS) {
    std::cout << "cvi_tdl Blip_Vqa_Venc inference run is fail!\n";
    return ret;
  }

  float* out = getOutputRawPtr<float>(0);

  CVI_SHAPE output_shape = getOutputShape(0);

  CVI_TDL_MemAllocInit(output_shape.dim[1], output_shape.dim[2], embeds_meta);
  memcpy(embeds_meta->images_embeds, out,
         output_shape.dim[1] * output_shape.dim[2] * sizeof(float));

  model_timer_.TicToc("post");
  return CVI_TDL_SUCCESS;
}
}  // namespace cvitdl
