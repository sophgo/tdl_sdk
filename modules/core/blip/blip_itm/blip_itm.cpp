
#include "blip_itm.hpp"
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

Blip_Itm::Blip_Itm() : Core(CVI_MEM_DEVICE) {
  m_preprocess_param[0].factor[0] = R_SCALE;
  m_preprocess_param[0].factor[1] = G_SCALE;
  m_preprocess_param[0].factor[2] = B_SCALE;
  m_preprocess_param[0].mean[0] = R_MEAN;
  m_preprocess_param[0].mean[1] = G_MEAN;
  m_preprocess_param[0].mean[2] = B_MEAN;
  m_preprocess_param[0].keep_aspect_ratio = true;
  m_preprocess_param[0].format = PIXEL_FORMAT_FP32_C3_PLANAR;
}

Blip_Itm::~Blip_Itm() {}

int Blip_Itm::inference(VIDEO_FRAME_INFO_S* frame, cvtdl_tokens* tokens_meta,
                        cvtdl_class_meta_t* cls_meta) {
  const TensorInfo& input_ids_info = getInputTensorInfo("input_ids");
  const TensorInfo& attention_mask_info = getInputTensorInfo("attention_mask");

  int32_t* input_ids_ptr = input_ids_info.get<int32_t>();
  int32_t* attention_mask_ptr = attention_mask_info.get<int32_t>();

  CVI_TDL_FreeCpp(cls_meta);

  for (int i = 0; i < tokens_meta->sentences_num; i++) {
    memcpy(input_ids_ptr, tokens_meta->input_ids[i], 35 * sizeof(int32_t));
    memcpy(attention_mask_ptr, tokens_meta->attention_mask[i], 35 * sizeof(int32_t));

    std::vector<VIDEO_FRAME_INFO_S*> frames = {frame};
    int ret = run(frames);
    if (ret != CVI_TDL_SUCCESS) {
      std::cout << "cvi_tdl Blip_Itm inference run is fail!\n";
      return ret;
    }

    float* out = getOutputRawPtr<float>(0);

    if (out[0] > cls_meta->score[0]) {
      cls_meta->score[0] = out[0];
      cls_meta->cls[0] = i;
    }
  }

  model_timer_.TicToc("post");
  return CVI_TDL_SUCCESS;
}
}  // namespace cvitdl
