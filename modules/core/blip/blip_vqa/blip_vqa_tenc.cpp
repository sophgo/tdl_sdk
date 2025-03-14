
#include "blip_vqa_tenc.hpp"
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

Blip_Vqa_Tenc::Blip_Vqa_Tenc() : Core(CVI_MEM_SYSTEM) {}

Blip_Vqa_Tenc::~Blip_Vqa_Tenc() {}

int Blip_Vqa_Tenc::inference(cvtdl_image_embeds* embeds_meta, cvtdl_tokens* tokens_meta) {
  const TensorInfo& image_embeds_info = getInputTensorInfo("image_embeds");
  const TensorInfo& input_ids_info = getInputTensorInfo("input_ids");
  const TensorInfo& attention_mask_info = getInputTensorInfo("attention_mask");

  float* image_embeds_ptr = image_embeds_info.get<float>();
  int32_t* input_ids_ptr = input_ids_info.get<int32_t>();
  int32_t* attention_mask_ptr = attention_mask_info.get<int32_t>();

  memcpy(image_embeds_ptr, embeds_meta->images_embeds,
         embeds_meta->height * embeds_meta->width * sizeof(float));
  memcpy(input_ids_ptr, tokens_meta->input_ids[0], 35 * sizeof(int32_t));
  memcpy(attention_mask_ptr, tokens_meta->attention_mask[0], 35 * sizeof(int32_t));

  VIDEO_FRAME_INFO_S tmp_frame;
  std::vector<VIDEO_FRAME_INFO_S*> frames = {&tmp_frame};
  int ret = run(frames);
  if (ret != CVI_TDL_SUCCESS) {
    std::cout << "cvi_tdl Blip_Vqa_Tenc inference run is fail!\n";
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
