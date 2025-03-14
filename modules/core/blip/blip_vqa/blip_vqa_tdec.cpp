
#include "blip_vqa_tdec.hpp"
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

Blip_Vqa_Tdec::Blip_Vqa_Tdec() : Core(CVI_MEM_SYSTEM) {}

Blip_Vqa_Tdec::~Blip_Vqa_Tdec() {}

int Blip_Vqa_Tdec::inference(cvtdl_image_embeds* embeds_meta, cvtdl_tokens* tokens_meta) {
  const TensorInfo& question_states_info = getInputTensorInfo(0);

  float* question_states_ptr = question_states_info.get<float>();

  memcpy(question_states_ptr, embeds_meta->images_embeds,
         embeds_meta->height * embeds_meta->width * sizeof(float));

  VIDEO_FRAME_INFO_S tmp_frame;
  std::vector<VIDEO_FRAME_INFO_S*> frames = {&tmp_frame};
  int ret = run(frames);
  if (ret != CVI_TDL_SUCCESS) {
    std::cout << "cvi_tdl Blip_Vqa_Tdec inference run is fail!\n";
    return ret;
  }

  CVI_SHAPE output_shape = getOutputShape(0);

  float* out = getOutputRawPtr<float>(0);

  CVI_TDL_FreeCpp(tokens_meta);
  tokens_meta->max_length = output_shape.dim[1];
  tokens_meta->sentences_num = 1;
  tokens_meta->input_ids = (int32_t**)malloc(tokens_meta->sentences_num * sizeof(int32_t*));
  tokens_meta->input_ids[0] = (int32_t*)malloc(output_shape.dim[1] * sizeof(int32_t));
  memset(tokens_meta->input_ids[0], 0, output_shape.dim[1] * sizeof(int32_t));

  for (int i = 0; i < output_shape.dim[1]; i++) {
    tokens_meta->input_ids[0][i] = (int32_t)out[i];
  }

  model_timer_.TicToc("post");
  return CVI_TDL_SUCCESS;
}
}  // namespace cvitdl
