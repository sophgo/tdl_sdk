#include "custom.hpp"

#include "core/cviai_types_mem.h"
#include "core/cviai_types_mem_internal.h"

#include <cvi_buffer.h>
#include <cvi_vpss.h>

namespace cviai {

Custom::Custom() {
  mp_config = std::make_unique<ModelConfig>();
  mp_config->skip_postprocess = true;
  mp_config->input_mem_type = CVI_MEM_DEVICE;
}

int Custom::initAfterModelOpened(float *factor, float *mean, bool &pad_reverse,
                                 bool &keep_aspect_ratio, bool &use_model_threshold) {
  if (mp_config->input_mem_type == CVI_MEM_DEVICE && !m_skip_vpss_preprocess && m_factor.empty()) {
    LOGE("VPSS is set to use. Please set factor, mean and initialize first.\n");
    return CVI_FAILURE;
  }
  if (m_factor.size() == 1) {
    for (int i = 0; i < 3; i++) {
      factor[i] = m_factor[0];
      mean[i] = m_mean[0];
    }
  } else if (m_factor.size() == 3) {
    for (int i = 0; i < 3; i++) {
      factor[i] = m_factor[i];
      mean[i] = m_mean[i];
    }
  } else {
    LOGE("factor and mean must have 1 or 3 values. Current: %zu.\n", m_factor.size());
    return CVI_FAILURE;
  }
  keep_aspect_ratio = m_keep_aspect_ratio;
  use_model_threshold = m_use_model_threashold;
  return CVI_SUCCESS;
}

int Custom::setSQParam(const float *factor, const float *mean, const uint32_t length,
                       const bool use_model_threshold, const bool keep_aspect_ratio) {
  if (length != 1 && length != 3) {
    LOGE("Scale parameter only supports legnth of 1 or 3. Given: %u.\n", length);
    return CVI_FAILURE;
  }
  m_factor.clear();
  m_mean.clear();
  for (uint32_t i = 0; i < length; i++) {
    m_factor.push_back(factor[i]);
    m_mean.push_back(mean[i]);
  }
  m_use_model_threashold = use_model_threshold;
  m_keep_aspect_ratio = keep_aspect_ratio;
  return CVI_SUCCESS;
}

int Custom::setPreProcessFunc(preProcessFunc func, bool use_tensor_input, bool use_vpss_sq) {
  if (func == NULL && use_tensor_input) {
    LOGE("Function pointer cannot be NULL if use_tensor_input is enabled.\n");
    return CVI_FAILURE;
  }
  preprocessfunc = func;
  if (use_tensor_input) {
    m_skip_vpss_preprocess = true;
  } else {
    m_skip_vpss_preprocess = !use_vpss_sq;
  }
  if (use_tensor_input) {
    mp_config->input_mem_type = CVI_MEM_SYSTEM;
  }
  return CVI_SUCCESS;
}

int Custom::setSkipPostProcess(const bool skip) {
  mp_config->skip_postprocess = skip;
  return CVI_SUCCESS;
}

int Custom::getNCHW(const char *tensor_name, uint32_t *n, uint32_t *c, uint32_t *h, uint32_t *w) {
  CVI_TENSOR *input = CVI_NN_GetTensorByName(tensor_name, mp_input_tensors, m_input_num);
  if (input == NULL) {
    LOGE("Tensor %s not found.\n", tensor_name);
    return CVI_FAILURE;
  }

  *n = input->shape.dim[0];
  *c = input->shape.dim[1];
  *h = input->shape.dim[2];
  *w = input->shape.dim[3];
  return CVI_SUCCESS;
}

int Custom::inference(VIDEO_FRAME_INFO_S *stInFrame) {
  if (preprocessfunc != NULL) {
    VIDEO_FRAME_INFO_S stOutFrame;
    memset(&stOutFrame, 0, sizeof(VIDEO_FRAME_INFO_S));
    preprocessfunc(stInFrame, &stOutFrame);
    auto ret = run(&stOutFrame);
    if (stOutFrame.stVFrame.u64PhyAddr[0] != 0 &&
        (stInFrame->stVFrame.u64PhyAddr[0] != stOutFrame.stVFrame.u64PhyAddr[0])) {
      CVI_VPSS_ReleaseChnFrame(0, 0, &stOutFrame);
    }
    return ret;
  }
  return run(stInFrame);
}

int Custom::getOutputTensor(const char *tensor_name, int8_t **tensor, uint32_t *tensor_count,
                            uint16_t *unit_size) {
  CVI_TENSOR *out = CVI_NN_GetTensorByName(tensor_name, mp_output_tensors, m_output_num);
  if (out == NULL) {
    LOGE("Tensor not found.\n");
    return CVI_FAILURE;
  }
  *tensor = (int8_t *)CVI_NN_TensorPtr(out);
  *tensor_count = CVI_NN_TensorCount(out);
  *unit_size = CVI_NN_TensorSize(out) / *tensor_count;
  return CVI_SUCCESS;
}

}  // namespace cviai