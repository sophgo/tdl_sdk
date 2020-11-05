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

int Custom::initAfterModelOpened(std::vector<initSetup> *data) {
  for (uint32_t idx = 0; idx < data->size(); idx++) {
    CustomSQParam *p_sqparam = nullptr;
    for (uint32_t j = 0; j < m_sq_params.size(); j++) {
      if (m_sq_params[j].idx == idx) {
        p_sqparam = &m_sq_params[j];
      }
    }
    if (p_sqparam == nullptr) {
      LOGE("Error! Factor and mean of input index %u is not set.\n", idx);
      return CVI_FAILURE;
    }
    if (p_sqparam->factor.size() == 1) {
      for (int i = 0; i < 3; i++) {
        (*data)[idx].factor[i] = p_sqparam->factor[0];
        (*data)[idx].mean[i] = p_sqparam->mean[0];
      }
    } else if (p_sqparam->factor.size() == 3) {
      for (int i = 0; i < 3; i++) {
        (*data)[idx].factor[i] = p_sqparam->factor[i];
        (*data)[idx].mean[i] = p_sqparam->mean[i];
      }
    } else {
      LOGE("factor and mean must have 1 or 3 values. Current: %zu.\n", p_sqparam->factor.size());
      return CVI_FAILURE;
    }
    (*data)[idx].keep_aspect_ratio = p_sqparam->keep_aspect_ratio;
    (*data)[idx].use_quantize_scale = p_sqparam->use_model_threashold;
  }
  m_processed_frames.resize(m_input_num);
  return CVI_SUCCESS;
}

int Custom::setSQParam(const uint32_t idx, const float *factor, const float *mean,
                       const uint32_t length, const bool use_model_threshold,
                       const bool keep_aspect_ratio) {
  if (length != 1 && length != 3) {
    LOGE("Scale parameter only supports legnth of 1 or 3. Given: %u.\n", length);
    return CVI_FAILURE;
  }
  CustomSQParam *p_sqparam = nullptr;
  for (uint32_t i = 0; i < m_sq_params.size(); i++) {
    if (m_sq_params[i].idx == idx) {
      p_sqparam = &m_sq_params[i];
    }
  }
  if (p_sqparam == nullptr) {
    m_sq_params.push_back(CustomSQParam());
    p_sqparam = &m_sq_params[m_sq_params.size() - 1];
  }
  p_sqparam->idx = idx;
  p_sqparam->factor.clear();
  p_sqparam->mean.clear();
  for (uint32_t i = 0; i < length; i++) {
    p_sqparam->factor.push_back(factor[i]);
    p_sqparam->mean.push_back(mean[i]);
  }
  p_sqparam->use_model_threashold = use_model_threshold;
  p_sqparam->keep_aspect_ratio = keep_aspect_ratio;
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

int Custom::getInputShape(const char *tensor_name, uint32_t *n, uint32_t *c, uint32_t *h,
                          uint32_t *w) {
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

int Custom::getInputNum() { return m_input_num; }

int Custom::getInputShape(const uint32_t idx, uint32_t *n, uint32_t *c, uint32_t *h, uint32_t *w) {
  if (idx >= (uint32_t)m_input_num) {
    LOGE("Index exceed maximum input. (max: %d)\n", m_input_num);
    return CVI_FAILURE;
  }
  CVI_TENSOR *input = mp_input_tensors + idx;
  *n = input->shape.dim[0];
  *c = input->shape.dim[1];
  *h = input->shape.dim[2];
  *w = input->shape.dim[3];
  return CVI_SUCCESS;
}

int Custom::inference(VIDEO_FRAME_INFO_S *inFrames, uint32_t num_of_frames) {
  if (num_of_frames != (uint32_t)m_input_num) {
    LOGE("The number of input frames does not match the number of input tensors. (%u != %d)\n",
         num_of_frames, m_input_num);
    return CVI_FAILURE;
  }
  std::vector<VIDEO_FRAME_INFO_S *> frames;
  if (preprocessfunc != NULL) {
    memset(m_processed_frames.data(), 0, sizeof(VIDEO_FRAME_INFO_S) * num_of_frames);
    preprocessfunc(inFrames, m_processed_frames.data(), num_of_frames);
    for (uint32_t i = 0; i < num_of_frames; i++) {
      frames.emplace_back(&m_processed_frames[i]);
    }
    auto ret = run(frames);
    for (uint32_t i = 0; i < num_of_frames; i++) {
      if (m_processed_frames[i].stVFrame.u64PhyAddr[0] != 0 &&
          (inFrames[i].stVFrame.u64PhyAddr[0] != m_processed_frames[i].stVFrame.u64PhyAddr[0])) {
        CVI_VPSS_ReleaseChnFrame(0, 0, &m_processed_frames[i]);
      }
    }

    return ret;
  }
  for (uint32_t i = 0; i < num_of_frames; i++) {
    frames.emplace_back(&inFrames[i]);
  }
  return run(frames);
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