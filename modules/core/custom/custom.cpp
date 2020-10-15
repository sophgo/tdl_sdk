#include "custom.hpp"

#include "core/cviai_types_mem.h"
#include "core/cviai_types_mem_internal.h"
#include "core/utils/vpss_helper.h"
#include "core_utils.hpp"

#include "cvi_sys.h"

namespace cviai {

Custom::Custom() {
  mp_config = std::make_unique<ModelConfig>();
  mp_config->skip_postprocess = true;
  mp_config->skip_preprocess = true;
  mp_config->input_mem_type = CVI_MEM_DEVICE;
}

int Custom::initAfterModelOpened() {
  if (mp_config->input_mem_type == CVI_MEM_DEVICE && !m_skip_vpss_preprocess && m_factor.empty()) {
    LOGE("VPSS is set to use. Please set factor, mean and initialize first.\n");
    return CVI_FAILURE;
  }
  m_vpss_chn_attr.clear();
  CVI_TENSOR *input = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, mp_input_tensors, m_input_num);
  // FIXME: quant_thresh is not correct
  // float quant_thresh = CVI_NN_TensorQuantScale(input);
  float quan_scale, mean_quan_scale;
  if (m_quant_threshold == 128.f) {
    quan_scale = 1;
    mean_quan_scale = 1;
  } else {
    quan_scale = 128.f / m_quant_threshold;
    mean_quan_scale = (-1) * quan_scale;
  }
  float factor[3], mean[3];
  if (m_factor.size() == 1) {
    float q_factor = m_factor[0] * quan_scale;
    float q_mean = m_mean[0] * mean_quan_scale;
    for (int i = 0; i < 3; i++) {
      factor[i] = q_factor;
      mean[i] = q_mean;
    }
  } else if (m_factor.size() == 3) {
    for (int i = 0; i < 3; i++) {
      float q_factor = m_factor[i] * quan_scale;
      float q_mean = m_mean[i] * mean_quan_scale;
      factor[i] = q_factor;
      mean[i] = q_mean;
    }
  } else {
    LOGE("factor and mean must have 1 or 3 values. Current: %zu.\n", m_factor.size());
    return CVI_FAILURE;
  }

  VPSS_CHN_ATTR_S vpssChnAttr;
  VPSS_CHN_SQ_HELPER(&vpssChnAttr, input->shape.dim[3], input->shape.dim[2],
                     PIXEL_FORMAT_RGB_888_PLANAR, factor, mean, false);
  if (!m_keep_aspect_ratio) {
    vpssChnAttr.stAspectRatio.enMode = ASPECT_RATIO_NONE;
  }
  m_vpss_chn_attr.push_back(vpssChnAttr);
  return CVI_SUCCESS;
}

int Custom::setSQParam(const float *factor, const float *mean, const uint32_t length,
                       const float threshold, const bool keep_aspect_ratio) {
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
  m_quant_threshold = threshold;
  m_keep_aspect_ratio = keep_aspect_ratio;
  return CVI_SUCCESS;
}

int Custom::setSQParamRaw(const float *factor, const float *mean, const uint32_t length,
                          const bool keep_aspect_ratio) {
  return setSQParam(factor, mean, length, 128.f, keep_aspect_ratio);
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
    mp_config->skip_preprocess = false;
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