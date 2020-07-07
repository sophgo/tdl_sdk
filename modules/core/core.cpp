#include "core.hpp"

#include <cstdlib>

namespace cviai {

Core::~Core() {
  modelClose();
  if (mp_config != nullptr) {
    delete mp_config;
  }
}

bool Core::modelOpen(const char *filepath) {
  CVI_RC ret = CVI_NN_RegisterModel(filepath, &mp_model_handle);
  if (ret != CVI_RC_SUCCESS) {
    printf("CVI_NN_RegisterModel failed, err %d\n", ret);
    return ret;
  }
  printf("CVI_NN_RegisterModel successed\n");
  if (mp_config == nullptr) {
    printf("config not set\n");
    return CVI_RC_FAILURE;
  }
  CVI_NN_SetConfig(mp_model_handle, OPTION_PREPARE_BUF_FOR_INPUTS, mp_config->init_input_buffer);
  CVI_NN_SetConfig(mp_model_handle, OPTION_PREPARE_BUF_FOR_OUTPUTS, mp_config->init_output_buffer);
  CVI_NN_SetConfig(mp_model_handle, OPTION_OUTPUT_ALL_TENSORS, mp_config->debug_mode);
  CVI_NN_SetConfig(mp_model_handle, OPTION_SKIP_PREPROCESS, mp_config->skip_preprocess);
  CVI_NN_SetConfig(mp_model_handle, OPTION_SKIP_POSTPROCESS, mp_config->skip_postprocess);

  ret = CVI_NN_GetInputOutputTensors(mp_model_handle, &mp_input_tensors, &m_input_num,
                                     &mp_output_tensors, &m_output_num);
  if (ret != CVI_RC_SUCCESS) {
    printf("CVI_NN_GetINputsOutputs failed\n");
    return ret;
  }

  if (!isModelInputInfoValid(mv_mii)) {
    printf("Invalid model input setup");
    return CVI_RC_FAILURE;
  }

  return ret;
}

bool Core::modelClose() {
  if (mp_model_handle != nullptr) {
    if (int ret = CVI_NN_CleanupModel(mp_model_handle) != CVI_RC_SUCCESS) {
      printf("CVI_NN_CleanupModel failed, err %d\n", ret);
      return ret;
    }
    // FIXME: Don't know should free or not.
    if (mp_input_tensors != nullptr) {
      free(mp_input_tensors);
    }
    if (mp_output_tensors != nullptr) {
      free(mp_output_tensors);
    }
  }
  return CVI_RC_SUCCESS;
}

int Core::run(VIDEO_FRAME_INFO_S *stFrame) {
  int ret = preProcessing(mp_model_handle, stFrame, mp_input_tensors, m_input_num);
  if (ret != CVI_RC_SUCCESS) {
    printf("Preprocess failed: %d\n", ret);
    return ret;
  }
  ret = CVI_NN_Forward(mp_model_handle, mp_input_tensors, m_input_num, mp_output_tensors,
                       m_output_num);
  if (ret != CVI_RC_SUCCESS) {
    printf("NN forward failed: %d\n", ret);
    return ret;
  }
  ret = postProcessing(mp_model_handle, stFrame, mp_output_tensors, m_output_num);
  if (ret != CVI_RC_SUCCESS) {
    printf("Postprocess failed: %d\n", ret);
    return ret;
  }
  return ret;
}

bool Core::isModelInputInfoValid(const std::vector<ModelInputInfo> &v_mii) {
  if (v_mii.empty()) {
    printf("Vector of ModelInputInfo cannot be empty.\n");
    return false;
  }
  for (size_t i = 0; i < v_mii.size(); i++) {
    if (v_mii[i].shape.dim_size != 4) {
      printf("Currently does not supprt dim_size != 4. Idx %u\n", (uint32_t)i);
      return false;
    }
    uint32_t c = v_mii[i].shape.dim[1];
    if (c != v_mii[i].v_qi.size()) {
      printf("Quantize info size not equal to channel size.\n (%d vs %u)", c,
             (uint32_t)v_mii[i].v_qi.size());
      return false;
    }
  }
  return true;
}

}  // namespace cviai