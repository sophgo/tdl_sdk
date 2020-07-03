#include "core.hpp"

namespace cviai {

bool core::modelOpen(const char *filepath, ModelConfig *config) {
  CVI_RC ret = CVI_NN_RegisterModel(filepath, &mp_model_handle);
  if (ret != CVI_RC_SUCCESS) {
      printf("CVI_NN_RegisterModel failed, err %d\n", ret);
      return ret;
  }
  printf("CVI_NN_RegisterModel successed\n");
  if (config != nullptr) {
    CVI_NN_SetConfig(mp_model_handle, OPTION_PREPARE_BUF_FOR_INPUTS, config->init_input_buffer);
    CVI_NN_SetConfig(mp_model_handle, OPTION_PREPARE_BUF_FOR_OUTPUTS, config->init_output_buffer);
    CVI_NN_SetConfig(mp_model_handle, OPTION_OUTPUT_ALL_TENSORS, config->debug_mode);
    CVI_NN_SetConfig(mp_model_handle, OPTION_SKIP_PREPROCESS, config->skip_preprocess);
    CVI_NN_SetConfig(mp_model_handle, OPTION_SKIP_POSTPROCESS, config->skip_postprocess);
  }
  ret = CVI_NN_GetInputOutputTensors(mp_model_handle, &mp_input_tensors, &m_input_num,
                                     &mp_output_tensors, &m_output_num);
  if (ret != CVI_RC_SUCCESS) {
      printf("CVI_NN_GetINputsOutputs failed\n");
      return ret;
  }
  return ret;
}

bool core::modelClose() {
  if (mp_model_handle != nullptr) {
    int ret = CVI_NN_CleanupModel(mp_model_handle);
    if (ret != CVI_RC_SUCCESS) {
      printf("CVI_NN_CleanupModel failed, err %d\n", ret);
      return ret;
    }
  }
  return CVI_RC_FAILURE;
}

}