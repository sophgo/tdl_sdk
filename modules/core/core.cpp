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
  CVI_NN_SetConfig(mp_model_handle, OPTION_INPUT_MEM_TYPE, mp_config->input_mem_type);
  CVI_NN_SetConfig(mp_model_handle, OPTION_OUTPUT_MEM_TYPE, mp_config->output_mem_type);

  ret = CVI_NN_GetInputOutputTensors(mp_model_handle, &mp_input_tensors, &m_input_num,
                                     &mp_output_tensors, &m_output_num);
  if (ret != CVI_RC_SUCCESS) {
    printf("CVI_NN_GetINputsOutputs failed\n");
    return ret;
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

int Core::run(VIDEO_FRAME_INFO_S *srcFrame) {
  if (mp_config->input_mem_type == 2) {
    // FIXME: Need to support multi-input and different fmt
    CVI_TENSOR *input = getInputTensor(0);
    CVI_VIDEO_FRAME_INFO info;
    info.type = CVI_FRAME_PLANAR;
    info.shape.dim_size = input->shape.dim_size;
    info.shape.dim[0] = input->shape.dim[0];
    info.shape.dim[1] = input->shape.dim[1];
    info.shape.dim[2] = input->shape.dim[2];
    info.shape.dim[3] = input->shape.dim[3];
    info.fmt = CVI_FMT_INT8;
    for (size_t i = 0; i < 3; ++i) {
      info.stride[i] = srcFrame->stVFrame.u32Stride[i];
      info.pyaddr[i] = srcFrame->stVFrame.u64PhyAddr[i];
    }
    CVI_NN_SetTensorWithVideoFrame(mp_input_tensors, &info);
  }
  int ret = CVI_NN_Forward(mp_model_handle, mp_input_tensors, m_input_num, mp_output_tensors,
                           m_output_num);
  if (ret != CVI_RC_SUCCESS) {
    printf("NN forward failed: %d\n", ret);
  }
  return ret;
}

CVI_TENSOR *Core::getInputTensor(int idx) {
  if (idx >= m_input_num) {
    return NULL;
  }
  return mp_input_tensors + idx;
}

CVI_TENSOR *Core::getOutputTensor(int idx) {
  if (idx >= m_output_num) {
    return NULL;
  }
  return mp_output_tensors + idx;
}

}  // namespace cviai