#include "core.hpp"
#include "utils/vpss_helper.h"

#include <cstdlib>

namespace cviai {

int Core::modelOpen(const char *filepath) {
  CVI_RC ret = CVI_NN_RegisterModel(filepath, &mp_model_handle);
  if (ret != CVI_RC_SUCCESS) {
    printf("CVI_NN_RegisterModel failed, err %d\n", ret);
    return ret;
  }
  printf("CVI_NN_RegisterModel successed\n");
  if (!mp_config) {
    printf("config not set\n");
    return CVI_RC_FAILURE;
  }
  if (mp_config->batch_size != 0) {
    CVI_NN_SetConfig(mp_model_handle, OPTION_BATCH_SIZE, mp_config->batch_size);
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
  ret = initAfterModelOpened();
  return ret;
}

int Core::modelClose() {
  if (mp_model_handle != nullptr) {
    if (int ret = CVI_NN_CleanupModel(mp_model_handle) != CVI_RC_SUCCESS) {
      printf("CVI_NN_CleanupModel failed, err %d\n", ret);
      return ret;
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
    if (int ret = CVI_NN_SetTensorWithVideoFrame(mp_input_tensors, &info) != CVI_RC_SUCCESS) {
      printf("NN set tensor with vi failed: %d\n", ret);
      return ret;
    }
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

int Core::setVpssEngine(VpssEngine *engine) {
  if (mp_vpss_inst != nullptr) {
    printf("Vpss engine instance already set.\n");
    return CVI_RC_FAILURE;
  }
  mp_vpss_inst = engine;
  return CVI_RC_SUCCESS;
}

float Core::getInputScale() { return m_input_scale; }

int Core::getScaleFrame(VIDEO_FRAME_INFO_S *frame, VPSS_CHN chn, VPSS_CHN_ATTR_S chnAttr,
                        VIDEO_FRAME_INFO_S *outFrame) {
  VPSS_GRP_ATTR_S vpss_grp_attr;
  VPSS_GRP_DEFAULT_HELPER(&vpss_grp_attr, frame->stVFrame.u32Width, frame->stVFrame.u32Height,
                          frame->stVFrame.enPixelFormat);
  int ret = CVI_VPSS_SetGrpAttr(mp_vpss_inst->getGrpId(), &vpss_grp_attr);

  ret = CVI_VPSS_SetChnAttr(mp_vpss_inst->getGrpId(), chn,
                            &chnAttr);
  if (ret != CVI_SUCCESS) {
    printf("CVI_VPSS_SetChnAttr failed with %#x\n", ret);
    return ret;
  }

  CVI_VPSS_SendFrame(mp_vpss_inst->getGrpId(), frame, -1);

  ret = CVI_VPSS_GetChnFrame(mp_vpss_inst->getGrpId(), chn, outFrame, 100);
  if (ret != CVI_SUCCESS) {
    printf("CVI_VPSS_GetChnFrame failed with %#x\n", ret);
    return ret;
  }

  return CVI_SUCCESS;
}

}  // namespace cviai