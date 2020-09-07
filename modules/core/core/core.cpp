#include "core.hpp"
#include "core/utils/vpss_helper.h"
#include "core_utils.hpp"

#include "tracer.h"

#include <cstdlib>

namespace cviai {

int Core::modelOpen(const char *filepath) {
  ScopedTrace st(__func__);
  CVI_RC ret = CVI_NN_RegisterModel(filepath, &mp_model_handle);
  if (ret != CVI_RC_SUCCESS) {
    printf("CVI_NN_RegisterModel failed, err %d\n", ret);
    return CVI_FAILURE;
  }
  printf("CVI_NN_RegisterModel successed\n");
  if (!mp_config) {
    printf("config not set\n");
    return CVI_FAILURE;
  }
  if (mp_config->batch_size != 0) {
    CVI_NN_SetConfig(mp_model_handle, OPTION_BATCH_SIZE, mp_config->batch_size);
  }
  CVI_NN_SetConfig(mp_model_handle, OPTION_PREPARE_BUF_FOR_INPUTS,
                   static_cast<int>(mp_config->init_input_buffer));
  CVI_NN_SetConfig(mp_model_handle, OPTION_PREPARE_BUF_FOR_OUTPUTS,
                   static_cast<int>(mp_config->init_output_buffer));
  CVI_NN_SetConfig(mp_model_handle, OPTION_OUTPUT_ALL_TENSORS,
                   static_cast<int>(mp_config->debug_mode));
  CVI_NN_SetConfig(mp_model_handle, OPTION_SKIP_PREPROCESS,
                   static_cast<int>(mp_config->skip_preprocess));
  CVI_NN_SetConfig(mp_model_handle, OPTION_SKIP_POSTPROCESS,
                   static_cast<int>(mp_config->skip_postprocess));
  CVI_NN_SetConfig(mp_model_handle, OPTION_INPUT_MEM_TYPE, mp_config->input_mem_type);
  CVI_NN_SetConfig(mp_model_handle, OPTION_OUTPUT_MEM_TYPE, mp_config->output_mem_type);

  ret = CVI_NN_GetInputOutputTensors(mp_model_handle, &mp_input_tensors, &m_input_num,
                                     &mp_output_tensors, &m_output_num);
  if (ret != CVI_RC_SUCCESS) {
    printf("CVI_NN_GetINputsOutputs failed\n");
    return CVI_FAILURE;
    ;
  }
  Tracer::TraceBegin("InitAtferModelOpened");
  ret = initAfterModelOpened();
  Tracer::TraceEnd();
  return ret;
}

int Core::modelClose() {
  ScopedTrace st(__func__);
  if (mp_model_handle != nullptr) {
    if (int ret = CVI_NN_CleanupModel(mp_model_handle) != CVI_RC_SUCCESS) {  // NOLINT
      printf("CVI_NN_CleanupModel failed, err %d\n", ret);
      return CVI_FAILURE;
    }
  }
  return CVI_SUCCESS;
}

int Core::vpssPreprocess(const VIDEO_FRAME_INFO_S *srcFrame, VIDEO_FRAME_INFO_S *dstFrame) {
  if (!m_use_vpss_crop) {
    mp_vpss_inst->sendFrame(srcFrame, &m_vpss_chn_attr[0], 1);
  } else {
    mp_vpss_inst->sendCropChnFrame(srcFrame, &m_crop_attr, &m_vpss_chn_attr[0], 1);
  }

  return mp_vpss_inst->getFrame(dstFrame, 0);
}

int Core::run(VIDEO_FRAME_INFO_S *srcFrame) {
  ScopedTrace st(__func__);
  int ret = CVI_SUCCESS;
  if (mp_config->input_mem_type == 2) {
    if (m_skip_vpss_preprocess) {
      ret |= runVideoForward(srcFrame);
    } else {
      VIDEO_FRAME_INFO_S stDstFrame;
      ret |= vpssPreprocess(srcFrame, &stDstFrame);
      ret |= runVideoForward(&stDstFrame);
      ret |= mp_vpss_inst->releaseFrame(&stDstFrame, 0);
    }
  } else {
    if (int rcret = CVI_NN_Forward(mp_model_handle, mp_input_tensors, m_input_num,  // NOLINT
                                   mp_output_tensors, m_output_num) != CVI_RC_SUCCESS) {
      printf("NN forward failed: %d\n", rcret);
      ret |= CVI_FAILURE;
    }
  }

  return ret;
}

int Core::runVideoForward(VIDEO_FRAME_INFO_S *srcFrame) {
  // FIXME: Need to support multi-input and different fmt
  int ret = CVI_SUCCESS;
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
    if (m_reverse_device_mem) {
      info.stride[i] = srcFrame->stVFrame.u32Stride[2 - i];
      info.pyaddr[i] = srcFrame->stVFrame.u64PhyAddr[2 - i];
    } else {
      info.stride[i] = srcFrame->stVFrame.u32Stride[i];
      info.pyaddr[i] = srcFrame->stVFrame.u64PhyAddr[i];
    }
  }
  if (int ret =
          CVI_NN_SetTensorWithVideoFrame(mp_model_handle, mp_input_tensors, &info) !=  // NOLINT
          CVI_RC_SUCCESS) {
    printf("NN set tensor with vi failed: %d\n", ret);
    return CVI_FAILURE;
  }
  if (int rcret = CVI_NN_Forward(mp_model_handle, mp_input_tensors, m_input_num,  // NOLINT
                                 mp_output_tensors, m_output_num) != CVI_RC_SUCCESS) {
    printf("NN forward failed: %d\n", rcret);
    ret |= CVI_FAILURE;
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

int Core::setIveInstance(IVE_HANDLE handle) {
  ive_handle = handle;
  return CVI_SUCCESS;
}

int Core::setVpssEngine(VpssEngine *engine) {
  mp_vpss_inst = engine;
  return CVI_SUCCESS;
}

void Core::skipVpssPreprocess(bool skip) { m_skip_vpss_preprocess = skip; }
void Core::setModelThreshold(float threshold) { m_model_threshold = threshold; }

}  // namespace cviai