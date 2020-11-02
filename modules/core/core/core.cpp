#include "core.hpp"
#include "core/utils/vpss_helper.h"
#include "cviai_trace.hpp"

namespace cviai {

int Core::modelOpen(const char *filepath) {
  TRACE_EVENT("cviai_core", "Core::modelOpen");
  if (!mp_config) {
    LOGE("config not set\n");
    return CVI_FAILURE;
  }
  CVI_RC ret = CVI_NN_RegisterModel(filepath, &mp_model_handle);
  if (ret != CVI_RC_SUCCESS) {
    LOGE("CVI_NN_RegisterModel failed, err %d\n", ret);
    modelClose();
    return CVI_FAILURE;
  }
  LOGI("CVI_NN_RegisterModel successed\n");
  if (mp_config->batch_size != 0) {
    CVI_NN_SetConfig(mp_model_handle, OPTION_BATCH_SIZE, mp_config->batch_size);
  }
  CVI_NN_SetConfig(mp_model_handle, OPTION_OUTPUT_ALL_TENSORS,
                   static_cast<int>(mp_config->debug_mode));
  // For models before version 1.2
  bool skip_preprocess = (mp_config->input_mem_type == CVI_MEM_DEVICE) ? true : false;
  CVI_NN_SetConfig(mp_model_handle, OPTION_SKIP_PREPROCESS, static_cast<int>(skip_preprocess));
  CVI_NN_SetConfig(mp_model_handle, OPTION_SKIP_POSTPROCESS,
                   static_cast<int>(mp_config->skip_postprocess));

  ret = CVI_NN_GetInputOutputTensors(mp_model_handle, &mp_input_tensors, &m_input_num,
                                     &mp_output_tensors, &m_output_num);
  if (ret != CVI_RC_SUCCESS) {
    LOGE("CVI_NN_GetINputsOutputs failed\n");
    modelClose();
    return CVI_FAILURE;
  }
  TRACE_EVENT_BEGIN("cviai_core", "InitAtferModelOpened");
  CVI_TENSOR *input = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, mp_input_tensors, m_input_num);
  // Assigning default values.
  std::vector<initSetup> data(m_input_num);
  for (uint32_t i = 0; i < (uint32_t)m_input_num; i++) {
    CVI_TENSOR *tensor = mp_input_tensors + i;
    float quant_scale = CVI_NN_TensorQuantScale(tensor);
    data[i].use_quantize_scale = quant_scale == 0 ? false : true;
  }
  ret = initAfterModelOpened(&data);
  if (ret != CVI_SUCCESS) {
    LOGE("Failed to init after open model.\n");
    modelClose();
    return CVI_FAILURE;
  }
  m_vpss_config.clear();
  for (uint32_t i = 0; i < (uint32_t)m_input_num; i++) {
    if (data[i].use_quantize_scale) {
      CVI_TENSOR *tensor = mp_input_tensors + i;
      float quant_scale = CVI_NN_TensorQuantScale(tensor);
      for (uint32_t j = 0; j < 3; j++) {
        data[i].factor[j] *= quant_scale;
        data[i].mean[j] *= quant_scale;
      }
      // FIXME: Behavior will changed in 1822.
      float factor_limit = 8191.f / 8192;
      for (uint32_t j = 0; j < 3; j++) {
        if (data[i].factor[j] > factor_limit) {
          data[i].factor[j] = factor_limit;
        }
      }
    }
    VPSSConfig vcfg;
    vcfg.rescale_type = data[i].rescale_type;
    vcfg.crop_attr.bEnable = data[i].use_crop;
    VPSS_CHN_SQ_HELPER(&vcfg.chn_attr, input->shape.dim[3], input->shape.dim[2],
                       PIXEL_FORMAT_RGB_888_PLANAR, data[i].factor, data[i].mean,
                       data[i].pad_reverse);
    if (!data[i].keep_aspect_ratio) {
      vcfg.chn_attr.stAspectRatio.enMode = ASPECT_RATIO_NONE;
    }
    m_vpss_config.push_back(vcfg);
  }
  TRACE_EVENT_END("cviai_core");
  return CVI_SUCCESS;
}

int Core::modelClose() {
  TRACE_EVENT("cviai_core", "Core::modelClose");
  if (mp_model_handle != nullptr) {
    if (int ret = CVI_NN_CleanupModel(mp_model_handle) != CVI_RC_SUCCESS) {  // NOLINT
      LOGE("CVI_NN_CleanupModel failed, err %d\n", ret);
      return CVI_FAILURE;
    }
    mp_model_handle = nullptr;
  }
  return CVI_SUCCESS;
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

int Core::getChnAttribute(const uint32_t width, const uint32_t height, VPSS_CHN_ATTR_S *attr) {
  if (!m_export_chn_attr) {
    return CVI_FAILURE;
  }
  if (!m_skip_vpss_preprocess) {
    LOGW("VPSS preprocessing is enabled. Remember to skip vpss preprocess.\n");
  }
  switch (m_vpss_config[0].rescale_type) {
    case RESCALE_CENTER: {
      *attr = m_vpss_config[0].chn_attr;
    } break;
    case RESCALE_RB: {
      CVI_TENSOR *input =
          CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, mp_input_tensors, m_input_num);
      auto &factor = m_vpss_config[0].chn_attr.stNormalize.factor;
      auto &mean = m_vpss_config[0].chn_attr.stNormalize.mean;
      VPSS_CHN_SQ_RB_HELPER(attr, width, height, input->shape.dim[3], input->shape.dim[2],
                            PIXEL_FORMAT_RGB_888_PLANAR, factor, mean, false);
      attr->stAspectRatio.u32BgColor = m_vpss_config[0].chn_attr.stAspectRatio.u32BgColor;
    } break;
    default: {
      LOGW("Unsupported rescale type.\n");
      return CVI_FAILURE;
    } break;
  }
  return CVI_SUCCESS;
}

void Core::setModelThreshold(float threshold) { m_model_threshold = threshold; }
float Core::getModelThreshold() { return m_model_threshold; };
bool Core::isInitialized() { return mp_model_handle == nullptr ? false : true; }

int Core::initAfterModelOpened(std::vector<initSetup> *data) { return CVI_SUCCESS; }

int Core::vpssPreprocess(const VIDEO_FRAME_INFO_S *srcFrame, VIDEO_FRAME_INFO_S *dstFrame) {
  if (!m_vpss_config[0].crop_attr.bEnable) {
    mp_vpss_inst->sendFrame(srcFrame, &m_vpss_config[0].chn_attr, 1);
  } else {
    mp_vpss_inst->sendCropChnFrame(srcFrame, &m_vpss_config[0].crop_attr,
                                   &m_vpss_config[0].chn_attr, 1);
  }

  return mp_vpss_inst->getFrame(dstFrame, 0);
}

int Core::run(VIDEO_FRAME_INFO_S *srcFrame) {
  TRACE_EVENT("cviai_core", "Core::run");
  int ret = CVI_SUCCESS;
  if (mp_config->input_mem_type == CVI_MEM_DEVICE) {
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
      LOGE("NN forward failed: %d\n", rcret);
      ret |= CVI_FAILURE;
    }
  }

  return ret;
}

int Core::runVideoForward(VIDEO_FRAME_INFO_S *srcFrame) {
  int ret = CVI_SUCCESS;
  if (int ret =
          CVI_NN_FeedTensorWithFrames(mp_model_handle, mp_input_tensors, CVI_FRAME_PLANAR,
                                      CVI_FMT_INT8, 3, srcFrame->stVFrame.u64PhyAddr,
                                      srcFrame->stVFrame.u32Height, srcFrame->stVFrame.u32Width,
                                      srcFrame->stVFrame.u32Stride[0]) != CVI_RC_SUCCESS) {
    LOGE("NN set tensor with vi failed: %d\n", ret);
    return CVI_FAILURE;
  }
  if (int rcret = CVI_NN_Forward(mp_model_handle, mp_input_tensors, m_input_num, mp_output_tensors,
                                 m_output_num) != CVI_RC_SUCCESS) {
    LOGE("NN forward failed: %d\n", rcret);
    ret |= CVI_FAILURE;
  }
  return ret;
}

}  // namespace cviai