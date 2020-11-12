#include "core.hpp"
#include "core/utils/vpss_helper.h"
#include "cviai_trace.hpp"

namespace cviai {

int Core::modelOpen(const char *filepath) {
  TRACE_EVENT("cviai_core", "Core::modelOpen");
  if (!mp_mi) {
    LOGE("config not set\n");
    return CVI_FAILURE;
  }

  CVI_RC ret = CVI_NN_RegisterModel(filepath, &mp_mi->handle);
  if (ret != CVI_RC_SUCCESS) {
    LOGE("CVI_NN_RegisterModel failed, err %d\n", ret);
    modelClose();
    return CVI_FAILURE;
  }
  LOGI("CVI_NN_RegisterModel successed\n");

  if (mp_mi->conf.batch_size != 0) {
    CVI_NN_SetConfig(mp_mi->handle, OPTION_BATCH_SIZE, mp_mi->conf.batch_size);
  }
  CVI_NN_SetConfig(mp_mi->handle, OPTION_OUTPUT_ALL_TENSORS,
                   static_cast<int>(mp_mi->conf.debug_mode));
  CVI_NN_SetConfig(mp_mi->handle, OPTION_SKIP_PREPROCESS, static_cast<int>(true));
  CVI_NN_SetConfig(mp_mi->handle, OPTION_SKIP_POSTPROCESS,
                   static_cast<int>(mp_mi->conf.skip_postprocess));

  ret = CVI_NN_GetInputOutputTensors(mp_mi->handle, &mp_mi->in.tensors, &mp_mi->in.num,
                                     &mp_mi->out.tensors, &mp_mi->out.num);
  if (ret != CVI_RC_SUCCESS) {
    LOGE("CVI_NN_GetINputsOutputs failed\n");
    modelClose();
    return CVI_FAILURE;
  }
  TRACE_EVENT_BEGIN("cviai_core", "InitAtferModelOpened");
  CVI_TENSOR *input =
      CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, mp_mi->in.tensors, mp_mi->in.num);
  // Assigning default values.
  std::vector<initSetup> data(mp_mi->in.num);
  for (uint32_t i = 0; i < (uint32_t)mp_mi->in.num; i++) {
    CVI_TENSOR *tensor = mp_mi->in.tensors + i;
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
  for (uint32_t i = 0; i < (uint32_t)mp_mi->in.num; i++) {
    if (data[i].use_quantize_scale) {
      CVI_TENSOR *tensor = mp_mi->in.tensors + i;
      float quant_scale = CVI_NN_TensorQuantScale(tensor);
      for (uint32_t j = 0; j < 3; j++) {
        data[i].factor[j] *= quant_scale;
        data[i].mean[j] *= quant_scale;
      }
      // FIXME: Behavior will changed in 1822.
      float factor_limit = 8191.f / 8192;
      for (uint32_t j = 0; j < 3; j++) {
        if (data[i].factor[j] > factor_limit) {
          LOGW("factor[%d] is bigger than limit: %f\n", i, data[i].factor[j]);
          data[i].factor[j] = factor_limit;
        }
      }
    }
    VPSSConfig vcfg;
    int32_t width, height;
    PIXEL_FORMAT_E format;
    // FIXME: Future support for nhwc input. Currently disabled.
    if (false) {
      width = input->shape.dim[2];
      height = input->shape.dim[1];
      format = PIXEL_FORMAT_RGB_888;
      vcfg.frame_type = CVI_FRAME_PACKAGE;
    } else {
      width = input->shape.dim[3];
      height = input->shape.dim[2];
      format = PIXEL_FORMAT_RGB_888_PLANAR;
      vcfg.frame_type = CVI_FRAME_PLANAR;
    }
    vcfg.rescale_type = data[i].rescale_type;
    vcfg.crop_attr.bEnable = data[i].use_crop;
    VPSS_CHN_SQ_HELPER(&vcfg.chn_attr, width, height, format, data[i].factor, data[i].mean,
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
  if (mp_mi->handle != nullptr) {
    if (int ret = CVI_NN_CleanupModel(mp_mi->handle) != CVI_RC_SUCCESS) {  // NOLINT
      LOGE("CVI_NN_CleanupModel failed, err %d\n", ret);
      return CVI_FAILURE;
    }
    mp_mi->handle = nullptr;
  }
  return CVI_SUCCESS;
}

CVI_TENSOR *Core::getInputTensor(int idx) {
  if (idx >= mp_mi->in.num) {
    return NULL;
  }
  return mp_mi->in.tensors + idx;
}

CVI_TENSOR *Core::getOutputTensor(int idx) {
  if (idx >= mp_mi->out.num) {
    return NULL;
  }
  return mp_mi->out.tensors + idx;
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

int Core::getChnAttribute(const uint32_t width, const uint32_t height, const uint32_t idx,
                          VPSS_CHN_ATTR_S *attr) {
  if (!m_export_chn_attr) {
    LOGE("This model does not support exporting channel attributes.\n");
    return CVI_FAILURE;
  }
  if (idx >= (uint32_t)mp_mi->in.num) {
    LOGE("Input index exceed input tensor num.\n");
    return CVI_FAILURE;
  }
  if (!m_skip_vpss_preprocess) {
    LOGW("VPSS preprocessing is enabled. Remember to skip vpss preprocess.\n");
  }
  switch (m_vpss_config[idx].rescale_type) {
    case RESCALE_CENTER: {
      *attr = m_vpss_config[idx].chn_attr;
    } break;
    case RESCALE_RB: {
      CVI_TENSOR *input = mp_mi->in.tensors + idx;
      auto &factor = m_vpss_config[idx].chn_attr.stNormalize.factor;
      auto &mean = m_vpss_config[idx].chn_attr.stNormalize.mean;
      VPSS_CHN_SQ_RB_HELPER(attr, width, height, input->shape.dim[3], input->shape.dim[2],
                            m_vpss_config[idx].chn_attr.enPixelFormat, factor, mean, false);
      attr->stAspectRatio.u32BgColor = m_vpss_config[idx].chn_attr.stAspectRatio.u32BgColor;
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
bool Core::isInitialized() { return mp_mi->handle == nullptr ? false : true; }

int Core::initAfterModelOpened(std::vector<initSetup> *data) { return CVI_SUCCESS; }

int Core::vpssPreprocess(const std::vector<VIDEO_FRAME_INFO_S *> &srcFrames,
                         std::vector<std::shared_ptr<VIDEO_FRAME_INFO_S>> *dstFrames) {
  int ret = CVI_SUCCESS;
  for (uint32_t i = 0; i < (uint32_t)srcFrames.size(); i++) {
    if (!m_vpss_config[i].crop_attr.bEnable) {
      mp_vpss_inst->sendFrame(srcFrames[i], &m_vpss_config[i].chn_attr, 1);
    } else {
      mp_vpss_inst->sendCropChnFrame(srcFrames[i], &m_vpss_config[i].crop_attr,
                                     &m_vpss_config[i].chn_attr, 1);
    }
    ret |= mp_vpss_inst->getFrame((*dstFrames)[i].get(), 0);
  }
  return ret;
}

int Core::run(std::vector<VIDEO_FRAME_INFO_S *> &frames) {
  TRACE_EVENT("cviai_core", "Core::run");
  int ret = CVI_SUCCESS;
  std::vector<std::shared_ptr<VIDEO_FRAME_INFO_S>> dstFrames;
  if (mp_mi->conf.input_mem_type == CVI_MEM_DEVICE) {
    if (m_skip_vpss_preprocess) {
      ret |= registerFrame2Tensor(frames);
    } else {
      if (m_vpss_config.size() != frames.size()) {
        LOGE("The size of vpss config does not match the number of frames. (%zu vs %zu)\n",
             m_vpss_config.size(), frames.size());
        return CVI_FAILURE;
      }
      dstFrames.resize(frames.size(), {new VIDEO_FRAME_INFO_S, [this](VIDEO_FRAME_INFO_S *f) {
                                         this->mp_vpss_inst->releaseFrame(f, 0);
                                         delete f;
                                       }});
      ret |= vpssPreprocess(frames, &dstFrames);
      ret |= registerFrame2Tensor(dstFrames);
    }
  }
  if (int rcret = CVI_NN_Forward(mp_mi->handle, mp_mi->in.tensors, mp_mi->in.num,  // NOLINT
                                 mp_mi->out.tensors, mp_mi->out.num) != CVI_RC_SUCCESS) {
    LOGE("NN forward failed: %d\n", rcret);
    ret |= CVI_FAILURE;
  }
  return ret;
}

template <typename T>
int Core::registerFrame2Tensor(std::vector<T> &frames) {
  int ret = CVI_SUCCESS;
  std::vector<uint64_t> paddrs;
  for (uint32_t i = 0; i < (uint32_t)frames.size(); i++) {
    T frame = frames[i];
    switch (frame->stVFrame.enPixelFormat) {
      case PIXEL_FORMAT_RGB_888_PLANAR:
        paddrs.push_back(frame->stVFrame.u64PhyAddr[0]);
        paddrs.push_back(frame->stVFrame.u64PhyAddr[1]);
        paddrs.push_back(frame->stVFrame.u64PhyAddr[2]);
        break;
      default:
        LOGE("Unsupported image type: %x.\n", frame->stVFrame.enPixelFormat);
        return CVI_FAILURE;
    }
  }
  if (int ret =
          CVI_NN_FeedTensorWithFrames(mp_mi->handle, mp_mi->in.tensors, m_vpss_config[0].frame_type,
                                      CVI_FMT_INT8, paddrs.size(), paddrs.data(),
                                      frames[0]->stVFrame.u32Height, frames[0]->stVFrame.u32Width,
                                      frames[0]->stVFrame.u32Stride[0]) != CVI_RC_SUCCESS) {
    LOGE("NN set tensor with vi failed: %d\n", ret);
    return CVI_FAILURE;
  }
  return ret;
}

}  // namespace cviai