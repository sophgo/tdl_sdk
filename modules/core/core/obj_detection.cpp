#include "obj_detection.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iterator>
#include "core/utils/vpss_helper.h"
#include "error_msg.hpp"

namespace cvitdl {

DetectionBase::DetectionBase() : Core(CVI_MEM_DEVICE) {}
int DetectionBase::vpssPreprocess(VIDEO_FRAME_INFO_S *srcFrame, VIDEO_FRAME_INFO_S *dstFrame,
                                  VPSSConfig &vpss_config) {
  int ret;
  LOGI("to vpss preprocess,crop_enable:%d\n", (int)vpss_config.crop_attr.bEnable);
  if (!vpss_config.crop_attr.bEnable) {
    ret = mp_vpss_inst->sendFrame(srcFrame, &vpss_config.chn_attr, &vpss_config.chn_coeff, 1);
  } else {
    ret = mp_vpss_inst->sendCropChnFrame(srcFrame, &vpss_config.crop_attr, &vpss_config.chn_attr,
                                         &vpss_config.chn_coeff, 1);
  }
  if (ret != CVI_SUCCESS) {
    LOGE("Send frame failed: %s!\n", get_vpss_error_msg(ret));
    return CVI_TDL_ERR_VPSS_SEND_FRAME;
  }

  ret = mp_vpss_inst->getFrame(dstFrame, 0, m_vpss_timeout);
  if (ret != CVI_SUCCESS) {
    LOGE("Get frame failed: %s!\n", get_vpss_error_msg(ret));
    return CVI_TDL_ERR_VPSS_GET_FRAME;
  }
  return 0;
}

void DetectionBase::set_algparam(const cvtdl_det_algo_param_t &alg_param) {
  alg_param_.anchor_len = alg_param.anchor_len;
  alg_param_.stride_len = alg_param.stride_len;
  alg_param_.cls = alg_param.cls;

  uint32_t *anchors = new uint32_t[alg_param.anchor_len];
  for (int i = 0; i < alg_param.anchor_len; i++) {
    anchors[i] = alg_param.anchors[i];
  }
  alg_param_.anchors = anchors;

  uint32_t *strides = new uint32_t[alg_param.stride_len];
  for (int i = 0; i < alg_param.stride_len; i++) {
    strides[i] = alg_param.strides[i];
  }
  alg_param_.strides = strides;
}

int DetectionBase::setupInputPreprocess(std::vector<InputPreprecessSetup> *data) {
  if (data->size() != 1) {
    LOGE("Detection model only has 1 input.\n");
    return CVI_TDL_ERR_INVALID_ARGS;
  }

  for (int i = 0; i < 3; i++) {
    (*data)[0].factor[i] = preprocess_param_.factor[i];
    (*data)[0].mean[i] = preprocess_param_.mean[i];
  }

  (*data)[0].format = preprocess_param_.format;
  // (*data)[0].use_quantize_scale = true; // #parse from model
  return CVI_TDL_SUCCESS;
}
// namespace cvitdl
}  // namespace cvitdl
