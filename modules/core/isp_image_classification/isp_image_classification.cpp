#include "isp_image_classification.hpp"
#include <core/core/cvtdl_errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <error_msg.hpp>
#include <iostream>
#include <numeric>
#include <vector>
#include "core/core/cvtdl_errno.h"
#include "core/cvi_tdl_types_mem.h"
#include "core/utils/vpss_helper.h"
#include "core_utils.hpp"
#include "cvi_sys.h"
#include <unistd.h>

#define topK 5

#define ENABLE_TEAISP_PQ_DUMP 0

#if ENABLE_TEAISP_PQ_DUMP
static bool dump_aipq = false;

static void dump_input_raw(const char *name, uint64_t phy_addr, size_t size) {
  void *data = CVI_SYS_MmapCache(phy_addr, size);
  FILE *fp = fopen(name, "wb");
  if (fp == NULL) {
    printf("Failed to open file %s\n", name);
    return;
  }
  fwrite(data, 1, size, fp);
  fflush(fp);
  fclose(fp);
  CVI_SYS_Munmap(data, size);
}

static void dump_input_output(const char *name, void *data, size_t size) {
  FILE *fp = fopen(name, "wb");
  if (fp == NULL) {
    printf("Failed to open file %s\n", name);
    return;
  }
  fwrite(data, 1, size, fp);
  fflush(fp);
  fclose(fp);
}
#endif

namespace cvitdl {

IspImageClassification::IspImageClassification() : Core(CVI_MEM_DEVICE) {
  float mean[3] = {123.675, 116.28, 103.52};
  float std[3] = {58.395, 57.12, 57.375};

  for (int i = 0; i < 3; i++) {
    m_preprocess_param[0].mean[i] = mean[i] / std[i];
    m_preprocess_param[0].factor[i] = 1.0 / std[i];
  }

  m_preprocess_param[0].format = PIXEL_FORMAT_RGB_888_PLANAR;
  m_preprocess_param[0].rescale_type = RESCALE_CENTER;
  m_preprocess_param[0].keep_aspect_ratio = true;
}

#ifdef __CV186X__
int IspImageClassification::onModelOpened() {
  // if (getNumOutputTensor() != 1) {
  //   LOGE("IspImageClassification only expected 1 output branch!\n");
  // }
  // 分配设备内存
  bm_malloc_device_byte(bm_handle, &(getModelInfo()->in.tensors[1].device_mem), 3 * sizeof(float));
  return CVI_TDL_SUCCESS;
}

int IspImageClassification::onModelClosed() {
  // if (getNumOutputTensor() != 1) {
  //   LOGE("IspImageClassification only expected 1 output branch!\n");
  // }

  // 释放设备内存
  // bm_free_device(bm_handle,getModelInfo()->in.tensors[1].device_mem);

  return CVI_TDL_SUCCESS;
}
#endif

std::vector<int> IspImageClassification::TopKIndex(std::vector<float> &vec, int topk) {
  std::vector<int> topKIndex;
  topKIndex.clear();

  std::vector<size_t> vec_index(vec.size());
  std::iota(vec_index.begin(), vec_index.end(), 0);

  std::sort(vec_index.begin(), vec_index.end(),
            [&vec](size_t index_1, size_t index_2) { return vec[index_1] > vec[index_2]; });

  int k_num = std::min<int>(vec.size(), topk);

  for (int i = 0; i < k_num; ++i) {
    topKIndex.emplace_back(vec_index[i]);
  }

  return topKIndex;
}

IspImageClassification::~IspImageClassification() {}

int IspImageClassification::vpssPreprocess(VIDEO_FRAME_INFO_S *srcFrame,
                                           VIDEO_FRAME_INFO_S *dstFrame, VPSSConfig &vpss_config) {
  auto &vpssChnAttr = vpss_config.chn_attr;
  auto &factor = vpssChnAttr.stNormalize.factor;
  auto &mean = vpssChnAttr.stNormalize.mean;
  vpssChnAttr.stNormalize.bEnable = false;
  vpssChnAttr.stAspectRatio.enMode = ASPECT_RATIO_NONE;

  VPSS_CHN_SQ_RB_HELPER(&vpssChnAttr, srcFrame->stVFrame.u32Width, srcFrame->stVFrame.u32Height,
                        vpssChnAttr.u32Width, vpssChnAttr.u32Height, PIXEL_FORMAT_RGB_888_PLANAR,
                        factor, mean, false);
  int ret = mp_vpss_inst->sendFrame(srcFrame, &vpssChnAttr, &vpss_config.chn_coeff, 1);
  if (ret != CVI_SUCCESS) {
    LOGE("vpssPreprocess Send frame failed: %s!\n", get_vpss_error_msg(ret));
    return CVI_TDL_ERR_VPSS_SEND_FRAME;
  }

  ret = mp_vpss_inst->getFrame(dstFrame, 0, m_vpss_timeout);
  if (ret != CVI_SUCCESS) {
    LOGE("get frame failed: %s!\n", get_vpss_error_msg(ret));
    return CVI_TDL_ERR_VPSS_GET_FRAME;
  }

  return CVI_TDL_SUCCESS;
}

int IspImageClassification::inference(VIDEO_FRAME_INFO_S *srcFrame, cvtdl_class_meta_t *cls_meta,
                                      cvtdl_isp_meta_t *isparg) {
  size_t input_num = getNumInputTensor();
  std::vector<VIDEO_FRAME_INFO_S *> frames = {srcFrame};
#if ENABLE_TEAISP_PQ_DUMP
  if (access("/tmp/dump_aipq", F_OK) == 0) {
    dump_aipq = true;
    system("rm /tmp/dump_aipq");
  }
  if (dump_aipq) {
    dump_input_raw("0-input_raw.bin", srcFrame->stVFrame.u64PhyAddr[0], srcFrame->stVFrame.u32Length[0]);
  }
#endif
#ifdef __CV186X__
  bm_memcpy_s2d_partial(bm_handle, getModelInfo()->in.tensors[1].device_mem, (void *)isparg->awb,
                        3 * sizeof(float));
  bm_status_t status =
      bm_mem_flush_device_mem(bm_handle, &(getModelInfo()->in.tensors[1].device_mem));
  assert(BM_SUCCESS == status);
#else
  const TensorInfo &tinfo = getInputTensorInfo(1); // awb
  float *input_ptr = tinfo.get<float>();
  memcpy(input_ptr, isparg->awb, 3 * sizeof(float));
#if ENABLE_TEAISP_PQ_DUMP
  if (dump_aipq) {
    dump_input_output("1-input_awb.bin", input_ptr, 3 * sizeof(float));
  }
#endif
  if (input_num == 3) { // compatible with old models
    const TensorInfo &tinfo_blc = getInputTensorInfo(2);  // blc
    input_ptr = tinfo_blc.get<float>();
    memcpy(input_ptr, &isparg->blc, 1 * sizeof(float));
#if ENABLE_TEAISP_PQ_DUMP
  if (dump_aipq) {
    dump_input_output("2-input_blc.bin", input_ptr, 1 * sizeof(float));
  }
#endif
  } else if (input_num == 4) { // compatible with old models
    const TensorInfo &tinfo_ccm = getInputTensorInfo(2);  // ccm
    input_ptr = tinfo_ccm.get<float>();
    memcpy(input_ptr, isparg->ccm, 9 * sizeof(float));
#if ENABLE_TEAISP_PQ_DUMP
  if (dump_aipq) {
    dump_input_output("2-input_ccm.bin", input_ptr, 9 * sizeof(float));
  }
#endif
    const TensorInfo &tinfo_blc = getInputTensorInfo(3);  // blc
    input_ptr = tinfo_blc.get<float>();
    memcpy(input_ptr, &isparg->blc, 1 * sizeof(float));
#if ENABLE_TEAISP_PQ_DUMP
  if (dump_aipq) {
    dump_input_output("3-input_blc.bin", input_ptr, 1 * sizeof(float));
  }
#endif
  } else if (input_num > 4) {
    LOGE("IspImageClassification input tensor number error\n");
    return -1;
  }
  srcFrame->stVFrame.enPixelFormat = PIXEL_FORMAT_RGB_888_PLANAR;
#endif
  if (srcFrame->stVFrame.u64PhyAddr[0] == 0 && hasSkippedVpssPreprocess()) {
    const TensorInfo &tinfo = getInputTensorInfo(0);
    int8_t *input_ptr = tinfo.get<int8_t>();
    printf("Copy during inference\n");
    memcpy(input_ptr, srcFrame->stVFrame.pu8VirAddr[0], srcFrame->stVFrame.u32Length[0]);
  }
  int ret = run(frames);
  if (ret != CVI_TDL_SUCCESS) {
    LOGE("IspImageClassification run inference failed\n");
    return ret;
  }

  outputParser(cls_meta);
  model_timer_.TicToc("post");
  return CVI_TDL_SUCCESS;
}

void softmax(std::vector<float> &x) {
  float max_val = *std::max_element(x.begin(), x.end());
  float sum = 0.0f;
  for (size_t i = 0; i < x.size(); i++) {
    x[i] = std::exp(x[i] - max_val);
    sum += x[i];
  }
  for (size_t i = 0; i < x.size(); i++) {
    x[i] /= sum;
  }
}

void IspImageClassification::outputParser(cvtdl_class_meta_t *cls_meta) {
  TensorInfo oinfo = getOutputTensorInfo(0);
  int8_t *ptr_int8 = static_cast<int8_t *>(oinfo.raw_pointer);
  uint8_t *ptr_uint8 = static_cast<uint8_t *>(oinfo.raw_pointer);
  float *ptr_float = static_cast<float *>(oinfo.raw_pointer);

#if ENABLE_TEAISP_PQ_DUMP
  if (dump_aipq) {
    dump_input_output("0-output.bin", ptr_float, 5 * sizeof(float));
    dump_aipq = false;
    printf("dump aipq done\n");
  }
#endif

  int channel_len = oinfo.shape.dim[1];

  int num_per_pixel = oinfo.tensor_size / oinfo.tensor_elem;
  float qscale = num_per_pixel == 1 ? oinfo.qscale : 1;

  std::vector<float> scores;
  // printf("output num:%d,datatype:%d\n",channel_len,oinfo.data_type);
  // int8 output tensor need to multiply qscale
  for (int i = 0; i < channel_len; i++) {
    if (num_per_pixel == 1) {
      scores.push_back(ptr_int8[i] * qscale);
    } else {
      scores.push_back(ptr_float[i]);
    }
    // printf("scales:%f\n", scores[i]);
  }
  softmax(scores);
  // for (int i = 0; i < channel_len; i++) {
  //   printf("scales:%f\n", scores[i]);
  // }

  std::vector<int> topKIndex = TopKIndex(scores, topK);

  // parse topK classes and scores
  for (uint32_t i = 0; i < topKIndex.size(); ++i) {
    cls_meta->cls[i] = topKIndex[i];
    cls_meta->score[i] = scores[topKIndex[i]];
  }
}
// namespace cvitdl
}  // namespace cvitdl
