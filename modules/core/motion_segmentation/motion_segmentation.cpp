#include "motion_segmentation.hpp"
#include <string.h>
#include <time.h>
#include <cmath>
#include <iostream>
#include "core/core/cvtdl_errno.h"
#include "cvi_tdl_log.hpp"
using namespace cvitdl;

void dump_frame(VIDEO_FRAME_INFO_S *frame, std::string name) {
  static int image_count = 0;
  std::string base_path = "/mnt/data/nfsuser_wzl/preds/motion_segmentation_debug/" + name +
                          std::to_string(image_count) + ".bin";
  printf("image format is : %d\n", frame->stVFrame.enPixelFormat);
  if (frame->stVFrame.pu8VirAddr[0] == NULL) {
    size_t image_size =
        frame->stVFrame.u32Length[0] + frame->stVFrame.u32Length[1] + frame->stVFrame.u32Length[2];
    printf("image_size: %d\n", image_size);
    frame->stVFrame.pu8VirAddr[0] =
        (CVI_U8 *)CVI_SYS_MmapCache(frame->stVFrame.u64PhyAddr[0], image_size);
    frame->stVFrame.pu8VirAddr[1] = frame->stVFrame.pu8VirAddr[0] + frame->stVFrame.u32Length[0];
    frame->stVFrame.pu8VirAddr[2] = frame->stVFrame.pu8VirAddr[1] + frame->stVFrame.u32Length[1];
  }

  printf("===== %d %d %d\n", frame->stVFrame.u32Length[0], frame->stVFrame.u32Length[1],
         frame->stVFrame.u32Length[2]);
  FILE *fp = fopen(base_path.c_str(), "wb");
  for (int i = 0; i < 10; i++) {
    std::cout << "this is core clip:" << static_cast<int>(frame->stVFrame.pu8VirAddr[0][i]) << "**"
              << std::endl;
  }
  for (int i = 0; i < 3; i++) {
    uint8_t *paddr = (uint8_t *)frame->stVFrame.pu8VirAddr[i];
    fwrite(paddr, frame->stVFrame.u32Length[i], 1, fp);
  }
  fclose(fp);
  image_count++;
}

MotionSegmentation::MotionSegmentation() : Core(CVI_MEM_DEVICE) {}

MotionSegmentation::~MotionSegmentation() {}

int MotionSegmentation::setupInputPreprocess(std::vector<InputPreprecessSetup> *data) {
  if (data->size() != 2) {
    LOGE("Motion Segmentation has 2 input.\n");
    return CVI_TDL_ERR_INVALID_ARGS;
  }
  for (int i = 0; i < 2; i++) {
    (*data)[i].factor[0] = 0.017124753831663668;
    (*data)[i].factor[1] = 0.01750700280112045;
    (*data)[i].factor[2] = 0.017429193899782137;
    (*data)[i].mean[0] = 2.1179039301310043;
    (*data)[i].mean[1] = 2.035714285714286;
    (*data)[i].mean[2] = 1.8044444444444445;
    (*data)[i].use_quantize_scale = true;
    // (*data)[i].factor[0] = 1;
    // (*data)[i].factor[1] = 1;
    // (*data)[i].factor[2] = 1;
    // (*data)[i].mean[0] = 0;
    // (*data)[i].mean[1] = 0;
    // (*data)[i].mean[2] = 0;
    // (*data)[i].use_quantize_scale = false;
    (*data)[i].use_crop = false;
    (*data)[i].keep_aspect_ratio = false;  // do not keep aspect ratio,resize directly
  }
  return CVI_TDL_SUCCESS;
}

// int MotionSegmentation::vpssPreprocess(VIDEO_FRAME_INFO_S* srcFrame, VIDEO_FRAME_INFO_S*
// dstFrame,
//                               VPSSConfig& vpss_config) {
//   auto& vpssChnAttr = vpss_config.chn_attr;
//   auto& factor = vpssChnAttr.stNormalize.factor;
//   auto& mean = vpssChnAttr.stNormalize.mean;
//   vpss_config.chn_coeff = VPSS_SCALE_COEF_NEAREST;
//   dump_frame(srcFrame,"srcframe");
//   std::cout<<"vpss_config.chn_coeff:"<<vpss_config.chn_coeff<<std::endl;
//   // int ret = mp_vpss_inst->sendCropChnFrame(srcFrame, &vpss_config.crop_attr,
//   &vpss_config.chn_attr,
//   //                                        &vpss_config.chn_coeff, 1);
//   int ret = mp_vpss_inst->sendFrame(srcFrame, &vpssChnAttr, &vpss_config.chn_coeff, 1);
//   if (ret != CVI_SUCCESS) {
//     LOGE("vpssPreprocess Send frame failed: %s!\n");
//     return CVI_TDL_ERR_VPSS_GET_FRAME;
//   }

//   ret = mp_vpss_inst->getFrame(dstFrame, 0, m_vpss_timeout);
//   if (ret != CVI_SUCCESS) {
//     LOGE("get frame failed: %s!\n");
//     return CVI_TDL_ERR_VPSS_GET_FRAME;
//   }
//   std::cout<<"save frame:"<<std::endl;
//   dump_frame(dstFrame, "dstFrame");
//   return CVI_TDL_SUCCESS;
// }

int MotionSegmentation::inference(VIDEO_FRAME_INFO_S *input0, VIDEO_FRAME_INFO_S *input1,
                                  cvtdl_seg_logits_t *seg_logits) {
  std::vector<VIDEO_FRAME_INFO_S *> inputs = {input0, input1};

  int ret = run(inputs);

  if (ret != CVI_TDL_SUCCESS) {
    LOGW("inference failed\n");
    return ret;
  }

  const TensorInfo &oinfo = getOutputTensorInfo(0);
  int byte_per_pixel = oinfo.tensor_size / oinfo.tensor_elem;
  float qscale_output = byte_per_pixel == 1 ? oinfo.qscale : 1;
  std::cout << "byte_per_pixel: " << byte_per_pixel << std::endl;
  seg_logits->w = oinfo.shape.dim[3];
  seg_logits->h = oinfo.shape.dim[2];
  seg_logits->c = oinfo.shape.dim[1];
  seg_logits->b = oinfo.shape.dim[0];

  seg_logits->is_int = (byte_per_pixel == 1);
  seg_logits->qscale = qscale_output;
  int8_t *int8_out_data = getOutputRawPtr<int8_t>(oinfo.tensor_name);
  float *float_out_data = getOutputRawPtr<float>(oinfo.tensor_name);
  if (byte_per_pixel == 1) {
    if (seg_logits->int_logits != NULL) {
      delete[] seg_logits->int_logits;
      seg_logits->int_logits = NULL;
    }
    seg_logits->int_logits =
        new int8_t[seg_logits->w * seg_logits->h * seg_logits->c * seg_logits->b];
    memcpy(seg_logits->int_logits, int8_out_data,
           seg_logits->w * seg_logits->h * seg_logits->c * seg_logits->b);
  } else {
    if (seg_logits->float_logits != NULL) {
      delete[] seg_logits->float_logits;
      seg_logits->float_logits = NULL;
    }
    seg_logits->float_logits =
        new float[seg_logits->w * seg_logits->h * seg_logits->c * seg_logits->b];
    memcpy(seg_logits->float_logits, float_out_data,
           seg_logits->w * seg_logits->h * seg_logits->c * seg_logits->b * sizeof(float));
  }

  return CVI_TDL_SUCCESS;
}