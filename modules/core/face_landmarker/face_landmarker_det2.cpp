#include "face_landmarker_det2.hpp"
#include <core/core/cvai_errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <error_msg.hpp>
#include <iostream>
#include <iterator>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include "coco_utils.hpp"
#include "core/core/cvai_errno.h"
#include "core/cviai_types_mem.h"
#include "core/cviai_types_mem_internal.h"
#include "core/utils/vpss_helper.h"
#include "core_utils.hpp"
#include "cvi_sys.h"

namespace cviai {

FaceLandmarkerDet2::FaceLandmarkerDet2() : Core(CVI_MEM_DEVICE) {}

int FaceLandmarkerDet2::onModelOpened() {
  for (size_t j = 0; j < getNumOutputTensor(); j++) {
    TensorInfo oinfo = getOutputTensorInfo(j);
    // printf("output:%s,dim:%d,%d,%d,%d\n",oinfo.tensor_name.c_str(),oj.dim[0],oj.dim[1],oj.dim[2],oj.dim[3]);
    std::string channel_name = oinfo.tensor_name.c_str();
    if (channel_name.compare("score_Gemm_f32") == 0) {
      out_names_["score"] = oinfo.tensor_name;
      printf("add to out_names: %s\n", oinfo.tensor_name.c_str());
      // printf("parse score branch output:%s\n", oinfo.tensor_name.c_str());
    } else if (channel_name.compare("x_pred_Add_f32") == 0) {
      out_names_["point_x"] = oinfo.tensor_name;
      printf("add to out_names: %s\n", oinfo.tensor_name.c_str());
      // printf("parse point output:%s\n", oinfo.tensor_name.c_str());
    } else {
      out_names_["point_y"] = oinfo.tensor_name;
      printf("add to out_names: %s\n", oinfo.tensor_name.c_str());
      // printf("parse point output:%s\n", oinfo.tensor_name.c_str());
    }
  }
  if (out_names_.count("score") == 0 || out_names_.count("point_x") == 0 ||
      out_names_.count("point_y") == 0) {
    return CVIAI_FAILURE;
  }
  return CVIAI_SUCCESS;
}

FaceLandmarkerDet2::~FaceLandmarkerDet2() {}

int FaceLandmarkerDet2::setupInputPreprocess(std::vector<InputPreprecessSetup> *data) {
  if (data->size() != 1) {
    LOGE("FaceLandmarkerDet2 only has 1 input.\n");
    return CVIAI_ERR_INVALID_ARGS;
  }

  (*data)[0].factor[0] = 1 / 127.5;
  (*data)[0].factor[1] = 1 / 127.5;
  (*data)[0].factor[2] = 1 / 127.5;
  (*data)[0].mean[0] = 1.0;
  (*data)[0].mean[1] = 1.0;
  (*data)[0].mean[2] = 1.0;

  // (*data)[0].factor[0] = 1;
  // (*data)[0].factor[1] = 1;
  // (*data)[0].factor[2] = 1;
  // (*data)[0].mean[0] = 0;
  // (*data)[0].mean[1] = 0;
  // (*data)[0].mean[2] = 0;
  (*data)[0].format = PIXEL_FORMAT_RGB_888_PLANAR;
  (*data)[0].use_quantize_scale = true;
  (*data)[0].rescale_type = RESCALE_NOASPECT;
  (*data)[0].keep_aspect_ratio = false;
  // printf("setup input preprocess finished! \n");
  return CVIAI_SUCCESS;
}

int dump_frame_result(const std::string &filepath, VIDEO_FRAME_INFO_S *frame) {
  FILE *fp = fopen(filepath.c_str(), "wb");
  if (fp == nullptr) {
    LOGE("failed to open: %s.\n", filepath.c_str());
    return CVI_FAILURE;
  }

  if (frame->stVFrame.pu8VirAddr[0] == NULL) {
    size_t image_size =
        frame->stVFrame.u32Length[0] + frame->stVFrame.u32Length[1] + frame->stVFrame.u32Length[2];
    frame->stVFrame.pu8VirAddr[0] =
        (CVI_U8 *)CVI_SYS_MmapCache(frame->stVFrame.u64PhyAddr[0], image_size);
    frame->stVFrame.pu8VirAddr[1] = frame->stVFrame.pu8VirAddr[0] + frame->stVFrame.u32Length[0];
    frame->stVFrame.pu8VirAddr[2] = frame->stVFrame.pu8VirAddr[1] + frame->stVFrame.u32Length[1];
  }
  // std::cout << "u32Width: " << frame->stVFrame.u32Width << " u32 Height: " <<
  // frame->stVFrame.u32Height << std::endl;
  for (int c = 0; c < 3; c++) {
    uint8_t *paddr = (uint8_t *)frame->stVFrame.pu8VirAddr[c];
    std::cout << "towrite channel:" << c << ",towritelen:" << frame->stVFrame.u32Length[c]
              << ",addr:" << (void *)paddr << std::endl;
    fwrite(paddr, frame->stVFrame.u32Length[c], 1, fp);
  }
  fclose(fp);
  return CVI_SUCCESS;
}

int FaceLandmarkerDet2::vpssPreprocess(VIDEO_FRAME_INFO_S *srcFrame, VIDEO_FRAME_INFO_S *dstFrame,
                                       VPSSConfig &vpss_config) {
  auto &vpssChnAttr = vpss_config.chn_attr;
  auto &factor = vpssChnAttr.stNormalize.factor;
  auto &mean = vpssChnAttr.stNormalize.mean;
  // set dump config
  // dump_frame_result("origin", srcFrame);
  vpssChnAttr.stNormalize.bEnable = false;

  VPSS_CHN_SQ_RB_HELPER(&vpssChnAttr, srcFrame->stVFrame.u32Width, srcFrame->stVFrame.u32Height,
                        vpssChnAttr.u32Width, vpssChnAttr.u32Height, PIXEL_FORMAT_RGB_888_PLANAR,
                        factor, mean, false);
  int ret = mp_vpss_inst->sendFrame(srcFrame, &vpssChnAttr, &vpss_config.chn_coeff, 1);
  if (ret != CVI_SUCCESS) {
    LOGE("vpssPreprocess Send frame failed: %s!\n", get_vpss_error_msg(ret));
    return CVIAI_ERR_VPSS_SEND_FRAME;
  }

  ret = mp_vpss_inst->getFrame(dstFrame, 0, m_vpss_timeout);
  if (ret != CVI_SUCCESS) {
    LOGE("get frame failed: %s!\n", get_vpss_error_msg(ret));
    return CVIAI_ERR_VPSS_GET_FRAME;
  }
  // std::cout << ""
  // std::cout << "vpass preprocessed u32Width: " << dstFrame->stVFrame.u32Width << " u32 Height: "
  // << dstFrame->stVFrame.u32Height << std::endl; dump_frame_result("vpss_processed", dstFrame);
  return CVIAI_SUCCESS;
}

int FaceLandmarkerDet2::inference(VIDEO_FRAME_INFO_S *srcFrame, cvai_face_t *facemeta) {
  // printf("processing inference..\n");
  std::vector<VIDEO_FRAME_INFO_S *> frames = {srcFrame};
  int ret = run(frames);
  // printf("run result %d \n", ret);
  if (ret != CVIAI_SUCCESS) {
    printf("FaceLandmarkerDet2 run inference failed\n");
    // printf("inference: frame_width %d \t frame_height %d \n", frames[0]->stVFrame.u32Width,
    // frames[0]->stVFrame.u32Height);
    return ret;
  }

  // printf("inference: frame_width %d \t frame_height %d \n", srcFrame->stVFrame.u32Width,
  // srcFrame->stVFrame.u32Height);

  CVI_SHAPE shape = getInputShape(0);

  outputParser(shape.dim[3], shape.dim[2], srcFrame->stVFrame.u32Width,
               srcFrame->stVFrame.u32Height, facemeta);
  model_timer_.TicToc("post");
  return CVIAI_SUCCESS;
}

void FaceLandmarkerDet2::outputParser(const int image_width, const int image_height,
                                      const int frame_width, const int frame_height,
                                      cvai_face_t *facemeta) {
  TensorInfo oinfo_x = getOutputTensorInfo(out_names_["point_x"]);
  float *output_point_x = getOutputRawPtr<float>(oinfo_x.tensor_name);

  TensorInfo oinfo_y = getOutputTensorInfo(out_names_["point_y"]);
  float *output_point_y = getOutputRawPtr<float>(oinfo_y.tensor_name);

  // TensorInfo oinfo_cls = getOutputTensorInfo(out_names_["score"]);
  float *output_score = getOutputRawPtr<float>("score_Gemm_f32");

  float score = 1.0 / (1.0 + exp(-output_score[0]));

  CVI_AI_MemAllocInit(1, 5, facemeta);
  facemeta->width = frame_width;
  facemeta->height = frame_height;
  facemeta->info[0].pts.score = score;

  for (int i = 0; i < 5; i++) {
    float x = output_point_x[i] * frame_width;
    float y = output_point_y[i] * frame_height;
    facemeta->info[0].pts.x[i] = x;
    facemeta->info[0].pts.y[i] = y;
  }
}
// namespace cviai
}  // namespace cviai
