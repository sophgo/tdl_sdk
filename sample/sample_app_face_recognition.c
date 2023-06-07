/*
input:
    fr_model_pat
    fd_model_path
    image_path
output:
    face feature vector
*/

#include "app/cviai_app.h"
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "evaluation/cviai_media.h"
#include "sample_comm.h"
#include "vi_vo_utils.h"

#include <cvi_sys.h>
#include <cvi_vb.h>
#include <cvi_vi.h>

#include <inttypes.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "stb_image.h"
#include "stb_image_write.h"

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("need fr_model_path, fd_model_path, image_path");
    return CVI_FAILURE;
  }
  int vpssgrp_width = 640;
  int vpssgrp_height = 640;
  CVI_S32 ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 1,
                                 vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 1);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  cviai_handle_t ai_handle = NULL;
  ret = CVI_AI_CreateHandle(&ai_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create ai handle failed with %#x!\n", ret);
    return ret;
  }
  cvai_face_t p_obj = {0};
  const char *fr_model_path =
      argv[1];  // /mnt/data/admin1_data/AI_CV/cv182x/ai_models/output/cv181x/cviface-v5-s.cvimodel
  const char *fd_model_path = argv
      [2];  // /mnt/data/admin1_data/AI_CV/cv182x/ai_models/output/cv181x/scrfd_432_768_1x.cvimodel
  const char *img_path = argv[3];  // /mnt/data/admin1_data/alios_test/a.jpg

  ret = CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_FACERECOGNITION, fr_model_path);
  ret = CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_SCRFDFACE, fd_model_path);

  VIDEO_FRAME_INFO_S bg;
  ret = CVI_AI_LoadBinImage(img_path, &bg, PIXEL_FORMAT_RGB_888_PLANAR);
  if (ret != CVI_SUCCESS) {
    printf("failed to open file\n");
    return ret;
  } else {
    printf("image read,width:%d\n", bg.stVFrame.u32Width);
  }

  // get bbox
  ret = CVI_AI_ScrFDFace(ai_handle, &bg, &p_obj);
  if (ret != CVI_SUCCESS) {
    printf("failed to run face detection\n");
    return ret;
  }
  // get face feature
  ret |= CVI_AI_FaceRecognition(ai_handle, &bg, &p_obj);
  if (ret != CVI_SUCCESS) {
    printf("failed to run face re\n");
    return ret;
  }
  //   printf("b\n");
  CVI_VPSS_ReleaseChnFrame(0, 0, &bg);  //(&bg);
  for (uint32_t i = 0; i < p_obj.size; i++) {
    printf("bbox=[%f,%f,%f,%f]\n", p_obj.info[i].bbox.x1, p_obj.info[i].bbox.y1,
           p_obj.info[i].bbox.x2, p_obj.info[i].bbox.y2);
    printf("feature=[");
    for (uint32_t j = 0; j < p_obj.info[i].feature.size; j++) {
      printf("%d ", p_obj.info[i].feature.ptr[j]);
    }
    printf("]");
    printf("\n");
  }
  CVI_AI_DestroyHandle(ai_handle);
  return true;
}