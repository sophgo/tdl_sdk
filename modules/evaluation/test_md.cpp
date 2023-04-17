#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "core/cviai_types_mem_internal.h"
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "evaluation/cviai_media.h"

#include <iostream>
#include <sstream>
int main(int argc, char *argv[]) {
  int vpssgrp_width = 1920;
  int vpssgrp_height = 1080;
  CVI_S32 ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 1,
                                 vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 1);
  if (ret != CVIAI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }
  cviai_handle_t ai_handle = NULL;
  ret = CVI_AI_CreateHandle(&ai_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create ai handle failed with %#x!\n", ret);
    return ret;
  }
  std::string strf1(argv[1]);
  std::string strf2(argv[2]);

  VIDEO_FRAME_INFO_S bg;
  CVI_AI_ReadImage(strf1.c_str(), &bg, PIXEL_FORMAT_YUV_400);
  std::cout << "read image1 done\n";
  VIDEO_FRAME_INFO_S frame;
  cvai_object_t obj_meta;
  memset(&obj_meta, 0, sizeof(cvai_object_t));
  CVI_AI_ReadImage(strf2.c_str(), &frame, PIXEL_FORMAT_YUV_400);
  std::cout << "read image2 done\n";
  CVI_AI_Set_MotionDetection_Background(ai_handle, &bg);
  std::cout << "set image bg done\n";
  CVI_AI_MotionDetection(ai_handle, &frame, &obj_meta, 30, 100);
  std::stringstream ss;
  ss << "boxes=[";
  for (size_t i = 0; i < obj_meta.size; i++) {
    ss << "[" << obj_meta.info[i].bbox.x1 << "," << obj_meta.info[i].bbox.y1 << ","
       << obj_meta.info[i].bbox.x2 << "," << obj_meta.info[i].bbox.y2 << "],";
  }
  std::cout << ss.str() << "]\n";
  CVI_AI_Free(&obj_meta);
  CVI_AI_ReleaseImage(&bg);
  CVI_AI_ReleaseImage(&frame);

  CVI_AI_DestroyHandle(ai_handle);

  return ret;
}
