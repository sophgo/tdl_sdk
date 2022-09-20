#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "core/cviai_types_mem_internal.h"
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "evaluation/cviai_media.h"
// #include "ive/ive.h"
// #include "sys_utils.hpp"
#include <iostream>
#include <sstream>
int main(int argc, char *argv[]) {
  cviai_handle_t ai_handle = NULL;
  CVI_S32 ret = CVI_AI_CreateHandle(&ai_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create ai handle failed with %#x!\n", ret);
    return ret;
  }
  std::string strf1("/mnt/data/admin1_data/alios_test/set/a.jpg");
  std::string strf2("/mnt/data/admin1_data/alios_test/set/b.jpg");

  VIDEO_FRAME_INFO_S bg;
  // printf("toread image:%s\n",argv[1]);
  CVI_AI_ReadImage(strf1.c_str(), &bg, PIXEL_FORMAT_YUV_400);

  VIDEO_FRAME_INFO_S frame;
  cvai_object_t obj_meta;
  CVI_AI_ReadImage(strf2.c_str(), &frame, PIXEL_FORMAT_YUV_400);

  CVI_AI_Set_MotionDetection_Background(ai_handle, &bg);
  CVI_AI_MotionDetection(ai_handle, &frame, &obj_meta, 30, 100);
  std::stringstream ss;
  ss << "boxes=[";
  for (int i = 0; i < obj_meta.size; i++) {
    ss << "[" << obj_meta.info[i].bbox.x1 << "," << obj_meta.info[i].bbox.y1 << ","
       << obj_meta.info[i].bbox.x2 << "," << obj_meta.info[i].bbox.y2 << "],";
  }
  std::cout << ss.str() << "]\n";
  CVI_AI_FreeCpp(&obj_meta);
  CVI_AI_ReleaseImage(&bg);
  CVI_AI_ReleaseImage(&frame);

  CVI_AI_DestroyHandle(ai_handle);
  return ret;
}
