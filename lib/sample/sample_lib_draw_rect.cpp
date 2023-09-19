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
#include "cvi_draw_rect.h"
#include "evaluation/cviai_media.h"

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
  for (int c = 0; c < 3; c++) {
    uint8_t *paddr = (uint8_t *)frame->stVFrame.pu8VirAddr[c];
    std::cout << "towrite channel:" << c << ",towritelen:" << frame->stVFrame.u32Length[c]
              << ",addr:" << (void *)paddr << std::endl;
    fwrite(paddr, frame->stVFrame.u32Length[c], 1, fp);
  }
  fclose(fp);
  return CVI_SUCCESS;
}

int main(int argc, char *argv[]) {
  int vpssgrp_width = 1920;
  int vpssgrp_height = 1080;
  CVI_S32 ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 1,
                                 vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 1);
  if (ret != CVIAI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  VIDEO_FRAME_INFO_S bg_view;
  std::string strf1(argv[1]);
  ret |= CVI_AI_ReadImage(strf1.c_str(), &bg_view, PIXEL_FORMAT_YUV_PLANAR_420);
  if (ret != CVI_SUCCESS) {
    printf("open img failed with %#x!\n", ret);
    return ret;
  } else {
    printf("image read,width:%d\n", bg_view.stVFrame.u32Width);
  }

  cvai_object_t obj_meta = {0};
  cvai_service_brush_t brushi;
  brushi.color.r = 255;
  brushi.color.g = 255;
  brushi.color.b = 255;
  brushi.size = 4;
  dump_frame_result("./test1.yuv", &bg_view);
  CVI_AI_MemAlloc(1, &obj_meta);
  obj_meta.size = 1;
  obj_meta.rescale_type = meta_rescale_type_e::RESCALE_CENTER;
  obj_meta.info[0].bbox.x1 = 100;
  obj_meta.info[0].bbox.y1 = 100;
  obj_meta.info[0].bbox.x2 = 150;
  obj_meta.info[0].bbox.y2 = 150;

  CVI_AI_ObjectDrawRect(&obj_meta, &bg_view, false, brushi);
  dump_frame_result("./test2.yuv", &bg_view);
  return ret;
}
