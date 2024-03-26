#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "core/cvi_tdl_types_mem_internal.h"
#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_media.h"
#include "mapi.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "sys_utils.hpp"

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }
cvitdl_handle_t tdl_handle = NULL;
static CVI_S32 vpssgrp_width = 1920;
static CVI_S32 vpssgrp_height = 1080;

bool CompareFileNames(std::string a, std::string b) { return a < b; }

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf(
        "Usage: %s <clip model path> <input image directory list.txt> <output result "
        "directory/>.\n",
        argv[0]);
    printf("clip model path: Path to clip bmodel.\n");
    printf("Input image directory: Directory containing input images for clip.\n");
    printf("Output result directory: Directory to save clip feature.bin\n");
    return CVI_FAILURE;
  }
  CVI_S32 ret = CVI_SUCCESS;
  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 3, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888, 3);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_TDL_CreateHandle(&tdl_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create tdl handle failed with %#x!\n", ret);
    return ret;
  }
  // MODEL_DIR
  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_CLIP, argv[1]);
  if (ret != CVI_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }
  //  IMAGE_LIST
  std::string image_list(argv[2]);

  //  dir_name
  std::string dir_name(argv[3]);

  std::cout << "to read file_list:" << image_list << std::endl;
  std::vector<std::string> file_list = read_file_lines(image_list);
  if (file_list.size() == 0) {
    std::cout << ", file_list empty\n";
    return -1;
  }

  std::sort(file_list.begin(), file_list.end(), CompareFileNames);

  std::string input_image_path;
  cvtdl_clip_feature clip_feature;

  for (size_t i = 0; i < file_list.size(); i++) {
    input_image_path = file_list[i];
    VIDEO_FRAME_INFO_S rgb_frame;

    size_t line_position = input_image_path.find_last_of('/');
    size_t dot_position = input_image_path.find_last_of('.');
    string pic_name =
        input_image_path.substr(line_position + 1, dot_position - line_position - 1).c_str();
    std::cout << "number of img:" << i << ";last of imgname:" << pic_name << std::endl;
    imgprocess_t img_handle;
    CVI_TDL_Create_ImageProcessor(&img_handle);
    ret = CVI_TDL_ReadImage_CenrerCrop_Resize(img_handle, input_image_path.c_str(), &rgb_frame,
                                              PIXEL_FORMAT_RGB_888_PLANAR, 288, 288);
    if (ret != CVI_SUCCESS) {
      printf("open img failed with %#x!\n", ret);
      return ret;
    }
    std::ofstream outfile(dir_name + pic_name + ".bin", std::ios::binary);
    if (!outfile) {
      std::cerr << "无法打开文件" << std::endl;
      return -1;
    }
    printf("SUCCESS to read image: %s\n", input_image_path.c_str());
    ret = CVI_TDL_Clip_Feature(tdl_handle, &rgb_frame, &clip_feature);
    if (ret != CVI_SUCCESS) {
      printf("Failed to CVI_TDL_Clip_Feature\n");
      return 0;
    }
    for (int y = 0; y < clip_feature.feature_dim; ++y) {
      outfile.write(reinterpret_cast<const char *>(&clip_feature.out_feature[y]), sizeof(float));
    }
    // 需要释放结构体out_feature指针
    free(clip_feature.out_feature);
    outfile.close();
    std::cout << "after free:" << std::endl;

    CVI_TDL_ReleaseImage(img_handle, &rgb_frame);
  }

  CVI_TDL_DestroyHandle(tdl_handle);

  return CVI_SUCCESS;
}
