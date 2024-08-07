#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "core/cvi_tdl_types_mem_internal.h"
#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_media.h"
#include "mapi.hpp"
#include "sys_utils.hpp"
#include "utils/clip_postprocess.hpp"
#include "utils/token.hpp"

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }
cvitdl_handle_t tdl_handle = NULL;
static CVI_S32 vpssgrp_width = 1920;
static CVI_S32 vpssgrp_height = 1080;

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf(
        "Usage: %s <clip model path> <input image directory list.txt> <output result "
        "directory/>.\n",
        argv[0]);
    printf("clip image model path: Path to clip image bmodel.\n");
    printf("Input image directory: Directory containing input images for clip.\n");
    printf("clip text model path: Path to clip text bmodel.\n");
    printf("Input text directory: Directory containing input text for clip.\n");
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

  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_CLIP_IMAGE, argv[1]);
  if (ret != CVI_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }

  std::string image_list(argv[2]);

  std::cout << "to read file_list:" << image_list << std::endl;
  std::vector<std::string> image_file_list = read_file_lines(image_list);
  if (image_file_list.size() == 0) {
    std::cout << ", file_list empty\n";
    return -1;
  }

  std::string input_image_path;
  cvtdl_clip_feature clip_feature_image;

  std::cout << image_file_list.size() << std::endl;

  int rows_image = image_file_list.size();
  int cols_image = 512;

  float** image_features = new float*[image_file_list.size()];

  for (size_t i = 0; i < image_file_list.size(); i++) {
    input_image_path = image_file_list[i];
    VIDEO_FRAME_INFO_S rgb_frame;

    size_t line_position = input_image_path.find_last_of('/');
    size_t dot_position = input_image_path.find_last_of('.');
    string pic_name =
        input_image_path.substr(line_position + 1, dot_position - line_position - 1).c_str();
    std::cout << "number of img:" << i << ";last of imgname:" << pic_name << std::endl;
    imgprocess_t img_handle;
    CVI_TDL_Create_ImageProcessor(&img_handle);
    ret = CVI_TDL_ReadImage_CenrerCrop_Resize(img_handle, input_image_path.c_str(), &rgb_frame,
                                              PIXEL_FORMAT_RGB_888_PLANAR, 224, 224);
    if (ret != CVI_SUCCESS) {
      printf("open img failed with %#x!\n", ret);
      return ret;
    }
    ret = CVI_TDL_Clip_Image_Feature(tdl_handle, &rgb_frame, &clip_feature_image);
    if (ret != CVI_SUCCESS) {
      printf("Failed to CVI_TDL_Clip_Feature\n");
      return 0;
    }

    image_features[i] = new float[clip_feature_image.feature_dim];
    for (int y = 0; y < clip_feature_image.feature_dim; ++y) {
      image_features[i][y] = clip_feature_image.out_feature[y];
    }

    CVI_TDL_Free(&clip_feature_image);
    std::cout << "after free:" << std::endl;

    CVI_TDL_ReleaseImage(img_handle, &rgb_frame);
  }

  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_CLIP_TEXT, argv[3]);
  if (ret != CVI_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }

  std::string text_list(argv[4]);

  std::cout << "to read file_list:" << text_list << std::endl;
  std::vector<std::string> text_file_list = read_file_lines(text_list);
  if (text_file_list.size() == 0) {
    std::cout << ", file_list empty\n";
    return -1;
  }
  cvtdl_clip_feature clip_feature_text;

  std::cout << text_file_list.size() << std::endl;

  std::string encoderFile = "./encoder.txt";
  std::string bpeFile = "./bpe_simple_vocab_16e6.txt";

  int32_t** tokens = (int32_t**)malloc(text_file_list.size() * sizeof(int32_t*));
  ret = CVI_TDL_Set_TextPreprocess(encoderFile.c_str(), bpeFile.c_str(), text_list.c_str(), tokens,
                                   text_file_list.size());
  if (ret != CVI_SUCCESS) {
    printf("CVI_TDL_Set_TextPreprocess\n");
    return 0;
  }

  float** text_features = new float*[text_file_list.size()];
  for (int i = 0; i < text_file_list.size(); i++) {
    CVI_U8 buffer[77 * sizeof(int32_t)];
    memcpy(buffer, tokens[i], sizeof(int32_t) * 77);
    VIDEO_FRAME_INFO_S Frame;
    Frame.stVFrame.pu8VirAddr[0] = buffer;
    Frame.stVFrame.u32Height = 1;
    Frame.stVFrame.u32Width = 77;

    ret = CVI_TDL_Clip_Text_Feature(tdl_handle, &Frame, &clip_feature_text);
    if (ret != CVI_SUCCESS) {
      printf("CVI_TDL_OpenClip_Text_Feature\n");
      return 0;
    }

    text_features[i] = new float[clip_feature_text.feature_dim];
    for (int y = 0; y < clip_feature_text.feature_dim; ++y) {
      text_features[i][y] = clip_feature_text.out_feature[y];
    }
    CVI_TDL_Free(&clip_feature_text);
  }

  for (int i = 0; i < text_file_list.size(); i++) {
    free(tokens[i]);
  }
  free(tokens);
  int* prods_index = (int*)malloc(image_file_list.size() * sizeof(int));
  float thres = 0.5;
  int function_id = 1;
  ret = CVI_TDL_Set_ClipPostprocess(text_features, text_file_list.size(), image_features,
                                    image_file_list.size(), prods_index, function_id, thres);

  delete[] text_features;
  delete[] image_features;

  for (int i = 0; i < image_file_list.size(); i++) {
    std::cout << prods_index[i] << " ";
  }
  std::cout << std::endl;
  free(prods_index);
  CVI_TDL_DestroyHandle(tdl_handle);
  return CVI_SUCCESS;
}
