#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "core/cvi_tdl_types_mem_internal.h"
#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_media.h"
#include "sys_utils.hpp"

int main(int argc, char *argv[]) {
  if (argc != 10) {
    printf(
        "Usage: %s <BLIP_VQA_VENC model path> <BLIP_VQA_TENC model path> <BLIP_VQA_TDEC model "
        "path>\n"
        "<vocab file path>  <input image dir.txt>  <input image list.txt> <input txt dir> <input "
        "txt list.txt> <result dir>\n",
        argv[0]);
    return CVI_FAILURE;
  }

  int vpssgrp_width = 1920;
  int vpssgrp_height = 1080;
  CVI_S32 ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 2,
                                 vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 2);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  cvitdl_handle_t tdl_handle = NULL;
  ret = CVI_TDL_CreateHandle(&tdl_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create tdl handle failed with %#x!\n", ret);
    return ret;
  }

  printf("It will take several minutes to load the blip vqa models, please wait ...\n");

  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_BLIP_VQA_VENC, argv[1]);
  if (ret != CVI_SUCCESS) {
    printf("Open model failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_BLIP_VQA_TENC, argv[2]);
  if (ret != CVI_SUCCESS) {
    printf("Open model failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_BLIP_VQA_TDEC, argv[3]);
  if (ret != CVI_SUCCESS) {
    printf("Open model failed with %#x!\n", ret);
    return ret;
  }

  std::string image_dir(argv[5]);
  std::string image_list(argv[6]);
  std::string txt_dir(argv[7]);
  std::string txt_list(argv[8]);
  std::string save_dir(argv[9]);

  if (image_dir.at(image_dir.size() - 1) != '/') {
    image_dir = image_dir + std::string("/");
  }
  if (txt_dir.at(txt_dir.size() - 1) != '/') {
    txt_dir = txt_dir + std::string("/");
  }
  if (save_dir.at(save_dir.size() - 1) != '/') {
    save_dir = save_dir + std::string("/");
  }

  std::cout << "to read image list:" << image_list << std::endl;
  std::vector<std::string> image_sub_paths = read_file_lines(image_list);

  std::cout << "to read txt list:" << txt_list << std::endl;
  std::vector<std::string> txt_paths = read_file_lines(txt_list);

  printf("image_sub_paths size:%d txt paths size:%d\\n", image_sub_paths.size(), txt_paths.size());

  if (image_sub_paths.size() == 0 || txt_paths.size() == 0) {
    std::cout << "image_sub_paths or txt_paths is empty!\n";
    return -1;
  }
  if (image_sub_paths.size() != txt_paths.size() && txt_paths.size() > 1) {
    printf("The size of image_sub_paths and txt_paths should be same!\n");
    return -1;
  }

  imgprocess_t img_handle;
  CVI_TDL_Create_ImageProcessor(&img_handle);

  cvtdl_tokens tokens_in = {0};
  cvtdl_image_embeds embeds_meta = {0};
  cvtdl_tokens tokens_out = {0};

  std::string question_path;

  ret = CVI_TDL_WordPieceInit(tdl_handle, argv[4]);
  if (ret != CVI_SUCCESS) {
    printf("CVI_TDL_WordPieceInit failed  with %#x!\n", ret);
    return 0;
  }

  if (txt_paths.size() == 1) {  // all the images use same one question
    question_path = txt_dir + txt_paths[0];
    ret = CVI_TDL_WordPieceToken(tdl_handle, question_path.c_str(), &tokens_in);
    if (ret != CVI_SUCCESS) {
      printf("CVI_TDL_WordPieceToken failed  with %#x!\n", ret);
      return 0;
    }
  }

  for (size_t i = 0; i < image_sub_paths.size(); i++) {
    VIDEO_FRAME_INFO_S bg;

    std::cout << "processing :" << i + 1 << "/" << image_sub_paths.size() << "\t"
              << image_sub_paths[i] << std::endl;

    std::string image_path = image_dir + image_sub_paths[i];

    ret = CVI_TDL_ReadImage(img_handle, image_path.c_str(), &bg, PIXEL_FORMAT_RGB_888_PLANAR);
    if (ret != CVI_SUCCESS) {
      printf("open img failed with %#x!\n", ret);
      return ret;
    }

    if (txt_paths.size() > 1) {  // one picture for one question

      question_path = txt_dir + txt_paths[i];
      ret = CVI_TDL_WordPieceToken(tdl_handle, question_path.c_str(), &tokens_in);
    }

    if (ret != CVI_SUCCESS) {
      printf("CVI_TDL_WordPieceToken failed  with %#x!\n", ret);
      return 0;
    }

    CVI_TDL_Blip_Vqa_Venc(tdl_handle, &bg, &embeds_meta);

    CVI_TDL_Blip_Vqa_Tenc(tdl_handle, &embeds_meta, &tokens_in);

    CVI_TDL_Blip_Vqa_Tdec(tdl_handle, &embeds_meta, &tokens_out);

    ret = CVI_TDL_WordPieceDecode(tdl_handle, &tokens_out);
    if (ret != CVI_SUCCESS) {
      printf("CVI_TDL_WordPieceDecode failed with %#x!\n", ret);
      return 0;
    }

    std::string file_name = get_file_name_without_extension(image_sub_paths[i]);
    std::string save_path = save_dir + file_name + ".txt";

    std::ofstream outFile(save_path);
    if (!outFile) {
      std::cout << "Error: Could not open file " << save_path << " for writing." << std::endl;
      return -1;
    }

    outFile << tokens_out.text[0] << std::endl;
    outFile.close();

    if (txt_paths.size() > 1) {
      CVI_TDL_Free(&tokens_in);
    }
    CVI_TDL_Free(&tokens_out);
    CVI_TDL_Free(&embeds_meta);
    CVI_TDL_ReleaseImage(img_handle, &bg);
  }

  if (txt_paths.size() == 1) {
    CVI_TDL_Free(&tokens_in);  // just free once
  }

  CVI_TDL_DestroyHandle(tdl_handle);
  CVI_TDL_Destroy_ImageProcessor(img_handle);
  return ret;
}
