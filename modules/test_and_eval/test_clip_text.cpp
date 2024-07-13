#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <algorithm>
#include <fstream>
#include <iomanip>
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
#include "utils/token.hpp"

cvitdl_handle_t tdl_handle = NULL;
static CVI_S32 vpssgrp_width = 1920;
static CVI_S32 vpssgrp_height = 1080;

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf(
        "Usage: %s <clip model path> <input image directory list.txt> <output result "
        "directory/>.\n",
        argv[0]);
    printf("clip model path: Path to clip bmodel.\n");
    printf("Input image directory: Directory containing input images for clip.\n");
    return CVI_FAILURE;
  }

  CVI_S32 ret = CVI_TDL_CreateHandle(&tdl_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create tdl handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_CLIP_TEXT, argv[1]);
  if (ret != CVI_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }

  std::string token_path(argv[2]);
  cvtdl_clip_feature clip_feature;

  std::string encoderFile = "./encoder.txt";
  std::string bpeFile = "./bpe_simple_vocab_16e6.txt";
  std::vector<std::vector<int32_t>> tokens;
  int result = cvitdl::token_bpe(encoderFile, bpeFile, token_path, tokens);
  std::ofstream outfile("a2_text_output.txt");

  for (int i = 0; i < tokens.size(); i++) {
    CVI_U8 buffer[tokens[0].size() * sizeof(int32_t)];
    memcpy(buffer, &tokens[i][0], sizeof(int32_t) * tokens[0].size());
    VIDEO_FRAME_INFO_S Frame;
    Frame.stVFrame.pu8VirAddr[0] = buffer;
    Frame.stVFrame.u32Height = 1;
    Frame.stVFrame.u32Width = tokens[0].size();

    ret = CVI_TDL_Clip_Text_Feature(tdl_handle, &Frame, &clip_feature);
    if (ret != CVI_SUCCESS) {
      printf("CVI_TDL_OpenClip_Text_Feature\n");
      return 0;
    }

    for (int y = 0; y < clip_feature.feature_dim; ++y) {
      outfile << clip_feature.out_feature[y];
      if (y < clip_feature.feature_dim - 1) {
        outfile << " ";
      }
    }
    outfile << "\n";
    free(clip_feature.out_feature);
  }
  outfile.close();

  std::cout << "after free:" << std::endl;

  CVI_TDL_DestroyHandle(tdl_handle);

  return CVI_SUCCESS;
}
