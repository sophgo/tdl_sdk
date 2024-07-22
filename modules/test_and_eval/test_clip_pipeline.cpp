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
#include "Eigen/Core"
#include "core/cvi_tdl_types_mem_internal.h"
#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_media.h"
#include "mapi.hpp"
#include "sys_utils.hpp"
#include "utils/token.hpp"

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }
cvitdl_handle_t tdl_handle = NULL;
static CVI_S32 vpssgrp_width = 1920;
static CVI_S32 vpssgrp_height = 1080;

void normalize_matrix(Eigen::MatrixXf& matrix) {
  for (int i = 0; i < matrix.rows(); ++i) {
    float norm = matrix.row(i).norm();
    if (norm != 0.0f) {
      matrix.row(i) /= norm;
    }
  }
}

Eigen::MatrixXf softmax(const Eigen::MatrixXf& input) {
  Eigen::MatrixXf result(input.rows(), input.cols());
  for (int i = 0; i < input.rows(); ++i) {
    float sum = 0.0;
    float maxVal = input.row(i).maxCoeff();
    Eigen::MatrixXf expInput = (input.row(i).array() - maxVal).exp();
    sum = expInput.sum();
    result.row(i) = expInput / sum;
  }
  return result;
}
int clip_postprocess(Eigen::MatrixXf& text_features, Eigen::MatrixXf& image_features,
                     Eigen::MatrixXf& prods) {
  normalize_matrix(image_features);
  normalize_matrix(text_features);
  Eigen::MatrixXf text_features_transposed = text_features.transpose();
  Eigen::MatrixXf result = 100.0 * image_features * text_features_transposed;
  prods = softmax(result);
  return 0;
}

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
  std::ofstream outfile("a2_output.txt");
  std::string input_image_path;
  cvtdl_clip_feature clip_feature_image;

  std::cout << image_file_list.size() << std::endl;

  int rows_image = image_file_list.size();
  int cols_image = 512;

  Eigen::MatrixXf clip_image_output(rows_image, cols_image);

  outfile << "********************image feature*********************************\n";
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
    std::cout << clip_feature_image.feature_dim << std::endl;
    for (int y = 0; y < clip_feature_image.feature_dim; ++y) {
      clip_image_output(i, y) = clip_feature_image.out_feature[y];
      outfile << clip_image_output(i, y);
      if (y < clip_feature_image.feature_dim - 1) {
        outfile << " ";
      }
    }

    free(clip_feature_image.out_feature);
    std::cout << "after free:" << std::endl;

    CVI_TDL_ReleaseImage(img_handle, &rgb_frame);
  }
  outfile << "********************text feature*********************************\n";

  ret = CVI_TDL_CreateHandle(&tdl_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create tdl handle failed with %#x!\n", ret);
    return ret;
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

  int rows_text = text_file_list.size();
  int cols_text = 512;
  Eigen::MatrixXf clip_text_output(rows_text, cols_text);

  std::string encoderFile = "./encoder.txt";
  std::string bpeFile = "./bpe_simple_vocab_16e6.txt";

  std::vector<std::vector<int32_t>> tokens;
  int result = cvitdl::token_bpe(encoderFile, bpeFile, text_list, tokens);
  if (result != 0) {
    printf("Tokenization error\n");
    return 0;
  }
  for (int i = 0; i < text_file_list.size(); i++) {
    CVI_U8 buffer[tokens[0].size() * sizeof(int32_t)];
    memcpy(buffer, &tokens[i][0], sizeof(int32_t) * tokens[0].size());
    VIDEO_FRAME_INFO_S Frame;
    Frame.stVFrame.pu8VirAddr[0] = buffer;
    Frame.stVFrame.u32Height = 1;
    Frame.stVFrame.u32Width = tokens[0].size();

    ret = CVI_TDL_Clip_Text_Feature(tdl_handle, &Frame, &clip_feature_text);
    if (ret != CVI_SUCCESS) {
      printf("CVI_TDL_OpenClip_Text_Feature\n");
      return 0;
    }

    for (int y = 0; y < clip_feature_text.feature_dim; ++y) {
      clip_text_output(i, y) = clip_feature_text.out_feature[y];
      outfile << clip_text_output(i, y);
      if (y < clip_feature_text.feature_dim - 1) {
        outfile << " ";
      }
    }
    outfile << "\n";
    free(clip_feature_text.out_feature);
  }
  outfile.close();

  Eigen::MatrixXf prods;
  int final_result = clip_postprocess(clip_text_output, clip_image_output, prods);

  std::ofstream out_prods("clip_pipeline_prods.txt");
  for (int i = 0; i < prods.rows(); ++i) {
    for (int j = 0; j < prods.cols(); ++j) {
      out_prods << prods(i, j) << " ";
    }
    out_prods << "\n";
  }
  out_prods.close();
  CVI_TDL_DestroyHandle(tdl_handle);
  return CVI_SUCCESS;
}
