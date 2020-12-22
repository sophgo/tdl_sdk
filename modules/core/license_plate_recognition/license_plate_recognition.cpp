#include "license_plate_recognition.hpp"
#include "decode_tool.hpp"

#include "core/cviai_types_mem.h"
#include "core/cviai_types_mem_internal.h"
#include "core_utils.hpp"
#include "face_utils.hpp"

#include "cvi_sys.h"
#include "opencv2/opencv.hpp"

#include <iostream>

#define LICENSE_PLATE_HEIGHT 24
#define LICENSE_PLATE_WIDTH 94

#define MEAN_R 0.485
#define MEAN_G 0.456
#define MEAN_B 0.406
#define STD_R 0.229
#define STD_G 0.224
#define STD_B 0.225

#define OUTPUT_NAME "id_code_ReduceMean_dequant"

#include <sstream>

namespace cviai {

LicensePlateRecognition::LicensePlateRecognition() {
  mp_mi = std::make_unique<CvimodelInfo>();
  mp_mi->conf.input_mem_type = CVI_MEM_SYSTEM;
}

LicensePlateRecognition::~LicensePlateRecognition() {}

int LicensePlateRecognition::inference(VIDEO_FRAME_INFO_S *frame,
                                       cvai_object_t *license_plate_meta) {
  // std::cout << "LicensePlateRecognition::inference" << std::endl;
  frame->stVFrame.pu8VirAddr[0] =
      (CVI_U8 *)CVI_SYS_MmapCache(frame->stVFrame.u64PhyAddr[0], frame->stVFrame.u32Length[0]);
  cv::Mat cv_frame(frame->stVFrame.u32Height, frame->stVFrame.u32Width, CV_8UC3,
                   frame->stVFrame.pu8VirAddr[0], frame->stVFrame.u32Stride[0]);
  // cv::imwrite("tmp_frame.jpg", cv_frame);
  std::stringstream s_str;
  for (size_t n = 0; n < license_plate_meta->size; n++) {
    // if (n > 0){
    //   continue;
    // }
    // std::cout << "n = " << n << std::endl;
    if (license_plate_meta->info[n].bpts.size == 0) {
      continue;
    }
    cv::Point2f src_points[4] = {
        cv::Point2f(license_plate_meta->info[n].bpts.x[0], license_plate_meta->info[n].bpts.y[0]),
        cv::Point2f(license_plate_meta->info[n].bpts.x[1], license_plate_meta->info[n].bpts.y[1]),
        cv::Point2f(license_plate_meta->info[n].bpts.x[2], license_plate_meta->info[n].bpts.y[2]),
        cv::Point2f(license_plate_meta->info[n].bpts.x[3], license_plate_meta->info[n].bpts.y[3]),
    };
    cv::Point2f dst_points[4] = {
        cv::Point2f(0, 0),
        cv::Point2f(LICENSE_PLATE_WIDTH, 0),
        cv::Point2f(LICENSE_PLATE_WIDTH, LICENSE_PLATE_HEIGHT),
        cv::Point2f(0, LICENSE_PLATE_HEIGHT),
    };
    cv::Mat M = cv::getPerspectiveTransform(src_points, dst_points);
    cv::Mat sub_cvFrame;
    cv::warpPerspective(cv_frame, sub_cvFrame, M,
                        cv::Size(LICENSE_PLATE_WIDTH, LICENSE_PLATE_HEIGHT), cv::INTER_LINEAR);
    cv::Mat greyMat;
    cv::cvtColor(sub_cvFrame, greyMat, cv::COLOR_RGB2GRAY); /* BGR or RGB ? */
    cv::cvtColor(greyMat, sub_cvFrame, cv::COLOR_GRAY2RGB);
    // s_str.str("");
    // s_str << "tmp_license_plate_" << n << ".jpg";
    // cv::imwrite(s_str.str().c_str(), greyMat);
    sub_cvFrame.convertTo(sub_cvFrame, CV_32FC3);

    sub_cvFrame = sub_cvFrame / 255.0;
    float mu[3] = {MEAN_R, MEAN_G, MEAN_B};
    float sigma[3] = {STD_R, STD_G, STD_B};
    std::vector<cv::Mat> rgbChannels(3);
    cv::split(sub_cvFrame, rgbChannels);
    for (int c = 0; c < 3; c++) {
      rgbChannels[c] = (rgbChannels[c] - mu[c]) / sigma[c];
    }
    // no-use cv::merge(rgbChannels, cv_frame);

    CVI_TENSOR *input =
        CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, mp_mi->in.tensors, mp_mi->in.num);

    uint16_t *input_ptr = (uint16_t *)CVI_NN_TensorPtr(input);
    int rows = sub_cvFrame.rows;
    int cols = sub_cvFrame.cols;
    for (int c = 0; c < 3; c++) {
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          uint16_t bf16_input = 0;
          floatToBF16((float *)rgbChannels[c].ptr(i, j), &bf16_input);
          memcpy(input_ptr + rows * cols * c + cols * i + j, &bf16_input, sizeof(uint16_t));
        }
      }
    }
    std::vector<VIDEO_FRAME_INFO_S *> frames = {frame};
    run(frames);

    CVI_TENSOR *out_tensor =
        CVI_NN_GetTensorByName(OUTPUT_NAME, mp_mi->out.tensors, mp_mi->out.num);

    float *out_code = (float *)CVI_NN_TensorPtr(out_tensor);

    std::string id_number = greedy_decode(out_code);

    // std::cout << "ID Number: " << id_number << std::endl;

    strncpy(license_plate_meta->info[n].name, id_number.c_str(),
            sizeof(license_plate_meta->info[n].name));
    // strcpy(license_plate_meta->info[n].name, id_number.c_str());
  }
  return CVI_SUCCESS;
}

}  // namespace cviai
