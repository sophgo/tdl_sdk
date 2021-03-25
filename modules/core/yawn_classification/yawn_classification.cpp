#include "yawn_classification.hpp"
#include "core/cviai_types_mem.h"
#include "core/cviai_types_mem_internal.h"
#include "core/utils/vpss_helper.h"
#include "core_utils.hpp"
#include "face_utils.hpp"

#define OPENEYERECOGNIZE_SCALE (1.0 / (255.0))
#define NAME_SCORE "prob_Sigmoid_dequant"
#define INPUT_SIZE 64

namespace cviai {

YawnClassification::YawnClassification() : Core(CVI_MEM_SYSTEM) {}

int YawnClassification::inference(VIDEO_FRAME_INFO_S *frame, cvai_face_t *meta) {
  if (frame->stVFrame.enPixelFormat != PIXEL_FORMAT_RGB_888) {
    LOGE("Error: pixel format not match PIXEL_FORMAT_RGB_888.\n");
    return CVI_FAILURE;
  }

  int img_width = frame->stVFrame.u32Width;
  int img_height = frame->stVFrame.u32Height;
  frame->stVFrame.pu8VirAddr[0] =
      (CVI_U8 *)CVI_SYS_MmapCache(frame->stVFrame.u64PhyAddr[0], frame->stVFrame.u32Length[0]);
  cv::Mat image(img_height, img_width, CV_8UC3, frame->stVFrame.pu8VirAddr[0],
                frame->stVFrame.u32Stride[0]);
  // just one face
  for (uint32_t i = 0; i < 1; i++) {
    cvai_face_info_t face_info =
        info_rescale_c(frame->stVFrame.u32Width, frame->stVFrame.u32Height, *meta, i);

    cv::Mat warp_image(cv::Size(112, 112), CV_8UC3);
    face_align(image, warp_image, face_info);
    cv::resize(warp_image, warp_image, cv::Size(INPUT_SIZE, INPUT_SIZE), 0, 0, cv::INTER_CUBIC);
    cv::cvtColor(warp_image, warp_image, cv::COLOR_RGB2GRAY);
    prepareInputTensor(warp_image);

    std::vector<VIDEO_FRAME_INFO_S *> frames = {frame};
    run(frames);

    float *score = getOutputRawPtr<float>(NAME_SCORE);
    meta->dms->yawn_score = score[0];
    CVI_AI_FreeCpp(&face_info);
  }

  CVI_SYS_Munmap((void *)frame->stVFrame.pu8VirAddr[0], frame->stVFrame.u32Length[0]);
  frame->stVFrame.pu8VirAddr[0] = NULL;
  frame->stVFrame.pu8VirAddr[1] = NULL;
  frame->stVFrame.pu8VirAddr[2] = NULL;

  return CVI_SUCCESS;
}

void YawnClassification::prepareInputTensor(cv::Mat &input_mat) {
  const TensorInfo &tinfo = getInputTensorInfo(0);
  uint16_t *input_ptr = tinfo.get<uint16_t>();
  cv::Mat temp_mat;
  input_mat.convertTo(temp_mat, CV_32FC1, OPENEYERECOGNIZE_SCALE * 1.0, 0);
  cv::add(-0.5, temp_mat, temp_mat);
  cv::multiply(temp_mat, cv::Scalar(2), temp_mat);
  for (int r = 0; r < temp_mat.rows; ++r) {
    for (int c = 0; c < temp_mat.cols; ++c) {
      uint16_t bf16_input = 0;
      floatToBF16((float *)temp_mat.ptr(r, c), &bf16_input);
      memcpy(input_ptr + temp_mat.cols * r + c, &bf16_input, sizeof(uint16_t));
    }
  }
}

}  // namespace cviai
