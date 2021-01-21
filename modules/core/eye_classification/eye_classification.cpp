#include "eye_classification.hpp"

#include "core/cviai_types_mem.h"
#include "core/cviai_types_mem_internal.h"
#include "core/utils/vpss_helper.h"
#include "core_utils.hpp"
#include "face_utils.hpp"

#include "cvi_sys.h"
#include "opencv2/opencv.hpp"

#define EYECLASSIFICATION_SCALE (1.0 / (255.0))
#define NAME_SCORE "prob_Sigmoid_dequant"

namespace cviai {

EyeClassification::EyeClassification() : Core(CVI_MEM_SYSTEM) {}

int EyeClassification::inference(VIDEO_FRAME_INFO_S *frame, cvai_face_t *meta) {
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

  cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);
  // just one face
  for (uint32_t i = 0; i < 1; i++) {
    cvai_face_info_t face_info =
        info_rescale_c(frame->stVFrame.u32Width, frame->stVFrame.u32Height, *meta, i);
    int eye_w = abs(face_info.pts.x[0] - face_info.pts.x[1]);

    float q_w = eye_w / 3;

    cv::Rect r_roi(int(std::max(face_info.pts.x[0] - q_w, face_info.bbox.x1)),
                   int(std::max(face_info.pts.y[0] - q_w, face_info.bbox.y1)),
                   int(std::min(face_info.pts.x[0] + q_w, face_info.bbox.x2)) -
                       int(std::max(face_info.pts.x[0] - q_w, face_info.bbox.x1)),
                   int(std::min(face_info.pts.y[0] + q_w, face_info.bbox.y2)) -
                       int(std::max(face_info.pts.y[0] - q_w, face_info.bbox.y1)));

    cv::Rect l_roi(int(std::max(face_info.pts.x[1] - q_w, face_info.bbox.x1)),
                   int(std::max(face_info.pts.y[1] - q_w, face_info.bbox.y1)),
                   int(std::min(face_info.pts.x[1] + q_w, face_info.bbox.x2)) -
                       int(std::max(face_info.pts.x[1] - q_w, face_info.bbox.x1)),
                   int(std::min(face_info.pts.y[1] + q_w, face_info.bbox.y2)) -
                       int(std::max(face_info.pts.y[1] - q_w, face_info.bbox.y1)));

    if (r_roi.width < 10 || r_roi.height < 10) {  // small images filter
      meta->info[i].r_eye_score = 0.0;
    } else {
      cv::Mat r_Image = image(r_roi);
      cv::resize(r_Image, r_Image, cv::Size(32, 32));
      prepareInputTensor(r_Image);

      std::vector<VIDEO_FRAME_INFO_S *> frames = {frame};
      run(frames);

      float *score = getOutputRawPtr<float>(NAME_SCORE);
      meta->info[i].r_eye_score = score[0];
    }
    // left eye patch
    if (l_roi.width < 10 || l_roi.height < 10) {  // mall images filter
      meta->info[i].l_eye_score = 0.0;
    } else {
      cv::Mat l_Image = image(l_roi);
      cv::resize(l_Image, l_Image, cv::Size(32, 32));
      prepareInputTensor(l_Image);

      std::vector<VIDEO_FRAME_INFO_S *> frames = {frame};
      run(frames);

      float *score = getOutputRawPtr<float>(NAME_SCORE);
      meta->info[i].l_eye_score = score[0];
    }
    CVI_AI_FreeCpp(&face_info);
  }

  CVI_SYS_Munmap((void *)frame->stVFrame.pu8VirAddr[0], frame->stVFrame.u32Length[0]);
  frame->stVFrame.pu8VirAddr[0] = NULL;
  frame->stVFrame.pu8VirAddr[1] = NULL;
  frame->stVFrame.pu8VirAddr[2] = NULL;

  return CVI_SUCCESS;
}

void EyeClassification::prepareInputTensor(cv::Mat &input_mat) {
  const TensorInfo &tinfo = getInputTensorInfo(0);
  uint16_t *input_ptr = tinfo.get<uint16_t>();
  cv::Mat temp_mat;
  input_mat.convertTo(temp_mat, CV_32FC1, EYECLASSIFICATION_SCALE * 1.0, 0);
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
