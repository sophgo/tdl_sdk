#include "face_landmarker.hpp"

#include "core/cviai_types_mem.h"
#include "core/cviai_types_mem_internal.h"
#include "core/utils/vpss_helper.h"
#include "core_utils.hpp"
#include "face_utils.hpp"

#include "cvi_sys.h"
#include "opencv2/opencv.hpp"

#define OPENEYERECOGNIZE_SCALE (1.0 / (255.0))
#define NAME_SCORE "fc1_Gemm_dequant"
#define INPUT_SIZE 192

namespace cviai {

FaceLandmarker::FaceLandmarker() : Core(CVI_MEM_SYSTEM) {}

int FaceLandmarker::inference(VIDEO_FRAME_INFO_S *frame, cvai_face_t *meta) {
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

    int max_side = 0;
    Preprocessing(&face_info, &max_side, img_width, img_height);

    // crop image and resize to input size
    cv::Mat roi_img =
        image(cv::Rect(face_info.bbox.x1, face_info.bbox.y1, face_info.bbox.x2 - face_info.bbox.x1,
                       face_info.bbox.y2 - face_info.bbox.y1));
    cv::resize(roi_img, roi_img, cv::Size(INPUT_SIZE, INPUT_SIZE), 0, 0, cv::INTER_CUBIC);
    cv::cvtColor(roi_img, roi_img, cv::COLOR_BGR2RGB);
    prepareInputTensor(roi_img);

    std::vector<VIDEO_FRAME_INFO_S *> frames = {frame};
    run(frames);

    const TensorInfo &tinfo = getOutputTensorInfo(NAME_SCORE);
    float *pts = tinfo.get<float>();
    size_t pts_size = tinfo.tensor_elem;
    cvai_pts_t landmarks;
    landmarks.size = 106;
    landmarks.x = (float *)malloc(sizeof(float) * 106);
    landmarks.y = (float *)malloc(sizeof(float) * 106);

    // change the [1,-1] coordination to image coordination
    for (int i = 0; i < (int)pts_size / 2; ++i) {
      landmarks.x[i] = (pts[2 * i] + 1.0f) * max_side / 2 + face_info.bbox.x1;
      landmarks.y[i] = (pts[2 * i + 1] + 1.0f) * max_side / 2 + face_info.bbox.y1;
    }
    meta->info[i].dms->landmarks = landmarks;
    CVI_AI_FreeCpp(&face_info);
  }

  CVI_SYS_Munmap((void *)frame->stVFrame.pu8VirAddr[0], frame->stVFrame.u32Length[0]);
  frame->stVFrame.pu8VirAddr[0] = NULL;
  frame->stVFrame.pu8VirAddr[1] = NULL;
  frame->stVFrame.pu8VirAddr[2] = NULL;

  return CVI_SUCCESS;
}

void FaceLandmarker::prepareInputTensor(cv::Mat &input_mat) {
  const TensorInfo &tinfo = getInputTensorInfo(0);
  uint16_t *input_ptr = tinfo.get<uint16_t>();
  cv::Mat temp_mat;
  input_mat.convertTo(temp_mat, CV_32FC3);
  int size = temp_mat.rows * temp_mat.cols;
  cv::Mat tmpchannels[3];
  cv::split(temp_mat, tmpchannels);
  for (int i = 0; i < 3; ++i) {
    for (int r = 0; r < temp_mat.rows; ++r) {
      for (int c = 0; c < temp_mat.cols; ++c) {
        uint16_t bf16_input = 0;
        floatToBF16((float *)tmpchannels[i].ptr(r, c), &bf16_input);
        memcpy(input_ptr + input_mat.cols * r + c + i * size, &bf16_input, sizeof(uint16_t));
      }
    }
  }
}

void FaceLandmarker::Preprocessing(cvai_face_info_t *face_info, int *max_side, int img_width,
                                   int img_height) {
  // scale to 1.5 times
  float half_width = (face_info->bbox.x2 - face_info->bbox.x1) / 4;
  float half_height = (face_info->bbox.y2 - face_info->bbox.y1) / 4;
  face_info->bbox.x1 = face_info->bbox.x1 - half_width;
  face_info->bbox.x2 = face_info->bbox.x2 + half_width;
  face_info->bbox.y1 = face_info->bbox.y1 - half_height;
  face_info->bbox.y2 = face_info->bbox.y2 + half_height;

  // square the roi
  *max_side =
      std::max(face_info->bbox.x2 - face_info->bbox.x1, face_info->bbox.y2 - face_info->bbox.y1);
  int offset_x = (*max_side - (int)face_info->bbox.x2 + face_info->bbox.x1) / 2;
  int offset_y = (*max_side - (int)face_info->bbox.y2 + face_info->bbox.y1) / 2;
  face_info->bbox.x1 = std::max(((int)face_info->bbox.x1 - offset_x), 0);
  face_info->bbox.x2 = std::min(((int)face_info->bbox.x1 + *max_side), img_width);
  face_info->bbox.y1 = std::max(((int)face_info->bbox.y1 - offset_y), 0);
  face_info->bbox.y2 = std::min(((int)face_info->bbox.y1 + *max_side), img_height);
}
}  // namespace cviai
