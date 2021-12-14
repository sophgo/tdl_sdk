#include "face_quality.hpp"

#include "core/core/cvai_errno.h"
#include "core/cviai_types_mem.h"
#include "core/utils/vpss_helper.h"
#include "core_utils.hpp"
#include "cvi_sys.h"
#include "face_utils.hpp"

// include core_c.h if opencv version greater than 4.5
#if CV_VERSION_MAJOR >= 4 && CV_VERSION_MINOR >= 5
#include "opencv2/core/core_c.h"
#endif

#include "opencv2/opencv.hpp"

#define SCALE_R (1.0 / (255.0 * 0.229))
#define SCALE_G (1.0 / (255.0 * 0.224))
#define SCALE_B (1.0 / (255.0 * 0.225))
#define MEAN_R (0.485 / 0.229)
#define MEAN_G (0.456 / 0.224)
#define MEAN_B (0.406 / 0.225)
#define NAME_SCORE "score_Softmax_dequant"

namespace cviai {

FaceQuality::FaceQuality() : Core(CVI_MEM_DEVICE) {}

FaceQuality::~FaceQuality() {}

int FaceQuality::setupInputPreprocess(std::vector<InputPreprecessSetup> *data) {
  if (data->size() != 1) {
    LOGE("Face quality only has 1 input.\n");
    return CVIAI_ERR_INVALID_ARGS;
  }

  std::vector<float> mean = {MEAN_R, MEAN_G, MEAN_B};
  std::vector<float> scale = {SCALE_R, SCALE_G, SCALE_B};
  for (uint32_t i = 0; i < 3; i++) {
    (*data)[0].factor[i] = scale[i];
    (*data)[0].mean[i] = mean[i];
  }
  (*data)[0].use_quantize_scale = true;

  return CVIAI_SUCCESS;
}

int FaceQuality::onModelOpened() { return allocateION(); }

int FaceQuality::onModelClosed() {
  releaseION();
  return CVIAI_SUCCESS;
}

CVI_S32 FaceQuality::allocateION() {
  CVI_SHAPE shape = getInputShape(0);
  if (CREATE_ION_HELPER(&m_wrap_frame, shape.dim[3], shape.dim[2], PIXEL_FORMAT_RGB_888, "tpu") !=
      CVI_SUCCESS) {
    LOGE("Cannot allocate ion for preprocess\n");
    return CVIAI_ERR_ALLOC_ION_FAIL;
  }
  return CVIAI_SUCCESS;
}

void FaceQuality::releaseION() {
  if (m_wrap_frame.stVFrame.u64PhyAddr[0] != 0) {
    CVI_SYS_IonFree(m_wrap_frame.stVFrame.u64PhyAddr[0], m_wrap_frame.stVFrame.pu8VirAddr[0]);
    m_wrap_frame.stVFrame.u64PhyAddr[0] = (CVI_U64)0;
    m_wrap_frame.stVFrame.u64PhyAddr[1] = (CVI_U64)0;
    m_wrap_frame.stVFrame.u64PhyAddr[2] = (CVI_U64)0;
    m_wrap_frame.stVFrame.pu8VirAddr[0] = NULL;
    m_wrap_frame.stVFrame.pu8VirAddr[1] = NULL;
    m_wrap_frame.stVFrame.pu8VirAddr[2] = NULL;
  }
}

int FaceQuality::inference(VIDEO_FRAME_INFO_S *frame, cvai_face_t *meta, bool *skip) {
  if (frame->stVFrame.enPixelFormat != PIXEL_FORMAT_RGB_888) {
    LOGE("Error: pixel format not match PIXEL_FORMAT_RGB_888.\n");
    return CVIAI_ERR_INVALID_ARGS;
  }

  int img_width = frame->stVFrame.u32Width;
  int img_height = frame->stVFrame.u32Height;
  bool do_unmap = false;
  if (frame->stVFrame.pu8VirAddr[0] == NULL) {
    frame->stVFrame.pu8VirAddr[0] =
        (CVI_U8 *)CVI_SYS_MmapCache(frame->stVFrame.u64PhyAddr[0], frame->stVFrame.u32Length[0]);
    do_unmap = true;
  }
  cv::Mat image(img_height, img_width, CV_8UC3, frame->stVFrame.pu8VirAddr[0],
                frame->stVFrame.u32Stride[0]);
  int ret = CVIAI_SUCCESS;
  for (uint32_t i = 0; i < meta->size; i++) {
    if (skip != NULL && skip[i]) {
      continue;
    }
    cvai_face_info_t face_info =
        info_rescale_c(frame->stVFrame.u32Width, frame->stVFrame.u32Height, *meta, i);
    cv::Mat warp_image(cv::Size(m_wrap_frame.stVFrame.u32Width, m_wrap_frame.stVFrame.u32Height),
                       image.type(), m_wrap_frame.stVFrame.pu8VirAddr[0],
                       m_wrap_frame.stVFrame.u32Stride[0]);

    face_align(image, warp_image, face_info);

    std::vector<VIDEO_FRAME_INFO_S *> frames = {&m_wrap_frame};
    ret = run(frames);
    if (ret != CVIAI_SUCCESS) {
      return ret;
    }

    float *score = getOutputRawPtr<float>(NAME_SCORE);
    meta->info[i].face_quality = score[1];

    CVI_AI_FreeCpp(&face_info);
  }
  if (do_unmap) {
    CVI_SYS_Munmap((void *)frame->stVFrame.pu8VirAddr[0], frame->stVFrame.u32Length[0]);
    frame->stVFrame.pu8VirAddr[0] = NULL;
    frame->stVFrame.pu8VirAddr[1] = NULL;
    frame->stVFrame.pu8VirAddr[2] = NULL;
  }
  return ret;
}

int FaceQuality::getAlignedFace(VIDEO_FRAME_INFO_S *srcFrame, VIDEO_FRAME_INFO_S *dstFrame,
                                cvai_face_info_t *face_info) {
  if (srcFrame->stVFrame.enPixelFormat != PIXEL_FORMAT_RGB_888) {
    LOGE("Error: pixel format not match PIXEL_FORMAT_RGB_888.\n");
    return CVIAI_ERR_INVALID_ARGS;
  }
  cvai_face_info_t face_info_rescale =
      info_rescale_c(srcFrame->stVFrame.u32Width, srcFrame->stVFrame.u32Height,
                     srcFrame->stVFrame.u32Width, srcFrame->stVFrame.u32Height, *face_info);
  bool do_unmap = false;
  if (srcFrame->stVFrame.pu8VirAddr[0] == NULL) {
    srcFrame->stVFrame.pu8VirAddr[0] = (CVI_U8 *)CVI_SYS_MmapCache(srcFrame->stVFrame.u64PhyAddr[0],
                                                                   srcFrame->stVFrame.u32Length[0]);
    do_unmap = true;
  }
  cv::Mat image(srcFrame->stVFrame.u32Height, srcFrame->stVFrame.u32Width, CV_8UC3,
                srcFrame->stVFrame.pu8VirAddr[0], srcFrame->stVFrame.u32Stride[0]);
  cv::Mat warp_image(cv::Size(m_wrap_frame.stVFrame.u32Width, m_wrap_frame.stVFrame.u32Height),
                     image.type(), m_wrap_frame.stVFrame.pu8VirAddr[0],
                     m_wrap_frame.stVFrame.u32Stride[0]);
  if (face_align(image, warp_image, face_info_rescale) != 0) {
    return CVIAI_ERR_INFERENCE;
  }
  // cv::cvtColor(warp_image, warp_image, cv::COLOR_RGB2BGR);
  // cv::imwrite("visual/aligned_face.jpg", warp_image);

  std::vector<cv::Mat> rgbChannels(3);
  split(warp_image, rgbChannels);

  CVI_U32 u32Height = m_wrap_frame.stVFrame.u32Height;
  CVI_U32 u32Width = m_wrap_frame.stVFrame.u32Width;

  for (int chn = 0; chn < 3; chn++) {
    for (int i = 0; i < (int)u32Height; i++) {
      for (int j = 0; j < (int)u32Width; j++) {
        int idx = i * (int)u32Width + j;
        /* BGR to RGB */
        dstFrame->stVFrame.pu8VirAddr[2 - chn][idx] = rgbChannels[chn].at<uchar>(i, j);
      }
    }
  }
  if (do_unmap) {
    CVI_SYS_Munmap((void *)srcFrame->stVFrame.pu8VirAddr[0], srcFrame->stVFrame.u32Length[0]);
    srcFrame->stVFrame.pu8VirAddr[0] = NULL;
    srcFrame->stVFrame.pu8VirAddr[1] = NULL;
    srcFrame->stVFrame.pu8VirAddr[2] = NULL;
  }

  return CVIAI_SUCCESS;
}

}  // namespace cviai
