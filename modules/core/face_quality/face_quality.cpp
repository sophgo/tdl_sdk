#include "face_quality.hpp"

#include "core/core/cvai_errno.h"
#include "core/cviai_types_mem.h"
#include "core/utils/vpss_helper.h"
#include "core_utils.hpp"
#include "cvi_sys.h"
#include "face_utils.hpp"
#include "image_utils.hpp"

// include core_c.h if opencv version greater than 4.5
#if CV_VERSION_MAJOR >= 4 && CV_VERSION_MINOR >= 5
#include "opencv2/core/core_c.h"
#endif

#ifdef ENABLE_CVIAI_CV_UTILS
#include "cv/imgproc.hpp"
#else
#include "opencv2/imgproc.hpp"
#endif

#define SCALE_R (1.0 / (255.0 * 0.229))
#define SCALE_G (1.0 / (255.0 * 0.224))
#define SCALE_B (1.0 / (255.0 * 0.225))
#define MEAN_R (0.485 / 0.229)
#define MEAN_G (0.456 / 0.224)
#define MEAN_B (0.406 / 0.225)
#define NAME_SCORE "score_Softmax_dequant"

static bool IS_SUPPORTED_FORMAT(VIDEO_FRAME_INFO_S *frame) {
  if (frame->stVFrame.enPixelFormat != PIXEL_FORMAT_RGB_888 &&
      frame->stVFrame.enPixelFormat != PIXEL_FORMAT_YUV_PLANAR_420 &&
      frame->stVFrame.enPixelFormat != PIXEL_FORMAT_NV21) {
    LOGE("Pixel format [%d] not match PIXEL_FORMAT_RGB_888 [%d], PIXEL_FORMAT_NV21 [%d].\n",
         frame->stVFrame.enPixelFormat, PIXEL_FORMAT_RGB_888, PIXEL_FORMAT_NV21);
    return false;
  }
  return true;
}

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
  if (false == IS_SUPPORTED_FORMAT(frame)) {
    return CVIAI_ERR_INVALID_ARGS;
  }

  CVI_U32 frame_size =
      frame->stVFrame.u32Length[0] + frame->stVFrame.u32Length[1] + frame->stVFrame.u32Length[2];
  bool do_unmap = false;
  if (frame->stVFrame.pu8VirAddr[0] == NULL) {
    frame->stVFrame.pu8VirAddr[0] =
        (CVI_U8 *)CVI_SYS_MmapCache(frame->stVFrame.u64PhyAddr[0], frame_size);
    frame->stVFrame.pu8VirAddr[1] = frame->stVFrame.pu8VirAddr[0] + frame->stVFrame.u32Length[0];
    frame->stVFrame.pu8VirAddr[2] = frame->stVFrame.pu8VirAddr[1] + frame->stVFrame.u32Length[1];
    do_unmap = true;
  }

  int ret = CVIAI_SUCCESS;
  for (uint32_t i = 0; i < meta->size; i++) {
    if (skip != NULL && skip[i]) {
      continue;
    }
    cvai_face_info_t face_info =
        info_rescale_c(frame->stVFrame.u32Width, frame->stVFrame.u32Height, *meta, i);
    ALIGN_FACE_TO_FRAME(frame, &m_wrap_frame, face_info);

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
    CVI_SYS_Munmap((void *)frame->stVFrame.pu8VirAddr[0], frame_size);
    frame->stVFrame.pu8VirAddr[0] = NULL;
    frame->stVFrame.pu8VirAddr[1] = NULL;
    frame->stVFrame.pu8VirAddr[2] = NULL;
  }
  return ret;
}

}  // namespace cviai
