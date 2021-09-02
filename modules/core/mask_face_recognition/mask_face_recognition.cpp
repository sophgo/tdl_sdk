#include "mask_face_recognition.hpp"

#include "core/core/cvai_errno.h"
#include "core/cviai_types_mem.h"
#include "core/cviai_types_mem_internal.h"
#include "core/face/cvai_face_helper.h"
#include "core/utils/vpss_helper.h"
#include "core_utils.hpp"
#include "cvi_sys.h"
#include "face_utils.hpp"
#include "opencv2/opencv.hpp"

#define FACE_ATTRIBUTE_MEAN (0.99609375)
#define FACE_ATTRIBUTE_SCALE (1 / 128.f)

#define FACE_OUT_NAME "pre_fc1"

namespace cviai {

MaskFaceRecognition::MaskFaceRecognition() : Core(CVI_MEM_DEVICE) {}

MaskFaceRecognition::~MaskFaceRecognition() {
  if (m_gdc_blk != (VB_BLK)-1) {
    CVI_SYS_Munmap((void *)m_wrap_frame.stVFrame.pu8VirAddr[0], m_wrap_frame.stVFrame.u32Length[0]);
    m_wrap_frame.stVFrame.pu8VirAddr[0] = NULL;
    CVI_VB_ReleaseBlock(m_gdc_blk);
  }
}

int MaskFaceRecognition::setupInputPreprocess(std::vector<InputPreprecessSetup> *data) {
  if (data->size() != 1) {
    LOGE("Face attribute only has 1 input.\n");
    return CVIAI_ERR_INVALID_ARGS;
  }
  for (uint32_t i = 0; i < 3; i++) {
    (*data)[0].factor[i] = FACE_ATTRIBUTE_SCALE;
    (*data)[0].mean[i] = FACE_ATTRIBUTE_MEAN;
  }
  (*data)[0].use_quantize_scale = true;
  return CVIAI_SUCCESS;
}

int MaskFaceRecognition::onModelOpened() {
  CVI_SHAPE shape = getInputShape(0);
  if (CREATE_VBFRAME_HELPER(&m_gdc_blk, &m_wrap_frame, shape.dim[3], shape.dim[2],
                            PIXEL_FORMAT_RGB_888) != CVI_SUCCESS) {
    return CVIAI_ERR_OPEN_MODEL;
  }

  m_wrap_frame.stVFrame.pu8VirAddr[0] = (CVI_U8 *)CVI_SYS_MmapCache(
      m_wrap_frame.stVFrame.u64PhyAddr[0], m_wrap_frame.stVFrame.u32Length[0]);

  return CVIAI_SUCCESS;
}

int MaskFaceRecognition::inference(VIDEO_FRAME_INFO_S *frame, cvai_face_t *meta) {
  uint32_t img_width = frame->stVFrame.u32Width;
  uint32_t img_height = frame->stVFrame.u32Height;
  cv::Mat image(img_height, img_width, CV_8UC3);
  frame->stVFrame.pu8VirAddr[0] =
      (CVI_U8 *)CVI_SYS_MmapCache(frame->stVFrame.u64PhyAddr[0], frame->stVFrame.u32Length[0]);
  char *va_rgb = (char *)frame->stVFrame.pu8VirAddr[0];
  uint32_t dst_width = image.cols;
  uint32_t dst_height = image.rows;

  for (size_t i = 0; i < (size_t)dst_height; i++) {
    memcpy(image.ptr(i, 0), va_rgb + frame->stVFrame.u32Stride[0] * i, dst_width * 3);
  }

  CVI_SYS_Munmap((void *)frame->stVFrame.pu8VirAddr[0], frame->stVFrame.u32Length[0]);
  frame->stVFrame.pu8VirAddr[0] = NULL;

  for (uint32_t i = 0; i < meta->size; ++i) {
    cvai_face_info_t face_info =
        info_rescale_c(frame->stVFrame.u32Width, frame->stVFrame.u32Height, *meta, i);
    cv::Mat warp_image(cv::Size(m_wrap_frame.stVFrame.u32Width, m_wrap_frame.stVFrame.u32Height),
                       image.type(), m_wrap_frame.stVFrame.pu8VirAddr[0],
                       m_wrap_frame.stVFrame.u32Stride[0]);

    face_align(image, warp_image, face_info);
    CVI_SYS_IonFlushCache(m_wrap_frame.stVFrame.u64PhyAddr[0], m_wrap_frame.stVFrame.pu8VirAddr[0],
                          m_wrap_frame.stVFrame.u32Length[0]);

    std::vector<VIDEO_FRAME_INFO_S *> frames = {&m_wrap_frame};
    if (int ret = run(frames) != CVIAI_SUCCESS) {
      return ret;
    }
    outputParser(meta, i);
    CVI_AI_FreeCpp(&face_info);
  }
  return CVIAI_SUCCESS;
}

void MaskFaceRecognition::outputParser(cvai_face_t *meta, int meta_i) {
  int8_t *face_blob = getOutputRawPtr<int8_t>(FACE_OUT_NAME);
  size_t face_feature_size = getOutputTensorElem(FACE_OUT_NAME);

  CVI_AI_MemAlloc(sizeof(int8_t), face_feature_size, TYPE_INT8, &meta->info[meta_i].feature);
  memcpy(meta->info[meta_i].feature.ptr, face_blob, face_feature_size);
}

}  // namespace cviai