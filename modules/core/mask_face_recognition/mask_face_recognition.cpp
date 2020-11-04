#include "mask_face_recognition.hpp"

#include "core/cviai_types_mem.h"
#include "core/cviai_types_mem_internal.h"
#include "core/face/cvai_face_helper.h"
#include "core/utils/vpss_helper.h"
#include "core_utils.hpp"
#include "face_utils.hpp"

#include "cvi_sys.h"
#include "opencv2/opencv.hpp"

#define FACE_ATTRIBUTE_MEAN (0.99609375)
#define FACE_ATTRIBUTE_SCALE (1 / 128.f)

#define FACE_OUT_NAME "pre_fc1"

namespace cviai {

MaskFaceRecognition::MaskFaceRecognition() {
  mp_config = std::make_unique<ModelConfig>();
  mp_config->skip_postprocess = true;
  mp_config->input_mem_type = CVI_MEM_DEVICE;
}

MaskFaceRecognition::~MaskFaceRecognition() {}

int MaskFaceRecognition::initAfterModelOpened(std::vector<initSetup> *data) {
  if (data->size() != 1) {
    LOGE("Face attribute only has 1 input.\n");
    return CVI_FAILURE;
  }
  for (uint32_t i = 0; i < 3; i++) {
    (*data)[0].factor[i] = FACE_ATTRIBUTE_SCALE;
    (*data)[0].mean[i] = FACE_ATTRIBUTE_MEAN;
  }
  (*data)[0].use_quantize_scale = true;

  CVI_TENSOR *input = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, mp_input_tensors, m_input_num);
  if (CREATE_VBFRAME_HELPER(&m_gdc_blk, &m_wrap_frame, input->shape.dim[3], input->shape.dim[2],
                            PIXEL_FORMAT_RGB_888) != CVI_SUCCESS) {
    return -1;
  }

  m_wrap_frame.stVFrame.pu8VirAddr[0] = (CVI_U8 *)CVI_SYS_MmapCache(
      m_wrap_frame.stVFrame.u64PhyAddr[0], m_wrap_frame.stVFrame.u32Length[0]);

  return 0;
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
    run(frames);
    outputParser(meta, i);
    CVI_AI_FreeCpp(&face_info);
  }
  return CVI_SUCCESS;
}

void MaskFaceRecognition::outputParser(cvai_face_t *meta, int meta_i) {
  CVI_TENSOR *out = CVI_NN_GetTensorByName(FACE_OUT_NAME, mp_output_tensors, m_output_num);
  int8_t *face_blob = (int8_t *)CVI_NN_TensorPtr(out);
  size_t face_feature_size = CVI_NN_TensorCount(out);

  CVI_AI_MemAlloc(sizeof(int8_t), face_feature_size, TYPE_INT8, &meta->info[meta_i].feature);
  memcpy(meta->info[meta_i].feature.ptr, face_blob, face_feature_size);
}

}  // namespace cviai