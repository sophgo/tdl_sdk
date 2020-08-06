#include "mask_face_recognition.hpp"

#include "core/cviai_types_free.h"
#include "core/face/cvai_face_helper.h"
#include "core_utils.hpp"
#include "face_utils.hpp"

#include "cvi_sys.h"
#include "opencv2/opencv.hpp"

#define FACE_ATTRIBUTE_MEAN (-0.99609375)
#define FACE_ATTRIBUTE_INPUT_THRESHOLD (1 / 128.0)

#define FACE_OUT_NAME "pre_fc1"

namespace cviai {

MaskFaceRecognition::MaskFaceRecognition() {
  mp_config = std::make_unique<ModelConfig>();
  mp_config->skip_postprocess = true;
}

MaskFaceRecognition::~MaskFaceRecognition() {}

int MaskFaceRecognition::inference(VIDEO_FRAME_INFO_S *frame, cvai_face_t *meta) {
  uint32_t img_width = frame->stVFrame.u32Width;
  uint32_t img_height = frame->stVFrame.u32Height;
  cv::Mat image(img_height, img_width, CV_8UC3);
  frame->stVFrame.pu8VirAddr[0] =
      (CVI_U8 *)CVI_SYS_Mmap(frame->stVFrame.u64PhyAddr[0], frame->stVFrame.u32Length[0]);
  char *va_rgb = (char *)frame->stVFrame.pu8VirAddr[0];
  uint32_t dst_width = image.cols;
  uint32_t dst_height = image.rows;

  for (size_t i = 0; i < (size_t)dst_height; i++) {
    memcpy(image.ptr(i, 0), va_rgb + frame->stVFrame.u32Stride[0] * i, dst_width * 3);
  }

  CVI_SYS_Munmap((void *)frame->stVFrame.pu8VirAddr[0], frame->stVFrame.u32Length[0]);

  for (int i = 0; i < meta->size; ++i) {
    cvai_face_info_t face_info =
        bbox_rescale(frame->stVFrame.u32Width, frame->stVFrame.u32Height, meta, i);

    prepareInputTensor(image, face_info);
    run(frame);
    outputParser(meta, i);
    CVI_AI_FreeCpp(&face_info);
  }
  return CVI_SUCCESS;
}

void MaskFaceRecognition::prepareInputTensor(cv::Mat src_image, cvai_face_info_t &face_info) {
  CVI_TENSOR *input = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, mp_input_tensors, m_input_num);
  cv::Mat image(input->shape.dim[2], input->shape.dim[3], src_image.type());

  face_align(src_image, image, face_info, input->shape.dim[3], input->shape.dim[2]);

  cv::Mat tmpchannels[3];
  cv::split(image, tmpchannels);

  for (int i = 0; i < 3; ++i) {
    tmpchannels[i].convertTo(tmpchannels[i], CV_32F, FACE_ATTRIBUTE_INPUT_THRESHOLD,
                             FACE_ATTRIBUTE_MEAN);

    int size = tmpchannels[i].rows * tmpchannels[i].cols;
    for (int r = 0; r < tmpchannels[i].rows; ++r) {
      memcpy((float *)CVI_NN_TensorPtr(input) + size * i + tmpchannels[i].cols * r,
             tmpchannels[i].ptr(r, 0), tmpchannels[i].cols * sizeof(float));
    }
  }
}

void MaskFaceRecognition::outputParser(cvai_face_t *meta, int meta_i) {
  CVI_TENSOR *out = CVI_NN_GetTensorByName(FACE_OUT_NAME, mp_output_tensors, m_output_num);
  int8_t *face_blob = (int8_t *)CVI_NN_TensorPtr(out);
  size_t face_feature_size = CVI_NN_TensorCount(out);

  CVI_AI_FreeCpp(&meta->face_info[meta_i].face_feature);
  meta->face_info[meta_i].face_feature.ptr = (int8_t *)malloc(sizeof(int8_t) * face_feature_size);
  meta->face_info[meta_i].face_feature.size = face_feature_size;
  meta->face_info[meta_i].face_feature.type = TYPE_INT8;
  memcpy(meta->face_info[meta_i].face_feature.ptr, face_blob, face_feature_size);
}

}  // namespace cviai