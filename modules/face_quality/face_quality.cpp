#include "face_quality.hpp"
#include "core_utils.hpp"
#include "face_utils.hpp"

#include "cvi_sys.h"
#include "opencv2/opencv.hpp"

#define SCALE_B (1.0 / (255.0 * 0.229))
#define SCALE_G (1.0 / (255.0 * 0.224))
#define SCALE_R (1.0 / (255.0 * 0.225))
#define MEAN_B -(0.485 / 0.229)
#define MEAN_G -(0.456 / 0.224)
#define MEAN_R -(0.406 / 0.225)
#define NAME_SCORE "score_Softmax_dequant"

namespace cviai {

FaceQuality::FaceQuality() { mp_config = std::make_unique<ModelConfig>(); }

int FaceQuality::inference(VIDEO_FRAME_INFO_S *frame, cvai_face_t *meta) {
  int img_width = frame->stVFrame.u32Width;
  int img_height = frame->stVFrame.u32Height;
  cv::Mat image(img_height, img_width, CV_8UC3);
  frame->stVFrame.pu8VirAddr[0] =
      (CVI_U8 *)CVI_SYS_Mmap(frame->stVFrame.u64PhyAddr[0], frame->stVFrame.u32Length[0]);
  char *va_rgb = (char *)frame->stVFrame.pu8VirAddr[0];
  int dst_width = image.cols;
  int dst_height = image.rows;

  for (int i = 0; i < dst_height; i++) {
    memcpy(image.ptr(i, 0), va_rgb + frame->stVFrame.u32Stride[0] * i, dst_width * 3);
  }
  CVI_SYS_Munmap((void *)frame->stVFrame.pu8VirAddr[0], frame->stVFrame.u32Length[0]);

  CVI_TENSOR *input = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, mp_input_tensors, m_input_num);

  std::vector<float> mean = {MEAN_B, MEAN_G, MEAN_R};
  std::vector<float> scale = {SCALE_B, SCALE_G, SCALE_R};
  for (int i = 0; i < meta->size; i++) {
    cvai_face_info_t face_info = bbox_rescale(frame, meta, i);
    cv::Mat crop_frame(input->shape.dim[3], input->shape.dim[2], image.type());
    face_align(image, crop_frame, face_info, input->shape.dim[2], input->shape.dim[3]);

    cv::Mat tmpchannels[3];
    cv::split(crop_frame, tmpchannels);

    for (int i = 0; i < 3; i++) {
      tmpchannels[i].convertTo(tmpchannels[i], CV_32F, scale[i], mean[i]);
      int size = tmpchannels[i].rows * tmpchannels[i].cols;
      for (int r = 0; r < tmpchannels[i].rows; ++r) {
        memcpy((float *)CVI_NN_TensorPtr(input) + size * i + tmpchannels[i].cols * r,
               tmpchannels[i].ptr(r, 0), tmpchannels[i].cols * sizeof(float));
      }
    }

    run(frame);

    CVI_TENSOR *out = CVI_NN_GetTensorByName(NAME_SCORE, mp_output_tensors, m_output_num);
    float *score = (float *)CVI_NN_TensorPtr(out);
    meta->face_info[i].face_quality = score[1];
    // cout << score[0] << "," << score[1] << endl;
  }
  // sleep(3);

  return CVI_RC_SUCCESS;
}

}  // namespace cviai
