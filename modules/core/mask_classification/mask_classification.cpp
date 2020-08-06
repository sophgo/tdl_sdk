#include "mask_classification.hpp"

#include "core/cviai_types_free.h"
#include "face_utils.hpp"

#include "cvi_sys.h"
#include "opencv2/opencv.hpp"

#define R_SCALE (1 / (255.0 * 0.229))
#define G_SCALE (1 / (255.0 * 0.224))
#define B_SCALE (1 / (255.0 * 0.225))
#define R_MEAN (-0.485 / 0.229)
#define G_MEAN (-0.456 / 0.224)
#define B_MEAN (-0.406 / 0.225)
#define CROP_PCT 0.875
#define MASK_OUT_NAME "logits_dequant"

namespace cviai {

MaskClassification::MaskClassification() { mp_config = std::make_unique<ModelConfig>(); }

MaskClassification::~MaskClassification() {}

int MaskClassification::inference(VIDEO_FRAME_INFO_S *stOutFrame, cvai_face_t *meta) {
  uint32_t img_width = stOutFrame->stVFrame.u32Width;
  uint32_t img_height = stOutFrame->stVFrame.u32Height;
  cv::Mat image(img_height, img_width, CV_8UC3);
  stOutFrame->stVFrame.pu8VirAddr[0] =
      (CVI_U8 *)CVI_SYS_Mmap(stOutFrame->stVFrame.u64PhyAddr[0], stOutFrame->stVFrame.u32Length[0]);
  char *va_rgb = (char *)stOutFrame->stVFrame.pu8VirAddr[0];
  uint32_t dst_width = image.cols;
  uint32_t dst_height = image.rows;

  for (size_t i = 0; i < (size_t)dst_height; i++) {
    memcpy(image.ptr(i, 0), va_rgb + stOutFrame->stVFrame.u32Stride[0] * i, dst_width * 3);
  }

  CVI_SYS_Munmap((void *)stOutFrame->stVFrame.pu8VirAddr[0], stOutFrame->stVFrame.u32Length[0]);

  CVI_TENSOR *input = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, mp_input_tensors, m_input_num);
  int input_w = input->shape.dim[3];
  int input_h = input->shape.dim[2];
  for (int i = 0; i < meta->size; i++) {
    cvai_face_info_t face_info = bbox_rescale(img_width, img_height, meta, i);

    cv::Rect box;
    box.x = face_info.bbox.x1;
    box.y = face_info.bbox.y1;
    box.width = face_info.bbox.x2 - box.x;
    box.height = face_info.bbox.y2 - box.y;

    cv::Mat crop_image = image(box);
    cv::Mat resized_image;
    int min_edge = std::min(crop_image.rows, crop_image.cols);
    int size = floor(input_h / CROP_PCT);
    cv::resize(
        crop_image, resized_image,
        cv::Size(int(size * crop_image.cols / min_edge), int(size * crop_image.rows / min_edge)));

    cv::Rect central_roi(int((resized_image.cols - input_w) / 2),
                         int((resized_image.rows - input_h) / 2), input_w, input_h);
    cv::Mat central_image = resized_image(central_roi);

    prepareInputTensor(central_image);

    run(stOutFrame);

    CVI_TENSOR *out = CVI_NN_GetTensorByName(MASK_OUT_NAME, mp_output_tensors, m_output_num);
    float *out_data = (float *)CVI_NN_TensorPtr(out);

    float max = std::max(out_data[0], out_data[1]);
    float f0 = std::exp(out_data[0] - max);
    float f1 = std::exp(out_data[1] - max);
    float score = f0 / (f0 + f1);

    meta->face_info[i].mask_score = score;
  }

  return CVI_SUCCESS;
}

void MaskClassification::prepareInputTensor(cv::Mat &image) {
  CVI_TENSOR *input = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, mp_input_tensors, m_input_num);

  cv::Mat tmpchannels[3];
  cv::split(image, tmpchannels);

  std::vector<float> scale = {R_SCALE, G_SCALE, B_SCALE};
  std::vector<float> mean = {R_MEAN, G_MEAN, B_MEAN};

  int size = tmpchannels[0].rows * tmpchannels[0].cols;
  for (int i = 0; i < 3; ++i) {
    tmpchannels[i].convertTo(tmpchannels[i], CV_32F, scale[i], mean[i]);
    for (int r = 0; r < tmpchannels[i].rows; ++r) {
      memcpy((float *)CVI_NN_TensorPtr(input) + size * i + tmpchannels[i].cols * r,
             tmpchannels[i].ptr(r, 0), tmpchannels[i].cols * sizeof(float));
    }
  }
}

}  // namespace cviai