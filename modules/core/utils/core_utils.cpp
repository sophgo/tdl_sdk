#include "core_utils.hpp"
#include "cviai_log.hpp"
#include "neon_utils.hpp"

#include <math.h>
#include <string.h>
#include <algorithm>

namespace cviai {
void SoftMaxForBuffer(const float *src, float *dst, size_t size) {
  float sum = 0;

  const float max = *std::max_element(src, src + size);

  for (size_t i = 0; i < size; i++) {
    dst[i] = std::exp(src[i] - max);
    sum += dst[i];
  }

  for (size_t i = 0; i < size; i++) {
    dst[i] /= sum;
  }
}

void Dequantize(const int8_t *q_data, float *data, float threshold, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    data[i] = float(q_data[i]) * threshold / 128.0;
  }
}

void clip_boxes(int width, int height, cvai_bbox_t &box) {
  if (box.x1 < 0) {
    box.x1 = 0;
  }
  if (box.y1 < 0) {
    box.y1 = 0;
  }
  if (box.x2 > width - 1) {
    box.x2 = width - 1;
  }
  if (box.y2 > height - 1) {
    box.y2 = height - 1;
  }
}

cvai_bbox_t box_rescale_c(const float frame_width, const float frame_height, const float nn_width,
                          const float nn_height, const cvai_bbox_t bbox, float *ratio,
                          float *pad_width, float *pad_height) {
  float ratio_height = (nn_height / frame_height);
  float ratio_width = (nn_width / frame_width);
  if (ratio_height > ratio_width) {
    *ratio = 1.0 / ratio_width;
    *pad_width = 0;
    *pad_height = (nn_height - frame_height * ratio_width) / 2;
  } else {
    *ratio = 1.0 / ratio_height;
    *pad_width = (nn_width - frame_width * ratio_height) / 2;
    *pad_height = 0;
  }

  float x1 = (bbox.x1 - (*pad_width)) * (*ratio);
  float x2 = (bbox.x2 - (*pad_width)) * (*ratio);
  float y1 = (bbox.y1 - (*pad_height)) * (*ratio);
  float y2 = (bbox.y2 - (*pad_height)) * (*ratio);
  cvai_bbox_t new_bbox;
  new_bbox.score = bbox.score;
  new_bbox.x1 = std::max(std::min(x1, (float)(frame_width - 1)), (float)0);
  new_bbox.x2 = std::max(std::min(x2, (float)(frame_width - 1)), (float)0);
  new_bbox.y1 = std::max(std::min(y1, (float)(frame_height - 1)), (float)0);
  new_bbox.y2 = std::max(std::min(y2, (float)(frame_height - 1)), (float)0);
  return new_bbox;
}

cvai_bbox_t box_rescale_rb(const float frame_width, const float frame_height, const float nn_width,
                           const float nn_height, const cvai_bbox_t bbox, float *ratio) {
  float ratio_height = (nn_height / frame_height);
  float ratio_width = (nn_width / frame_width);
  *ratio = 1.0 / std::min(ratio_height, ratio_width);

  float x1 = bbox.x1 * (*ratio);
  float x2 = bbox.x2 * (*ratio);
  float y1 = bbox.y1 * (*ratio);
  float y2 = bbox.y2 * (*ratio);

  cvai_bbox_t new_bbox;
  new_bbox.score = bbox.score;
  new_bbox.x1 = std::max(std::min(x1, (float)(frame_width - 1)), (float)0);
  new_bbox.x2 = std::max(std::min(x2, (float)(frame_width - 1)), (float)0);
  new_bbox.y1 = std::max(std::min(y1, (float)(frame_height - 1)), (float)0);
  new_bbox.y2 = std::max(std::min(y2, (float)(frame_height - 1)), (float)0);
  return new_bbox;
}

cvai_bbox_t box_rescale(const float frame_width, const float frame_height, const float nn_width,
                        const float nn_height, const cvai_bbox_t bbox,
                        const BOX_RESCALE_TYPE type) {
  float ratio;
  switch (type) {
    case BOX_RESCALE_TYPE::CENTER: {
      float pad_width, pad_height;
      return box_rescale_c(frame_width, frame_height, nn_width, nn_height, bbox, &ratio, &pad_width,
                           &pad_height);
    } break;
    case BOX_RESCALE_TYPE::RB: {
      return box_rescale_rb(frame_width, frame_height, nn_width, nn_height, bbox, &ratio);
    } break;
  }
  cvai_bbox_t box;
  memset(&box, 0, sizeof(cvai_bbox_t));
  return box;
}

void NeonQuantizeScale(VIDEO_FRAME_INFO_S *inFrame, const float *qFactor, const float *qMean,
                       VIDEO_FRAME_INFO_S *outFrame) {
  if (inFrame->stVFrame.enPixelFormat != PIXEL_FORMAT_RGB_888 ||
      outFrame->stVFrame.enPixelFormat != PIXEL_FORMAT_RGB_888_PLANAR) {
    LOGE("Input must be PIXEL_FORMAT_RGB_888. Output must be PIXEL_FORMAT_RGB_888_PLANAR.\n");
    return;
  }
  bool do_unmap_in = false, do_unmap_out = false;
  if (inFrame->stVFrame.pu8VirAddr[0] == NULL) {
    inFrame->stVFrame.pu8VirAddr[0] = (CVI_U8 *)CVI_SYS_MmapCache(inFrame->stVFrame.u64PhyAddr[0],
                                                                  inFrame->stVFrame.u32Length[0]);
    do_unmap_in = true;
  }
  if (outFrame->stVFrame.pu8VirAddr[0] == NULL) {
    outFrame->stVFrame.pu8VirAddr[0] = (CVI_U8 *)CVI_SYS_MmapCache(outFrame->stVFrame.u64PhyAddr[0],
                                                                   outFrame->stVFrame.u32Length[0]);
    do_unmap_out = true;
  }
  cv::Mat image(cv::Size(inFrame->stVFrame.u32Width, inFrame->stVFrame.u32Height), CV_8UC3,
                inFrame->stVFrame.pu8VirAddr[0], inFrame->stVFrame.u32Stride[0]);
  std::vector<cv::Mat> channels;
  for (uint32_t i = 0; i < 3; i++) {
    channels.push_back(cv::Mat(cv::Size(outFrame->stVFrame.u32Width, outFrame->stVFrame.u32Height),
                               CV_8UC1, outFrame->stVFrame.pu8VirAddr[0],
                               outFrame->stVFrame.u32Stride[0]));
  }
  const std::vector<float> mean = {qFactor[0], qFactor[1], qFactor[2]};
  const std::vector<float> factor = {qMean[0], qMean[1], qMean[2]};
  NormalizeAndSplitToU8(image, mean, factor, channels);
  CVI_SYS_IonFlushCache(outFrame->stVFrame.u64PhyAddr[0], outFrame->stVFrame.pu8VirAddr[0],
                        outFrame->stVFrame.u32Length[0]);
  if (do_unmap_in) {
    CVI_SYS_Munmap((void *)inFrame->stVFrame.pu8VirAddr[0], inFrame->stVFrame.u32Length[0]);
    inFrame->stVFrame.pu8VirAddr[0] = NULL;
  }
  if (do_unmap_out) {
    CVI_SYS_Munmap((void *)outFrame->stVFrame.pu8VirAddr[0], outFrame->stVFrame.u32Length[0]);
    outFrame->stVFrame.pu8VirAddr[0] = NULL;
  }
}

}  // namespace cviai
