#include "core_utils.hpp"

#include <math.h>
#include <algorithm>

namespace cviai {
void SoftMaxForBuffer(float *src, float *dst, size_t size) {
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
                          const float nn_height, const cvai_bbox_t bbox) {
  float x1, x2, y1, y2;
  if (frame_width >= frame_height) {
    float ratio_x = frame_width / nn_width;
    float bbox_y_height = nn_height * frame_height / frame_width;
    float ratio_y = frame_height / bbox_y_height;
    float bbox_padding_top = (nn_height - bbox_y_height) / 2;
    x1 = bbox.x1 * ratio_x;
    x2 = bbox.x2 * ratio_x;
    y1 = (bbox.y1 - bbox_padding_top) * ratio_y;
    y2 = (bbox.y2 - bbox_padding_top) * ratio_y;
  } else {
    float ratio_y = frame_height / nn_height;
    float bbox_x_height = nn_width * frame_width / frame_height;
    float ratio_x = frame_width / bbox_x_height;
    float bbox_padding_left = (nn_width - bbox_x_height) / 2;
    x1 = (bbox.x1 - bbox_padding_left) * ratio_x;
    x2 = (bbox.x2 - bbox_padding_left) * ratio_x;
    y1 = bbox.y1 * ratio_y;
    y2 = bbox.y2 * ratio_y;
  }
  cvai_bbox_t new_bbox;
  new_bbox.score = bbox.score;
  new_bbox.x1 = std::max(std::min(x1, (float)(frame_width - 1)), (float)0);
  new_bbox.x2 = std::max(std::min(x2, (float)(frame_width - 1)), (float)0);
  new_bbox.y1 = std::max(std::min(y1, (float)(frame_height - 1)), (float)0);
  new_bbox.y2 = std::max(std::min(y2, (float)(frame_height - 1)), (float)0);
  return new_bbox;
}

cvai_bbox_t box_rescale_small_ratio_major(const float frame_width, const float frame_height,
                                          const float nn_width, const float nn_height,
                                          const cvai_bbox_t bbox) {
  float x1, x2, y1, y2;
  float ratio_height = (nn_height / frame_height);
  float ratio_width = (nn_width / frame_width);
  float ratio = 1.0 / std::min(ratio_height, ratio_width);

  x1 = bbox.x1 * ratio;
  x2 = bbox.x2 * ratio;
  y1 = bbox.y1 * ratio;
  y2 = bbox.y2 * ratio;

  cvai_bbox_t new_bbox;
  new_bbox.score = bbox.score;
  new_bbox.x1 = std::max(std::min(x1, (float)(frame_width - 1)), (float)0);
  new_bbox.x2 = std::max(std::min(x2, (float)(frame_width - 1)), (float)0);
  new_bbox.y1 = std::max(std::min(y1, (float)(frame_height - 1)), (float)0);
  new_bbox.y2 = std::max(std::min(y2, (float)(frame_height - 1)), (float)0);
  return new_bbox;
}

cvai_bbox_t box_rescale_rb(const float frame_width, const float frame_height, const float nn_width,
                           const float nn_height, const cvai_bbox_t bbox) {
  float x1, x2, y1, y2;
  if (frame_width >= frame_height) {
    float ratio_x = frame_width / nn_width;
    float bbox_y_height = nn_height * frame_height / frame_width;
    float ratio_y = frame_height / bbox_y_height;
    x1 = bbox.x1 * ratio_x;
    x2 = bbox.x2 * ratio_x;
    y1 = bbox.y1 * ratio_y;
    y2 = bbox.y2 * ratio_y;
  } else {
    float ratio_y = frame_height / nn_height;
    float bbox_x_height = nn_width * frame_width / frame_height;
    float ratio_x = frame_width / bbox_x_height;
    x1 = bbox.x1 * ratio_x;
    x2 = bbox.x2 * ratio_x;
    y1 = bbox.y1 * ratio_y;
    y2 = bbox.y2 * ratio_y;
  }
  cvai_bbox_t new_bbox;
  new_bbox.score = bbox.score;
  new_bbox.x1 = std::max(std::min(x1, (float)(frame_width - 1)), (float)0);
  new_bbox.x2 = std::max(std::min(x2, (float)(frame_width - 1)), (float)0);
  new_bbox.y1 = std::max(std::min(y1, (float)(frame_height - 1)), (float)0);
  new_bbox.y2 = std::max(std::min(y2, (float)(frame_height - 1)), (float)0);
  return new_bbox;
}

}  // namespace cviai
