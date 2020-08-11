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

}  // namespace cviai