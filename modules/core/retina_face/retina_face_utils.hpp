#pragma once
#include "anchor_generator.h"

#include "core/core/cvai_core_types.h"
#include "core_utils.hpp"

#include "opencv2/opencv.hpp"

#include <vector>

namespace cviai {

inline void __attribute__((always_inline))
bbox_pred(const anchor_box &anchor, cv::Vec4f regress, float ratio, cvai_bbox_t &bbox) {
  float width = anchor.x2 - anchor.x1 + 1;
  float height = anchor.y2 - anchor.y1 + 1;
  float ctr_x = anchor.x1 + 0.5 * (width - 1.0);
  float ctr_y = anchor.y1 + 0.5 * (height - 1.0);

  float pred_ctr_x = regress[0] * width + ctr_x;
  float pred_ctr_y = regress[1] * height + ctr_y;
  float pred_w = FastExp(regress[2]) * width;
  float pred_h = FastExp(regress[3]) * height;

  bbox.x1 = (pred_ctr_x - 0.5 * (pred_w - 1.0)) * ratio;
  bbox.y1 = (pred_ctr_y - 0.5 * (pred_h - 1.0)) * ratio;
  bbox.x2 = (pred_ctr_x + 0.5 * (pred_w - 1.0)) * ratio;
  bbox.y2 = (pred_ctr_y + 0.5 * (pred_h - 1.0)) * ratio;
}

inline void __attribute__((always_inline))
landmark_pred(const anchor_box &anchor, float ratio, cvai_pts_t &facePt) {
  float width = anchor.x2 - anchor.x1 + 1;
  float height = anchor.y2 - anchor.y1 + 1;
  float ctr_x = anchor.x1 + 0.5 * (width - 1.0);
  float ctr_y = anchor.y1 + 0.5 * (height - 1.0);

  for (size_t j = 0; j < facePt.size; j++) {
    facePt.x[j] = (facePt.x[j] * width + ctr_x) * ratio;
    facePt.y[j] = (facePt.y[j] * height + ctr_y) * ratio;
  }
}

}  // namespace cviai
