#pragma once
#include "face/cvai_face_types.h"

#include "anchor_generator.h"
#include "core_utils.hpp"
#include "face_utils.hpp"

#include "opencv2/opencv.hpp"

#include <vector>

namespace cviai {

static void bbox_pred(const anchor_box &anchor, cv::Vec4f regress, float ratio, cvai_bbox_t &bbox) {
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

static void landmark_pred(const anchor_box &anchor, float ratio, cvai_pts_t &facePt) {
  float width = anchor.x2 - anchor.x1 + 1;
  float height = anchor.y2 - anchor.y1 + 1;
  float ctr_x = anchor.x1 + 0.5 * (width - 1.0);
  float ctr_y = anchor.y1 + 0.5 * (height - 1.0);

  for (size_t j = 0; j < facePt.size; j++) {
    facePt.x[j] = (facePt.x[j] * width + ctr_x) * ratio;
    facePt.y[j] = (facePt.y[j] * height + ctr_y) * ratio;
  }
}

static int softmax_by_channel(float *input, float *output, const std::vector<int64_t> &shape) {
  int *iter = new int[shape[1]];
  float *ex = new float[shape[1]];

  for (int N = 0; N < shape[0]; ++N) {
    for (int H = 0; H < shape[2]; ++H) {
      for (int W = 0; W < shape[3]; ++W) {
        float max_val = std::numeric_limits<float>::lowest();
        for (int C = 0; C < shape[1]; ++C) {
          iter[C] =
              (N * shape[1] * shape[2] * shape[3]) + (C * shape[2] * shape[3]) + (H * shape[3]) + W;
        }

        for (int C = 0; C < shape[1]; ++C) {
          max_val = std::max(input[iter[C]], max_val);
        }

        // find softmax divisor
        float sum_of_ex = 0.0f;
        for (int C = 0; C < shape[1]; ++C) {
          float x = input[iter[C]] - max_val;
          ex[C] = FastExp(x);
          sum_of_ex += ex[C];
        }

        // calculate softmax
        for (int C = 0; C < shape[1]; ++C) {
          output[iter[C]] = ex[C] / sum_of_ex;
        }
      }
    }
  }

  delete[] iter;
  delete[] ex;

  return 0;
}

}  // namespace cviai