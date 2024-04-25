#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <map>
#include <memory>
#include <numeric>
#include <string>

#include <core/core/cvtdl_errno.h>
#include <error_msg.hpp>
#include "coco_utils.hpp"
#include "core/core/cvtdl_core_types.h"
#include "core/cvi_tdl_types_mem_internal.h"
#include "core/object/cvtdl_object_types.h"
#include "core/utils/vpss_helper.h"
#include "core_utils.hpp"
#include "cvi_comm.h"
#include "lane_detection.hpp"
#include "misc.hpp"
#include "object_utils.hpp"

float H_GAP = 2.0f / 9;
int MAX_LANE = 5;
int NUM_POINTS = 56;

static const float STD_R = (255.0 * 0.229);
static const float STD_G = (255.0 * 0.224);
static const float STD_B = (255.0 * 0.225);
static const float MODEL_MEAN_R = 0.485 * 255.0;
static const float MODEL_MEAN_G = 0.456 * 255.0;
static const float MODEL_MEAN_B = 0.406 * 255.0;

#define FACTOR_R (1.0 / STD_R)
#define FACTOR_G (1.0 / STD_G)
#define FACTOR_B (1.0 / STD_B)
#define MEAN_R (MODEL_MEAN_R / STD_R)
#define MEAN_G (MODEL_MEAN_G / STD_G)
#define MEAN_B (MODEL_MEAN_B / STD_B)

using namespace std;

namespace cvitdl {

float ld_sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }

float sample_comb(int x) {  // C(3, x)

  if (x == 0 || x == 3) {
    return 1;
  }
  return 3;
}

BezierLaneNet::BezierLaneNet() : Core(CVI_MEM_DEVICE) {
  for (int i = 0; i < NUM_POINTS; i++) {
    // float t = H_GAP + i*(1 - H_GAP)/NUM_POINTS;

    float t = (float)i / NUM_POINTS;

    for (int k = 0; k < 4; k++) {
      c_matrix[i][k] = pow(t, k) * pow(1 - t, 3 - k) * sample_comb(k);
    }
  }
}

BezierLaneNet::~BezierLaneNet() {}

int BezierLaneNet::setupInputPreprocess(std::vector<InputPreprecessSetup> *data) {
  if (data->size() != 1) {
    LOGE("BezierLaneNet only has 1 input.\n");
    return CVI_TDL_ERR_INVALID_ARGS;
  }
  (*data)[0].factor[0] = static_cast<float>(FACTOR_R);
  (*data)[0].factor[1] = static_cast<float>(FACTOR_G);
  (*data)[0].factor[2] = static_cast<float>(FACTOR_B);
  (*data)[0].mean[0] = static_cast<float>(MEAN_R);
  (*data)[0].mean[1] = static_cast<float>(MEAN_G);
  (*data)[0].mean[2] = static_cast<float>(MEAN_B);
  (*data)[0].use_quantize_scale = true;
  (*data)[0].rescale_type = RESCALE_RB;
  return CVI_TDL_SUCCESS;
}

int BezierLaneNet::inference(VIDEO_FRAME_INFO_S *srcFrame, cvtdl_lane_t *lane_meta) {
  std::vector<VIDEO_FRAME_INFO_S *> frames = {srcFrame};
  int ret = run(frames);
  if (ret != CVI_TDL_SUCCESS) {
    LOGE("BezierLaneNet run inference failed!\n");
    return ret;
  }

  CVI_SHAPE shape = getInputShape(0);
  outputParser(shape.dim[3], shape.dim[2], srcFrame->stVFrame.u32Width,
               srcFrame->stVFrame.u32Height, lane_meta);
  model_timer_.TicToc("post");
  return CVI_TDL_SUCCESS;
}

void BezierLaneNet::outputParser(const int nn_width, const int nn_height, const int frame_width,
                                 const int frame_height, cvtdl_lane_t *lane_meta) {
  float *curves = getOutputRawPtr<float>(0);
  float *logits = getOutputRawPtr<float>(1);

  CVI_SHAPE output0_shape = getOutputShape(0);
  CVI_SHAPE output1_shape = getOutputShape(1);

  // for (int i = 0; i< 4; i++){
  //   printf("output0_shape.dim[%d]: %d\n", i, output0_shape.dim[i]);
  //   printf("output1_shape.dim[%d]: %d\n", i, output1_shape.dim[i]);

  //   printf("curves[i]:%f, logits[i]:%f \n", curves[i], logits[i]);
  // }

  map<float, int> valid_scores;

  int counter = 0;
  for (int i = -4; i < -4 + output1_shape.dim[1];
       i++) {  // max_pool1d, window_size=9, stride=1, padding=4

    float *score_start = logits + std::max(i, 0);
    float *score_end = logits + std::min(i + 8, output1_shape.dim[1] - 1);

    auto iter = std::max_element(score_start, score_end);
    int pos = iter - logits;

    if (pos == counter) {
      float score = ld_sigmoid(logits[counter]);
      if (score > 0.4) {
        valid_scores.insert(make_pair(score, counter));
      }
    }

    counter++;
  }

  printf("valid_scores.size(): %d\n", valid_scores.size());

  //

  CVI_TDL_MemAllocInit(std::min((int)valid_scores.size(), MAX_LANE), lane_meta);

  map<float, int>::reverse_iterator map_iter;
  counter = 0;

  for (map_iter = valid_scores.rbegin(); map_iter != valid_scores.rend(); map_iter++) {
    if (counter == MAX_LANE) break;

    int valid_index = map_iter->second;

    int start_index = valid_index * output0_shape.dim[2] * output0_shape.dim[3];

    // printf("####################################################\n");

    for (int i = 0; i < NUM_POINTS; i++) {
      float x = c_matrix[i][0] * curves[start_index] + c_matrix[i][1] * curves[start_index + 2] +
                c_matrix[i][2] * curves[start_index + 4] + c_matrix[i][3] * curves[start_index + 6];

      float y = c_matrix[i][0] * curves[start_index + 1] +
                c_matrix[i][1] * curves[start_index + 3] +
                c_matrix[i][2] * curves[start_index + 5] + c_matrix[i][3] * curves[start_index + 7];

      lane_meta->lane[counter].x[i] = x * frame_width;
      lane_meta->lane[counter].y[i] = y * frame_height;

      // printf("x: %f, y:%f\n", x, y);
    }

    counter++;
  }
}

}  // namespace cvitdl