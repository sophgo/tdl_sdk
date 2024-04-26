#include "polylanenet.hpp"
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>
#include "core/core/cvtdl_core_types.h"
#include "core/core/cvtdl_errno.h"
#include "core/cvi_tdl_types_mem.h"
#include "core/cvi_tdl_types_mem_internal.h"
#include "core/object/cvtdl_object_types.h"
#include "core/utils/vpss_helper.h"
#include "core_utils.hpp"
#include "cvi_comm.h"
#include "cvi_sys.h"
#include "error_msg.hpp"
#include "misc.hpp"
#include "object_utils.hpp"

namespace cvitdl {

Polylanenet::Polylanenet() : Core(CVI_MEM_DEVICE) {}
Polylanenet::~Polylanenet() {}

float Polylanenet::ld_sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }

void dump_frame(VIDEO_FRAME_INFO_S *frame, FILE *fp) {
  if (frame->stVFrame.pu8VirAddr[0] == NULL) {
    size_t image_size =
        frame->stVFrame.u32Length[0] + frame->stVFrame.u32Length[1] + frame->stVFrame.u32Length[2];
    frame->stVFrame.pu8VirAddr[0] =
        (CVI_U8 *)CVI_SYS_MmapCache(frame->stVFrame.u64PhyAddr[0], image_size);
    frame->stVFrame.pu8VirAddr[1] = frame->stVFrame.pu8VirAddr[0] + frame->stVFrame.u32Length[0];
    frame->stVFrame.pu8VirAddr[2] = frame->stVFrame.pu8VirAddr[1] + frame->stVFrame.u32Length[1];
  }

  printf("===== %d %d %d\n", frame->stVFrame.u32Length[0], frame->stVFrame.u32Length[1],
         frame->stVFrame.u32Length[2]);
  for (int i = 0; i < 3; i++) {
    uint8_t *paddr = (uint8_t *)frame->stVFrame.pu8VirAddr[i];
    printf("towrite channel len: %d, %d\n", frame->stVFrame.u32Length[i], i);
    fwrite(paddr, frame->stVFrame.u32Length[i], 1, fp);
  }
}

int Polylanenet::setupInputPreprocess(std::vector<InputPreprecessSetup> *data) {
  if (data->size() != 1) {
    LOGE("Polylanenet only has 1 input.\n");
    return CVI_TDL_ERR_INVALID_ARGS;
  }
  (*data)[0].factor[0] = 0.01712475;
  (*data)[0].factor[1] = 0.0175070;
  (*data)[0].factor[2] = 0.0174291;

  (*data)[0].mean[0] = 2.117903;
  (*data)[0].mean[1] = 2.035714;
  (*data)[0].mean[2] = 1.804444;
  (*data)[0].format = PIXEL_FORMAT_RGB_888_PLANAR;
  (*data)[0].use_quantize_scale = true;
  (*data)[0].use_quantize_scale = true;
  (*data)[0].rescale_type = RESCALE_RB;

  std::cout << "setupInputPreprocess done" << std::endl;
  return CVI_TDL_SUCCESS;
}

int Polylanenet::inference(VIDEO_FRAME_INFO_S *frame, cvtdl_lane_t *lane_meta) {
  lane_meta->height = frame->stVFrame.u32Height;
  lane_meta->width = frame->stVFrame.u32Width;
  std::vector<VIDEO_FRAME_INFO_S *> frames = {frame};
  int ret = run(frames);
  if (ret != CVI_TDL_SUCCESS) {
    return ret;
  }
  // FILE *output;
  // output = fopen("/mnt/data/zwp/polylane_package/dataset/frame.bin", "wb");
  // dump_frame(frame,output);
  // fclose(output);
  outputParser(lane_meta);
  model_timer_.TicToc("post");
  return CVI_TDL_SUCCESS;
}

void Polylanenet::set_lower(float th) {
  if (th > 0 && th < 1) {
    LOWERE = th;
  }
}

int Polylanenet::outputParser(cvtdl_lane_t *lane_meta) {
  float *out_feature = getOutputRawPtr<float>(0);
  CVI_SHAPE output_shape = getOutputShape(0);
  std::vector<std::vector<float>> point_map;
  //(5 lanes) * (1 conf + 2 (upper & lower) + 4 poly coeffs)
  for (int i = 0; i < 5; i++) {
    float score = ld_sigmoid(out_feature[i * 7]);
    std::cout << "number:" << i << ";score:" << score << std::endl;
    if (score > 0.5 and out_feature[i * 7 + 2] > LOWERE) {
      std::vector<float> line_info(out_feature + i * 7 + 1, out_feature + i * 7 + 7);
      point_map.push_back(line_info);
    }
  }
  lane_meta->size = point_map.size();
  if (lane_meta->size > 0) {
    lane_meta->lane = (cvtdl_lane_point_t *)malloc(sizeof(cvtdl_lane_point_t) * lane_meta->size);
  } else {
    return CVI_TDL_SUCCESS;
  }
  std::cout << "LOWERE:" << LOWERE << std::endl;
  for (int i = 0; i < lane_meta->size; i++) {
    float lower = std::max(LOWERE, point_map[i][0]);
    float upper = std::min(1.0f, point_map[i][1]);
    float slice = (upper - lower) / NUM_POINTS;
    std::cout << "lower:" << lower << ";upper:" << upper << ";slice:" << slice << std::endl;
    for (int j = 0; j < NUM_POINTS; j++) {
      lane_meta->lane[i].y[j] = static_cast<float>((lower + j * slice) * lane_meta->height);
      float polyValue = 0;
      for (int p = 0; p < 4; ++p) {
        polyValue += point_map[i][p + 2] * std::pow((lower + j * slice), 3 - p);
      }
      lane_meta->lane[i].x[j] = static_cast<float>(polyValue * lane_meta->width);
    }
  }

  return CVI_TDL_SUCCESS;
}

}  // namespace cvitdl
