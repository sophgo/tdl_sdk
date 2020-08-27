#pragma once
#include "ive/ive.h"

class TamperDetectorMD {
 public:
  TamperDetectorMD() = delete;
  // TamperDetectorMD(VIDEO_FRAME_INFO_S *init_frame, float momentum=0.05, int update_interval=10);
  TamperDetectorMD(VIDEO_FRAME_INFO_S *init_frame, float momentum, int update_interval);
  int update(VIDEO_FRAME_INFO_S *frame);
  int detect(VIDEO_FRAME_INFO_S *frame);
  int detect(VIDEO_FRAME_INFO_S *frame, float *moving_score);
  IVE_IMAGE_S &getMean();
  IVE_IMAGE_S &getDiff();
  void printMean();
  void printDiff();
  void free();
  void print_info();

 private:
  IVE_HANDLE ive_handle;
  int nChannels;
  CVI_U16 strideWidth, height, width, area;
  IVE_IMAGE_S mean;
  IVE_IMAGE_S diff;
  float momentum;
  int update_interval;
  int counter;
  // IVE_IMAGE_S MIN_DIFF_M;

  // float alertMovingScore;
  // float alertContourScore;
};