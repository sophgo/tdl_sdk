#include <stdio.h>
#include <syslog.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "ive/ive.h"

#include "tamper_detection.hpp"

#define INIT_DIFF 12
#define MIN_DIFF 8

#define DEBUG (0)
#define DEBUG_PRINT_CHANNELS_NUM 1
#define STATIC_CAST_UINT(d) static_cast<uint>(d)

#if DEGUB
void print_IVE_IMAGE_S(IVE_IMAGE_S &ive_image, int c = 3) {
  int nChannels = c;
  int strideWidth = ive_image.u16Stride[0];
  int height = ive_image.u16Height;
  int width = ive_image.u16Width;
  if (width > 64) {
    std::cout << "width " << width << " is too long" << std::endl;
    return;
  }

  for (int cc = 0; cc < nChannels; cc++) {
    for (int ii = 0; ii < height; ii++) {
      for (int jj = 0; jj < width; jj++) {
        std::cout << std::setw(5)
                  << STATIC_CAST_UINT(ive_image.pu8VirAddr[cc][ii * strideWidth + jj]) << ", ";
      }
      std::cout << std::endl;
    }
    std::cout << "-------------------------------------" << std::endl;
  }
}
#endif

// TamperDetectorMD::TamperDetectorMD(){
// }

TamperDetectorMD::TamperDetectorMD(IVE_HANDLE handle, VIDEO_FRAME_INFO_S *init_frame,
                                   float momentum, int update_interval) {
  CVI_S32 ret = CVI_SUCCESS;
  this->ive_handle = handle;
  this->nChannels = 3;

  IVE_IMAGE_S new_frame;
  ret = CVI_IVE_VideoFrameInfo2Image(init_frame, &new_frame);
  if (ret != CVI_SUCCESS) {
    syslog(LOG_ERR, "Convert to video frame failed with %#x!\n", ret);
  }
  this->strideWidth = new_frame.u16Stride[0];
  this->height = new_frame.u16Height;
  this->width = new_frame.u16Width;
  CVI_U32 imgSize = strideWidth * height;

  CVI_IVE_CreateImage(this->ive_handle, &this->mean, IVE_IMAGE_TYPE_U8C3_PLANAR, width, height);
  memcpy(this->mean.pu8VirAddr[0], new_frame.pu8VirAddr[0], imgSize);
  memcpy(this->mean.pu8VirAddr[1], new_frame.pu8VirAddr[1], imgSize);
  memcpy(this->mean.pu8VirAddr[2], new_frame.pu8VirAddr[2], imgSize);

  // // Map the image.
  // CVI_U32 imageLength = init_frame->stVFrame.u32Length[0] + init_frame->stVFrame.u32Length[1] +
  //                       init_frame->stVFrame.u32Length[2];
  // init_frame->stVFrame.pu8VirAddr[0] = (CVI_U8 *)CVI_SYS_Mmap(init_frame->stVFrame.u64PhyAddr[0],
  // imageLength); Convert to IVE image. Note this function does not map or unmap for you. ret =
  // CVI_IVE_VideoFrameInfo2Image(init_frame, &this->mean); if (ret != CVI_SUCCESS) {
  //     printf("Convert to video frame failed with %#x!\n", ret);
  // }

  // Flush to DRAM before IVE function.
  CVI_IVE_BufFlush(this->ive_handle, &this->mean);

  CVI_IVE_CreateImage(this->ive_handle, &this->diff, IVE_IMAGE_TYPE_U8C3_PLANAR, width, height);
  memset(this->diff.pu8VirAddr[0], INIT_DIFF, imgSize);
  memset(this->diff.pu8VirAddr[1], INIT_DIFF, imgSize);
  memset(this->diff.pu8VirAddr[2], INIT_DIFF, imgSize);
  // Flush to DRAM before IVE function.
  CVI_IVE_BufFlush(this->ive_handle, &this->diff);

  // this->alertMovingScore = alertMovingScore;
  this->momentum = momentum;
  this->update_interval = update_interval;
  this->counter = 1;

  CVI_SYS_FreeI(this->ive_handle, &new_frame);

#if DEBUG
  std::cout << "==== this->mean ====" << std::endl;
  print_IVE_IMAGE_S(this->mean, DEBUG_PRINT_CHANNELS_NUM);
  std::cout << "==== this->diff ====" << std::endl;
  print_IVE_IMAGE_S(this->diff, DEBUG_PRINT_CHANNELS_NUM);

  // this->update_interval = 1;
#endif
}

int TamperDetectorMD::update(VIDEO_FRAME_INFO_S *frame) {
  // printf("[update] start\n");
  CVI_S32 ret = CVI_SUCCESS;
  IVE_IMAGE_S new_frame;
  ret = CVI_IVE_VideoFrameInfo2Image(frame, &new_frame);
  if (ret != CVI_SUCCESS) {
    syslog(LOG_ERR, "Convert to video frame failed with %#x!\n", ret);
    return ret;
  }
#if DEBUG
  std::cout << "==== [update] input frame ====" << std::endl;
  CVI_IVE_BufFlush(this->ive_handle, &new_frame);
  print_IVE_IMAGE_S(new_frame, DEBUG_PRINT_CHANNELS_NUM);
#endif
#if DEBUG
  std::cout << "==== this->mean ====" << std::endl;
  CVI_IVE_BufFlush(this->ive_handle, &this->mean);
  print_IVE_IMAGE_S(this->mean, DEBUG_PRINT_CHANNELS_NUM);
#endif

  IVE_DST_IMAGE_S frame_diff_1, frame_diff_2;
  IVE_DST_IMAGE_S min_diff;
  IVE_DST_IMAGE_S u8c1_image;
  CVI_IVE_CreateImage(this->ive_handle, &frame_diff_1, IVE_IMAGE_TYPE_U8C3_PLANAR, width, height);
  CVI_IVE_CreateImage(this->ive_handle, &frame_diff_2, IVE_IMAGE_TYPE_U8C3_PLANAR, width, height);
  CVI_IVE_CreateImage(this->ive_handle, &min_diff, IVE_IMAGE_TYPE_U8C3_PLANAR, width, height);
  CVI_U32 imgSize = strideWidth * height;
  memset(min_diff.pu8VirAddr[0], MIN_DIFF, imgSize);
  memset(min_diff.pu8VirAddr[1], MIN_DIFF, imgSize);
  memset(min_diff.pu8VirAddr[2], MIN_DIFF, imgSize);
  // Flush to DRAM before IVE function.
  CVI_IVE_BufFlush(this->ive_handle, &min_diff);

  u8c1_image.enType = IVE_IMAGE_TYPE_U8C1;

  // Setup control parameter, Sub.
  IVE_SUB_CTRL_S iveSubCtrl;
  iveSubCtrl.enMode = IVE_SUB_MODE_ABS;  // ABS_DIFF
  // Run IVE sub.
  ret |= CVI_IVE_Sub(this->ive_handle, &new_frame, &this->mean, &frame_diff_1, &iveSubCtrl, false);
  if (ret != CVI_SUCCESS) {
    std::cout << "error: sub" << std::endl;
    return ret;
  }
#if DEBUG
  std::cout << "==== [update] frame diff (1) ====" << std::endl;
  CVI_IVE_BufFlush(this->ive_handle, &frame_diff_1);
  print_IVE_IMAGE_S(frame_diff_1, DEBUG_PRINT_CHANNELS_NUM);
#endif

  // Setup control parameter, Add.
  IVE_ADD_CTRL_S iveAddCtrl;
  iveAddCtrl.aX = 1.0;
  iveAddCtrl.bY = 1.0;
  ret |= CVI_IVE_Add(this->ive_handle, &frame_diff_1, &frame_diff_1, &frame_diff_1, &iveAddCtrl,
                     false);
#if DEBUG
  std::cout << "==== [update] frame diff (2) ====" << std::endl;
  CVI_IVE_BufFlush(this->ive_handle, &frame_diff_1);
  print_IVE_IMAGE_S(frame_diff_1, DEBUG_PRINT_CHANNELS_NUM);
#endif

  iveAddCtrl.aX = 1.0 - this->momentum;
  iveAddCtrl.bY = this->momentum;
#if DEBUG
  std::cout << "==== momentum ====" << std::endl
            << "aX = " << iveAddCtrl.aX << std::endl
            << "bY = " << iveAddCtrl.bY << std::endl;
  std::cout << "==== [update] (Y) frame ====" << std::endl;
  print_IVE_IMAGE_S(new_frame, DEBUG_PRINT_CHANNELS_NUM);
  std::cout << "==== [update] (X) old mean ====" << std::endl;
  print_IVE_IMAGE_S(this->mean, DEBUG_PRINT_CHANNELS_NUM);
#endif
  // Update mean
  ret |= CVI_IVE_Add(this->ive_handle, &this->mean, &new_frame, &this->mean, &iveAddCtrl, false);
  if (ret != CVI_SUCCESS) {
    std::cout << "error: add" << std::endl;
    return ret;
  }
#if DEBUG
  std::cout << "==== [update] new mean ====" << std::endl;
  CVI_IVE_BufRequest(this->ive_handle, &this->mean);
  print_IVE_IMAGE_S(this->mean, DEBUG_PRINT_CHANNELS_NUM);

  std::cout << "==== momentum ====" << std::endl
            << "aX = " << iveAddCtrl.aX << std::endl
            << "bY = " << iveAddCtrl.bY << std::endl;
  std::cout << "==== [update] (Y) diff ====" << std::endl;
  print_IVE_IMAGE_S(frame_diff_1, DEBUG_PRINT_CHANNELS_NUM);
  std::cout << "==== [update] (X) old diff ====" << std::endl;
  print_IVE_IMAGE_S(this->diff, DEBUG_PRINT_CHANNELS_NUM);
#endif
  // Update diff
  ret |= CVI_IVE_Add(this->ive_handle, &this->diff, &frame_diff_1, &this->diff, &iveAddCtrl, false);
  if (ret != CVI_SUCCESS) {
    std::cout << "error: add" << std::endl;
    return ret;
  }
#if DEBUG
  std::cout << "==== [update] new diff ====" << std::endl;
  CVI_IVE_BufRequest(this->ive_handle, &this->diff);
  print_IVE_IMAGE_S(this->diff, DEBUG_PRINT_CHANNELS_NUM);
#endif

  iveSubCtrl.enMode = IVE_SUB_MODE_NORMAL;
  ret |= CVI_IVE_Sub(this->ive_handle, &this->diff, &min_diff, &frame_diff_1, &iveSubCtrl, false);
#if DEBUG
  std::cout << "==== [update] min diff ====" << std::endl;
  CVI_IVE_BufFlush(this->ive_handle, &min_diff);
  print_IVE_IMAGE_S(min_diff, DEBUG_PRINT_CHANNELS_NUM);
  std::cout << "==== [update] diff sub (move) ====" << std::endl;
  CVI_IVE_BufFlush(this->ive_handle, &frame_diff_1);
  print_IVE_IMAGE_S(frame_diff_1, DEBUG_PRINT_CHANNELS_NUM);
#endif
  u8c1_image.tpu_block = frame_diff_1.tpu_block;
  IVE_THRESH_CTRL_S iveThreshCtrl;
  iveThreshCtrl.enMode = IVE_THRESH_MODE_BINARY;
  iveThreshCtrl.u8LowThr = 0;
  iveThreshCtrl.u8MinVal = 0;
  iveThreshCtrl.u8MaxVal = 255;
  ret |= CVI_IVE_Thresh(this->ive_handle, &u8c1_image, &u8c1_image, &iveThreshCtrl, false);
#if DEBUG
  std::cout << "==== [update] diff sub (move) (thr) ====" << std::endl;
  CVI_IVE_BufFlush(this->ive_handle, &frame_diff_1);
  print_IVE_IMAGE_S(frame_diff_1, DEBUG_PRINT_CHANNELS_NUM);
#endif
  memset(frame_diff_2.pu8VirAddr[0], 255, imgSize);
  memset(frame_diff_2.pu8VirAddr[1], 255, imgSize);
  memset(frame_diff_2.pu8VirAddr[2], 255, imgSize);
  // Flush to DRAM before IVE function.
  CVI_IVE_BufFlush(this->ive_handle, &frame_diff_2);
  ret |= CVI_IVE_Sub(this->ive_handle, &frame_diff_2, &frame_diff_1, &frame_diff_2, &iveSubCtrl,
                     false);
#if DEBUG
  std::cout << "==== [update] frame_diff_2 ====" << std::endl;
  CVI_IVE_BufFlush(this->ive_handle, &frame_diff_2);
  print_IVE_IMAGE_S(frame_diff_2, DEBUG_PRINT_CHANNELS_NUM);
#endif

  ret |= CVI_IVE_And(this->ive_handle, &this->diff, &frame_diff_1, &frame_diff_1, false);
  ret |= CVI_IVE_And(this->ive_handle, &min_diff, &frame_diff_2, &frame_diff_2, false);
  u8c1_image.tpu_block = this->diff.tpu_block;
  ret |= CVI_IVE_Or(this->ive_handle, &frame_diff_1, &frame_diff_2, &u8c1_image, false);
#if DEBUG
  std::cout << "==== [update] final diff ====" << std::endl;
  CVI_IVE_BufRequest(this->ive_handle, &this->diff);
  print_IVE_IMAGE_S(this->diff, DEBUG_PRINT_CHANNELS_NUM);
#endif

  CVI_SYS_FreeI(this->ive_handle, &new_frame);
  CVI_SYS_FreeI(this->ive_handle, &frame_diff_1);
  CVI_SYS_FreeI(this->ive_handle, &frame_diff_2);
  CVI_SYS_FreeI(this->ive_handle, &min_diff);

  this->counter = 1;
  return ret;
}

int TamperDetectorMD::detect(VIDEO_FRAME_INFO_S *frame, float *moving_score) {
#if DEBUG
  std::cout << "==== [detect] this->mean ====" << std::endl;
  CVI_IVE_BufFlush(this->ive_handle, &this->mean);
  print_IVE_IMAGE_S(this->mean, DEBUG_PRINT_CHANNELS_NUM);
#endif
  CVI_S32 ret = CVI_SUCCESS;
  IVE_IMAGE_S new_frame;
  // Map the image.
  // CVI_U32 imageLength = frame->stVFrame.u32Length[0] + frame->stVFrame.u32Length[1] +
  //                       frame->stVFrame.u32Length[2];
  // frame->stVFrame.pu8VirAddr[0] = (CVI_U8 *)CVI_SYS_Mmap(frame->stVFrame.u64PhyAddr[0],
  // imageLength); Convert to IVE image. Note this function does not map or unmap for you.
  ret = CVI_IVE_VideoFrameInfo2Image(frame, &new_frame);
  if (ret != CVI_SUCCESS) {
    syslog(LOG_ERR, "Convert to video frame failed with %#x!\n", ret);
  }

#if DEBUG
  std::cout << "==== [detect] input frame ====" << std::endl;
  CVI_IVE_BufFlush(this->ive_handle, &new_frame);
  print_IVE_IMAGE_S(new_frame, DEBUG_PRINT_CHANNELS_NUM);
#endif
#if DEBUG
  std::cout << "==== [detect] this->mean ====" << std::endl;
  CVI_IVE_BufFlush(this->ive_handle, &this->mean);
  print_IVE_IMAGE_S(this->mean, DEBUG_PRINT_CHANNELS_NUM);
#endif

  IVE_DST_IMAGE_S frame_diff, frame_move;
  IVE_DST_IMAGE_S u8c1_image_1;
  CVI_IVE_CreateImage(this->ive_handle, &frame_diff, IVE_IMAGE_TYPE_U8C3_PLANAR, width, height);
  CVI_IVE_CreateImage(this->ive_handle, &frame_move, IVE_IMAGE_TYPE_U8C3_PLANAR, width, height);

  u8c1_image_1.enType = IVE_IMAGE_TYPE_U8C1;

  // Setup control parameter, Sub.
  IVE_SUB_CTRL_S iveSubCtrl;
  iveSubCtrl.enMode = IVE_SUB_MODE_ABS;  // ABS_DIFF
  // Run IVE sub.
  ret |= CVI_IVE_Sub(this->ive_handle, &new_frame, &this->mean, &frame_diff, &iveSubCtrl, false);
  if (ret != CVI_SUCCESS) {
    std::cout << "error: sub" << std::endl;
    exit(1);
  }
#if DEBUG
  std::cout << "==== [detect] frame_diff ====" << std::endl;
  CVI_IVE_BufFlush(this->ive_handle, &frame_diff);
  print_IVE_IMAGE_S(frame_diff, DEBUG_PRINT_CHANNELS_NUM);
#endif

  iveSubCtrl.enMode = IVE_SUB_MODE_NORMAL;
  ret |= CVI_IVE_Sub(this->ive_handle, &frame_diff, &this->diff, &frame_move, &iveSubCtrl, false);
#if DEBUG
  std::cout << "==== [detect] move ====" << std::endl;
  CVI_IVE_BufFlush(this->ive_handle, &frame_move);
  print_IVE_IMAGE_S(frame_move, DEBUG_PRINT_CHANNELS_NUM);
#endif

  u8c1_image_1.tpu_block = frame_move.tpu_block;
  IVE_THRESH_CTRL_S iveThreshCtrl;
  iveThreshCtrl.enMode = IVE_THRESH_MODE_BINARY;
  iveThreshCtrl.u8LowThr = 0;
  iveThreshCtrl.u8MinVal = 0;
  iveThreshCtrl.u8MaxVal = 255;
  ret |= CVI_IVE_Thresh(this->ive_handle, &u8c1_image_1, &u8c1_image_1, &iveThreshCtrl, false);
#if DEBUG
  std::cout << "==== [detect] move (thr) ====" << std::endl;
  CVI_IVE_BufRequest(this->ive_handle, &frame_move);
  print_IVE_IMAGE_S(frame_move, DEBUG_PRINT_CHANNELS_NUM);
#endif

  ret |= CVI_IVE_And(this->ive_handle, &frame_diff, &frame_move, &frame_diff, false);
#if DEBUG
  std::cout << "==== [detect] frame_diff (move) ====" << std::endl;
  CVI_IVE_BufFlush(this->ive_handle, &frame_diff);
  print_IVE_IMAGE_S(frame_diff, DEBUG_PRINT_CHANNELS_NUM);
#endif

  // Refresh CPU cache before CPU use.
  CVI_IVE_BufRequest(this->ive_handle, &frame_diff);
  int diff_sum = 0;
  for (int c = 0; c < 3; c++) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        diff_sum += frame_diff.pu8VirAddr[c][i * strideWidth + j];
      }
    }
  }
#if DEBUG
  std::cout << "diff_sum = " << diff_sum << std::endl;
  std::cout << "height * width = " << height << " * " << width << " = " << height * width
            << std::endl;
  std::cout << "moving_score = " << diff_sum / (height * width) << std::endl;
#endif
  *moving_score = static_cast<float>(diff_sum) / (height * width);

  if (this->update_interval > 0 && this->counter % this->update_interval == 0) {
    this->update(frame);
    this->counter = 1;
  } else {
    this->counter = (this->counter + 1) % this->update_interval;
  }

  CVI_SYS_FreeI(this->ive_handle, &new_frame);
  CVI_SYS_FreeI(this->ive_handle, &frame_diff);
  CVI_SYS_FreeI(this->ive_handle, &frame_move);

  return ret;
}

int TamperDetectorMD::detect(VIDEO_FRAME_INFO_S *frame) {
  // float moving_score, contour_score;
  float moving_score;
  return this->detect(frame, &moving_score);
}

IVE_IMAGE_S &TamperDetectorMD::getMean() { return this->mean; }

IVE_IMAGE_S &TamperDetectorMD::getDiff() { return this->diff; }

void TamperDetectorMD::free() {
  CVI_SYS_FreeI(this->ive_handle, &this->mean);
  CVI_SYS_FreeI(this->ive_handle, &this->diff);
  CVI_IVE_DestroyHandle(this->ive_handle);
}

void TamperDetectorMD::print_info() {
  std::cout << "TamperDetectorMD.nChannels   = " << this->nChannels << std::endl
            << "TamperDetectorMD.strideWidth = " << this->strideWidth << std::endl
            << "TamperDetectorMD.height      = " << this->height << std::endl
            << "TamperDetectorMD.width       = " << this->width << std::endl
            << "TamperDetectorMD.momentum        = " << this->momentum << std::endl
            << "TamperDetectorMD.update_interval = " << this->update_interval << std::endl;
}
