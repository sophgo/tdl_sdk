#pragma once

#include <vector>
#include "image/base_image.hpp"

#if defined(__CV181X__) || defined(__CV180X__) || defined(__CV182X__) || \
    defined(__CV183X__) || defined(__CV184X__) || defined(__CV186X__)
#include <cvi_comm_vpss.h>
#include <cvi_errno.h>
#include <cvi_sys.h>
#include <cvi_type.h>
#include "cvi_venc.h"
#include "image/vpss_image.hpp"
#endif

class ViEncoder {
 public:
  ViEncoder(int VeChn = 0, int width = 1920, int height = 1080, int fps = 30,
            int bitrate = 4096, int gop = 60);
  ~ViEncoder();

  bool encodeFrame(const std::shared_ptr<BaseImage>& image,
                   std::vector<uint8_t>& encode_stream);

  int getEncoderMode() { return encoder_mode_; }
  void setEncoderMode(int encoder_mode) { encoder_mode_ = encoder_mode; }

 private:
  int encoder_mode_;
  int width_, height_, fps_, bitrate_, gop_;

#if defined(__CV181X__) || defined(__CV180X__) || defined(__CV182X__) || \
    defined(__CV183X__) || defined(__CV184X__) || defined(__CV186X__)
  VENC_CHN VeChn_;
  bool is_initialized_;
  bool is_started_;

  int32_t setH264Entropy();
  int32_t setH264Vui();
  int32_t setH264Trans();
  int32_t start();
#endif
};