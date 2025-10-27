#pragma once

#include <vector>
#include "image/base_image.hpp"

#if defined(__CV181X__) || defined(__CV180X__) || defined(__CV182X__) || \
    defined(__CV183X__) || defined(__CV184X__) || defined(__CV186X__)
#include <cvi_comm_vpss.h>
#include <cvi_errno.h>
#include <cvi_math.h>
#include <cvi_sys.h>
#include <cvi_type.h>
#include "cvi_venc.h"
#include "image/vpss_image.hpp"
#endif

class ImageEncoder {
 public:
  ImageEncoder(int VeChn = 1);
  ~ImageEncoder();
  bool encodeFrame(const std::shared_ptr<BaseImage>& image,
                   std::vector<uint8_t>& encode_img, int jpeg_quality = 90);

 private:
#if defined(__CV181X__) || defined(__CV180X__) || defined(__CV182X__) || \
    defined(__CV183X__) || defined(__CV184X__) || defined(__CV186X__)
  VENC_CHN VeChn_;
#endif
};
