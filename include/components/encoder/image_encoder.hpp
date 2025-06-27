#pragma once

#include <vector>
#include "image/base_image.hpp"
class ImageEncoder {
 public:
  bool encodeFrame(const std::shared_ptr<BaseImage>& image,
                   std::vector<uint8_t>& encode_img, int jpeg_quality = 90);
};
