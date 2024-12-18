#include "image/base_image.hpp"

#include <cstdint>
#include <random>

int32_t BaseImage::randomFill() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);
  auto virtual_address = getVirtualAddress();
  uint32_t image_size = getImageByteSize();
  if (virtual_address.empty() || image_size == 0) {
    return -1;
  }
  uint8_t* data = virtual_address[0];
  for (size_t i = 0; i < image_size; ++i) {
    data[i] = static_cast<uint8_t>(dis(gen));
  }
  int32_t ret = flushCache();
  if (ret != 0) {
    std::cerr << "base image flush cache failed" << std::endl;
    return ret;
  }
  return 0;
}
