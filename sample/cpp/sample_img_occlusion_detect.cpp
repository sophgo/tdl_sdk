#include <iostream>
#include <memory>
#include "cv/occlusion_detect/occlusion_detect.hpp"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Usage: ./occlusion_detect image.jpg" << std::endl;
    return -1;
  }

  std::shared_ptr<BaseImage> image = ImageFactory::readImage(argv[1]);
  if (!image) {
    std::cerr << "Failed to load image" << std::endl;
    return -1;
  }

  cvtdl_occlusion_meta_t meta;
  meta.crop_bbox = {0.0f, 0.0f, 1.0f, 1.0f};
  meta.laplacian_th = 30;
  meta.occ_ratio_th = 0.5;
  meta.sensitive_th = 5;

  OcclusionDetector detector;
  int ret = detector.detect(image, &meta);
  if (ret != 0) {
    std::cerr << "Detection failed" << std::endl;
    return -1;
  }

  std::cout << "Occlusion score: " << meta.occ_score << std::endl;
  std::cout << "Occlusion class: " << meta.occ_class << std::endl;

  return 0;
}
