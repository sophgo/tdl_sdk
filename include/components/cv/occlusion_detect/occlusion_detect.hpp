#ifndef OCCLUSION_DETECTOR_HPP
#define OCCLUSION_DETECTOR_HPP

#include "image/base_image.hpp"

struct cvtdl_bbox_t {
  float x1, y1, x2, y2;
};

struct cvtdl_occlusion_meta_t {
  cvtdl_bbox_t crop_bbox;
  double laplacian_th;
  double occ_ratio_th;
  int sensitive_th;
  float occ_score;
  int occ_class;
};

class OcclusionDetector {
 public:
  OcclusionDetector();

  // 主检测接口
  int detect(std::shared_ptr<BaseImage> image, cvtdl_occlusion_meta_t* meta);

  // 重置状态
  void reset();

 private:
  std::vector<int> occlusionStates_;
};

#endif  // OCCLUSION_DETECTOR_HPP
