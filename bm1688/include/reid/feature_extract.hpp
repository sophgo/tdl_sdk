#ifndef FACE_EXTRACT_HPP_
#define FACE_EXTRACT_HPP_
#include "framework/base_model.hpp"

class FeatureExtract : public BaseModel {
 public:
  FeatureExtract(const stNetParam &param) { net_param_ = param; }

  ~FeatureExtract() {}

  bmStatus_t setup();

  bmStatus_t extract(const std::vector<cv::Mat> &images,
                     std::vector<std::vector<float>> &features);

  bmStatus_t extract_direct(const std::vector<std::vector<cv::Mat>> &frame_bgrs,
                            std::vector<std::vector<float>> &features);

 private:
  bmStatus_t postprocess(std::vector<std::vector<float>> &features);
};

#endif
