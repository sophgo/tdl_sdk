#ifndef FACE_CSSD_HPP_
#define FACE_CSSD_HPP_
#include "face/face_common.hpp"
#include "framework/base_model.hpp"
#include "netcompact/tensor.hpp"

class FaceCSSD : public BaseModel {
 public:
  explicit FaceCSSD(const stNetParam &param);

  ~FaceCSSD() {}

  bmStatus_t setup();

  bmStatus_t detect(const std::vector<cv::Mat> &images, const float threshold,
                    std::vector<std::vector<FaceRect>> &results);

  bmStatus_t detect_direct(
      const std::vector<std::vector<cv::Mat>> &frame_bgrs, float threshold,
      const std::vector<std::vector<float>> &frame_rescale_params,
      const std::vector<cv::Size> &frame_sizes,
      std::vector<std::vector<FaceRect>> &results);

 private:
  bmStatus_t postprocess(
      const std::vector<cv::Size> &frame_sizes, const float threshold,
      const std::vector<std::vector<float>> &frame_scale_params,
      std::vector<std::vector<FaceRect>> &results);

  std::shared_ptr<nncompact::Tensor> resized_img_buffer_;
  std::vector<cv::Mat> tmp_bgr_planar_;
  std::vector<float> prior_box_data_;
};

#endif
