#ifndef DETECTION_YOLOV5_HPP_
#define DETECTION_YOLOV5_HPP_

#include "common/obj_det_utils.hpp"
#include "framework/base_model.hpp"

class YOLOV5 : public BaseModel {
 public:
  YOLOV5(const stNetParam &param) { net_param_ = param; }
  ~YOLOV5() {}
  bmStatus_t setup();
  bmStatus_t detect(const std::vector<cv::Mat> &images, const float threshold,
                    std::vector<std::vector<ObjectBox>> &results);
  bmStatus_t postprocess(
      const std::vector<cv::Size> &frame_sizes, const float threshold,
      const std::vector<std::vector<float>> &frame_scale_params,
      std::vector<std::vector<ObjectBox>> &results);

  bmStatus_t postprocess_3out(
      const std::vector<cv::Size> &frame_sizes, const float threshold,
      const std::vector<std::vector<float>> &frame_scale_params,
      std::vector<std::vector<ObjectBox>> &results);
  bmStatus_t detect_direct(
      const std::vector<std::vector<cv::Mat>> &frame_bgrs, float threshold,
      const std::vector<std::vector<float>> &frame_rescale_params,
      const std::vector<cv::Size> &frame_sizes,
      std::vector<std::vector<ObjectBox>> &results);
  void set_nms_thresh(float thresh) { nms_thresh_ = thresh; }
  void set_label_nms_thresh(const std::map<int, float> &label_nms_thresh) {
    label_nms_threshold_ = label_nms_thresh;
  }

 private:
  float nms_threshold_;
  std::map<int, float> label_nms_threshold_;

  float nms_thresh_ = 0.5;
};
#endif
