#ifndef OBJ_LANDMARK_HPP_
#define OBJ_LANDMARK_HPP_
#include "common/obj_det_utils.hpp"
#include "framework/base_model.hpp"

class ObjectLandmark : public BaseModel {
 public:
  ObjectLandmark(const stNetParam &param) { net_param_ = param; }

  ~ObjectLandmark() {}

  bmStatus_t setup();

  bmStatus_t detect(const std::vector<cv::Mat> &image, const float threshold,
                    std::vector<stObjPts> &objPt);

  void preprocess(const cv::Mat &img, cv::Mat &tmp_resized,
                  std::vector<cv::Mat> &tmp_bgr, std::vector<cv::Mat> &bgr);

  bmStatus_t detect_direct(const std::vector<std::vector<cv::Mat>> &frame_bgrs,
                           float threshold,
                           const std::vector<cv::Size> &frame_sizes,
                           std::vector<stObjPts> &objPt);

 private:
  bmStatus_t preprocess(std::vector<cv::Mat>::const_iterator &img_iter,
                        int batch_size);

  bmStatus_t postprocess(const std::vector<cv::Size> &img_size,
                         const float threshold, std::vector<stObjPts> &objPts);

  std::vector<std::string> output_layers_;
};
#endif