#ifndef CARPLATE_OCR_HPP_
#define CARPLATE_OCR_HPP_

#include "common/obj_det_utils.hpp"
#include "framework/base_model.hpp"
class CarplateOCR : public BaseModel {
 public:
  CarplateOCR(const stNetParam &param) { net_param_ = param; }

  ~CarplateOCR() {}

  bmStatus_t setup();

  bmStatus_t detect(const std::vector<cv::Mat> &image, const float threshold,
                    std::vector<stCarplate> &carplates);

  void preprocess(const cv::Mat &img, cv::Mat &tmp_resized,
                  std::vector<cv::Mat> &tmp_bgr, std::vector<cv::Mat> &bgr);

  bmStatus_t detect_direct(const std::vector<std::vector<cv::Mat>> &frame_bgrs,
                           float threshold, std::vector<stCarplate> &carplates);

 private:
  bmStatus_t preprocess(std::vector<cv::Mat>::const_iterator &img_iter,
                        int batch_size);

  bmStatus_t postprocess(const float threshold,
                         std::vector<stCarplate> &stCarplate);

  std::vector<std::string> output_layers_;
};

#endif
