#ifndef FACE_LANDMARK_HPP_
#define FACE_LANDMARK_HPP_
#include "face/face_common.hpp"
#include "framework/base_model.hpp"

class FaceLandmark : public BaseModel {
 public:
  FaceLandmark(const stNetParam &param) { net_param_ = param; }

  ~FaceLandmark() {}

  bmStatus_t setup();

  bmStatus_t detect(const std::vector<cv::Mat> &image, const float threshold,
                    std::vector<FacePts> &facePt);

  void preprocess(const cv::Mat &img, cv::Mat &tmp_resized,
                  cv::Mat &tmp_transposed, std::vector<cv::Mat> &tmp_bgr,
                  std::vector<cv::Mat> &bgr);

  bmStatus_t detect_direct(const std::vector<std::vector<cv::Mat>> &frame_bgrs,
                           float threshold,
                           const std::vector<cv::Size> &frame_sizes,
                           std::vector<FacePts> &facePt);

 private:
  bmStatus_t preprocess(std::vector<cv::Mat>::const_iterator &img_iter,
                        int batch_size);

  bmStatus_t postprocess(const std::vector<cv::Size> &img_size,
                         const float threshold, std::vector<FacePts> &facePts);

  std::vector<std::string> output_layers_;

  cv::Mat tmp_resized_;
  cv::Mat tmp_transposed_;
};

/****************************************************************
 * For BMMark
 ***************************************************************/

class BMMark : public BaseModel {
 public:
  explicit BMMark(const stNetParam &param);

  ~BMMark() override = default;

  bmStatus_t preprocess(const cv::Mat &img, cv::Mat &tmp_resized,
                        cv::Mat &tmp_transposed, std::vector<cv::Mat> &tmp_bgr,
                        std::vector<cv::Mat> &bgr);
  bmStatus_t detect_direct(const std::vector<std::vector<cv::Mat>> &frame_bgrs,
                           float threshold,
                           const std::vector<cv::Size> &frame_sizes,
                           std::vector<FacePts> &facePt);

  bmStatus_t preprocess(const cv::Mat &img, std::vector<cv::Mat> &dst,
                        IMGTransParam &trans_param);

  bmStatus_t detect_direct(const std::vector<std::vector<cv::Mat>> &rois,
                           const std::vector<IMGTransParam> &trans_params,
                           const float &threshold,
                           std::vector<FacePts> &facePt);

  bmStatus_t detect(const std::vector<cv::Mat> &imgs, const float &threshold,
                    std::vector<FacePts> &pts);

  bmStatus_t setup() override;

 private:
  bmStatus_t preprocess(std::vector<cv::Mat>::const_iterator &img_iter,
                        int batch_size,
                        std::vector<IMGTransParam> &trans_params);

  bmStatus_t postprocess(const std::vector<IMGTransParam>::const_iterator &iter,
                         const float &threshold, std::vector<FacePts> &facePts);

 private:
  int size = 48;
  std::vector<std::string> output_layers_;

  // std::vector<IMGTransParam> trans_params;
};

#endif
