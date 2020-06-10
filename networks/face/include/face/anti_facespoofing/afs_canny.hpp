// Copyright 2019 Bitmain Inc.
// License
// Author
#pragma once
#include "core/image_net.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>

namespace qnn {
namespace vision {

/**
 * @brief The AntiFaceSpoofing class
 *
 */
class AntiFaceSpoofingCanny : public ImageNet {
    public:
    explicit AntiFaceSpoofingCanny(const std::string &model_path, QNNCtx *qnn_ctx = nullptr);
    void Detect(const cv::Mat &srcImg, cv::Rect face_rect, bool &is_real);
    void SetThreshold(float threshold);
    const float GetThreshold();
    const float GetConfidence();

    protected:
    void PrepareImage(const cv::Mat &image, cv::Mat &prepared, float &ratio) override;

    private:
    void DetectAndDrawLine(const cv::Mat &src_img, cv::Mat &dst_img, cv::Rect &rect, int len_threshold);
    int GetMedian(const cv::Mat &img);

    private:
    cv::Rect m_face_rect;
    float m_crop_expand_ratio = 2.0;
    float m_confidence = 0.0;
    float m_threshold;
};

}  // namespace vision
}  // namespace qnn