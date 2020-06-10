// Copyright 2018 Bitmain Inc.
// License
// Author Yangwen Huang <yangwen.huang@bitmain.com>
/**
 * @file anti_facespoofingdepth.hpp
 * @ingroup NETWORKS_FD
 */
#pragma once

#include "core/image_net.hpp"
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace qnn {
namespace vision {

class AntiFaceSpoofingClassify : public ImageNet {
    public:
    /**
     * @brief Construct a new AntiFaceSpoofingClassify object.
     *
     * @param model_path Bmodel file path
     * @param qnn_ctx The QNNCtx is an internal cross-model memory manager
     */
    AntiFaceSpoofingClassify(const std::string& model_path, QNNCtx* qnn_ctx = nullptr);

    /**
     * @brief Construct a new AntiFaceSpoofingClassify object.
     *
     * @param model_path Bmodel file path
     * @param threshold The threshold for detecting realface
     * @param qnn_ctx The QNNCtx is an internal cross-model memory manager
     */
    AntiFaceSpoofingClassify(const std::string& model_path, float threshold,
                             QNNCtx* qnn_ctx = nullptr);

    /**
     * @brief Destroy the AntiFaceSpoofingClassify object
     *
     */
    virtual ~AntiFaceSpoofingClassify() override{};

    /**
     * @brief Set the threshold of detecting a real face.
     *
     * @param threshold Threshold value
     */
    void SetThreshold(float threshold);

    /**
     * @brief Get the threshold of detecting a real face.
     *
     * @return threshold
     */
    const uint GetThreshold();

    /**
     * @brief Get the confidence of the current result from a face.
     *
     * @return confidence
     */
    const float GetConfidence();

    /**
     * @brief
     *
     * @param crop_img Input cropped image
     * @param images Output vectors
     */
    void Preprocess(const cv::Mat& crop_img, std::vector<cv::Mat>& images);

    /**
     * @brief Detect AntiFaceSpoofing by patch based network
     *
     * @param crop_img Input cropped image
     * @param is_real Output bool result
     */
    void Detect(const cv::Mat& crop_img, bool& is_real);

    private:
    /**
     * @brief Init function for Constructor
     *
     * @param threshold The realface threshold
     */
    void Init(float threshold);

    float m_currconfidence = 0.0;
    std::vector<int> m_cvt_code;
    float m_realface_threshold; /**< The real face threshold */
};

}  // namespace vision
}  // namespace qnn
