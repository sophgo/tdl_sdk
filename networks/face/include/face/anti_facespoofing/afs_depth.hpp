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

#define DEPTH_DO_VISUAL true

namespace qnn {
namespace vision {

class AntiFaceSpoofingDepth : public ImageNet {
    public:
    /**
     * @brief Construct a new AntiFaceSpoofingDepth object.
     *
     * @param model_path Bmodel file path
     * @param qnn_ctx The QNNCtx is an internal cross-model memory manager
     */
    AntiFaceSpoofingDepth(const std::string& model_path, QNNCtx* qnn_ctx = nullptr);

    /**
     * @brief Construct a new AntiFaceSpoofingDepth object.
     *
     * @param model_path Bmodel file path
     * @param threshold The threshold for detecting realface
     * @param qnn_ctx The QNNCtx is an internal cross-model memory manager
     */
    AntiFaceSpoofingDepth(const std::string& model_path, size_t threshold,
                          QNNCtx* qnn_ctx = nullptr);

    /**
     * @brief Destroy the AntiFaceSpoofingDepth object
     *
     */
    virtual ~AntiFaceSpoofingDepth() override{};

    /**
     * @brief Set the threshold of detecting a real face.
     *
     * @param threshold Threshold value
     */
    void SetThreshold(size_t threshold);

    /**
     * @brief Get the threshold of detecting a real face.
     *
     * @return threshold
     */
    const uint GetThreshold();

    /**
     * @brief Get the NN counter result.
     *
     * @return const size_t The counter value
     */
    const size_t GetCounterVal();

    /**
     * @brief Detect AntiFaceSpoofing by patch based network
     *
     * @param crop_img Input cropped image
     * @param is_real Output bool result
     */
    void Detect(const cv::Mat& crop_img, bool& is_real);

    /**
     * @brief Override ImageNet's function and do nothing
     *
     * @param image The input image
     * @param prepared The ouepue image
     * @param ratio The output resize ratio if resized
     */
    void PrepareImage(const cv::Mat& image, cv::Mat& prepared, float& ratio) override;

    /**
     * @brief Get the depth visual map result
     *
     * @return cv::Mat The depth map
     */
    cv::Mat GetDepthVisualMat();

    private:
    /**
     * @brief Init function for Constructor
     *
     * @param threshold The realface threshold
     */
    void Init(size_t threshold);

    /**
     * @brief
     *
     * @param crop_img Input cropped image
     * @param images Output vectors
     */
    void Preprocess(const cv::Mat& crop_img, std::vector<cv::Mat>& images);

    /**
     * @brief Use DetectInference instead of Detect in ImageNet cause ImageNet uses fixed channel
     * size of 3
     *
     * @param images The input image array
     * @param is_real Real face flag
     */
    void DetectInference(const std::vector<cv::Mat>& images, bool& is_real);

    std::vector<int> m_cvt_code; /**< OpenCV cvt code list */
    size_t m_realface_threshold; /**< The real face threshold */
    size_t m_counter_result = 0; /**< The NN result */
    cv::Mat m_visual;            /**< The visual map from NN */

    bool m_make_visual = DEPTH_DO_VISUAL; /**< Do visual results in host */
};

}  // namespace vision
}  // namespace qnn
