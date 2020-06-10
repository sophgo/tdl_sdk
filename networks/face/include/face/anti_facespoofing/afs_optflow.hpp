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

struct OptDataQueue {
    // From settings
    uint frame_interval = 3; /**< The frame interval for collecting image stacks */
    float roi_scale = 1.f;   /**< The scaling factor for detected face roi */

    // Data
    uint cur_interval = 0;          /**< The counter used with the frame_interval */
    cv::Rect roi;                   /**< The input detected face roi */
    std::vector<cv::Mat> img_queue; /**< The collected image queue */
};

class AntiFaceSpoofingOpticalFlow : public ImageNet {
    public:
    /**
     * @brief Construct a new AntiFaceSpoofingOpticalFlow object.
     *
     * @param model_path Bmodel file path
     * @param qnn_ctx The QNNCtx is an internal cross-model memory manager
     */
    AntiFaceSpoofingOpticalFlow(const std::string& model_path, QNNCtx* qnn_ctx = nullptr);

    /**
     * @brief Construct a new AntiFaceSpoofingOpticalFlow object.
     *
     * @param model_path Bmodel file path
     * @param threshold The threshold for detecting realface
     * @param qnn_ctx The QNNCtx is an internal cross-model memory manager
     */
    AntiFaceSpoofingOpticalFlow(const std::string& model_path, float threshold,
                                QNNCtx* qnn_ctx = nullptr);

    /**
     * @brief Destroy the AntiFaceSpoofingOpticalFlow object
     *
     */
    virtual ~AntiFaceSpoofingOpticalFlow() override{};

    /**
     * @brief Set the scale of a face ROI.
     *
     * @param threshold scale value
     */
    void SetROIScale(float scale);

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
     * @brief Override ImageNet's function and do nothing
     *
     * @param image The input image
     * @param prepared The ouepue image
     * @param ratio The output resize ratio if resized
     */
    void PrepareImage(const cv::Mat& image, cv::Mat& prepared, float& ratio) override;

    /**
     * @brief Detect AntiFaceSpoofing by patch based network
     *
     * @param img Input image
     * @param roi Detect face roi
     * @param is_real Output bool result
     */
    void Detect(const cv::Mat& img, const cv::Rect& face_roi, bool& is_real);

    private:
    /**
     * @brief Init function for Constructor
     *
     * @param threshold The realface threshold
     */
    void Init(float threshold);

    /**
     * @brief Get N stack roi-images cross frames with given interval
     *
     * @param image The raw image
     * @param face_roi The detected face roi
     * @param queue_num The queue num
     * @return true If the size of the stack is larger than queue_num
     * @return false If the size of the stack is smaller than queue_num
     */
    bool GetNImages(const cv::Mat& image, const cv::Rect& face_roi, const uint queue_num);

    /**
     * @brief Do optical flow and vizualize flow for network input
     *
     * @param output_vec The output optical flow vector
     */
    void DoOptFlow(std::vector<cv::Mat>& output_vec);

    OptDataQueue m_odq; /**< The Data queue structure used in GetNImages */

    float m_currconfidence;     /**< The confidence for deciding real face */
    float m_realface_threshold; /**< The real face threshold */
};

}  // namespace vision
}  // namespace qnn
