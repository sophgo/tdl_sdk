// Copyright 2018 Bitmain Inc.
// License
// Author Yangwen Huang <yangwen.huang@bitmain.com>
/**
 * @file pnet.hpp
 * @ingroup NETWORKS_FD
 */
#pragma once
#include "core/image_net.hpp"
#include "face.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace qnn {
namespace vision {

/**
 * @brief The PNet class
 *
 */
class PNet : public ImageNet {
    public:
    /**
     * @brief Construct a new PNet object
     *
     * @param model_path The bmodel path for pnet
     * @param shapes The input shape. Note this variable may be deprecated in
     * the future
     * @param qnn_ctx The QNNCtx is an internal cross-model memory manager.
     */
    PNet(const std::string &model_path, const std::vector<NetShape> &shapes,
         QNNCtx *qnn_ctx = nullptr);

    /**
     * @brief Destroy the PNet object
     *
     */
    virtual ~PNet() override {}

    /**
     * @brief Set the Confidence Threshold
     *
     * @param value The confidence threshold input
     */
    void SetConfidenceThreshold(const float value);

    /**
     * @brief Set model channel in nchw (true) or ncwh (false)
     *
     * @param value true for nchw / false for ncwh
     */
    void SetNCHW(bool value);
    void SetFastExp(bool fast);

    /**
     * @brief Find the possible face bbox
     *
     * @param image The input image in the format of rgb
     * @param dst The output resized image PNET_INPUT_IMG_
     * @param out_ratio The resize ratio
     * @param face_rects The output face bbox
     */
    void Classify(const cv::Mat &image, cv::Mat *dst, float *out_ratio,
                  std::vector<face_detect_rect_t> *face_rects);

    private:
    void PrePadding(const cv::Mat &src, cv::Mat *dst, int *nearest_width, int *nearest_height);
    /**
     * @brief Generate bbox from tensor buffer
     *
     * @param output_info The output info from bmodel
     * @param confidence The confidence output from tensor
     * @param reg The reg output from tensor
     * @param scale The input image scale
     * @param thresh The confidence threshold
     * @param image_width The image width
     * @param image_height The image height
     * @param image_id The input image id
     * @param condidate Output face bbox candidates
     */
    void GenerateBoundingBox(const bmnet_output_info_t &output_info, const OutputTensor &confidence,
                             const OutputTensor &reg, const float scale, const float thresh,
                             const int image_width, const int image_height, const int image_id,
                             std::vector<face_info_regression_t> *condidate);

    float m_confidence = 1.; /**< The confidence threshold */
    bool is_nchw = false;
    bool m_fast_exp = true;
    std::vector<std::string> m_layer_name; /**< The layer name in string */
    std::vector<NetShape> m_supported_shapes;
};

}  // namespace vision
}  // namespace qnn
