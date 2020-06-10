// Copyright 2018 Bitmain Inc.
// License
// Author Yangwen Huang <yangwen.huang@bitmain.com>
/**
 * @file rnet.hpp
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
 * @brief The RNet class
 *
 */
class RNet : public ImageNet {
    public:
    /**
     * @brief Construct a new RNet object
     *
     * @param model_path The bmodel path for rnet
     * @param qnn_ctx The QNNCtx is an internal cross-model memory manager.
     */
    RNet(const std::string &model_path, QNNCtx *qnn_ctx = nullptr);

    /**
     * @brief Destroy the RNet object
     *
     */
    virtual ~RNet() override {}

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

    /**
     * @brief Classify subimages in one image from pnet
     *
     * @param images The input image in the format of rgb
     * @param squared_bboxes The original squared boxes
     * @param pad_bboxes The padded boxes
     * @param face_rects The output refined result from pnet
     */
    void Classify(const cv::Mat &images, const std::vector<face_detect_rect_t> &squared_bboxes,
                  const std::vector<face_detect_rect_t> &pad_bboxes,
                  std::vector<face_detect_rect_t> *face_rects);

    private:
    float m_confidence = 1.; /**< The confidence threshold */
    bool is_nchw = false;
    std::vector<std::string> m_layer_name; /**< The layer name in string */
};

}  // namespace vision
}  // namespace qnn
