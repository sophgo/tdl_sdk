// Copyright 2018 Bitmain Inc.
// License
// Author Yangwen Huang <yangwen.huang@bitmain.com>
/**
 * @file facepose.hpp
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

class Facepose : public ImageNet {
    public:
    /**
     * @brief Construct a new Facepose object.
     *
     * @param model_path Bmodel file path
     * @param qnn_ctx The QNNCtx is an internal cross-model memory manager
     */
    Facepose(const std::string &model_path, QNNCtx *qnn_ctx = nullptr);
    /**
     * @brief Destroy the Facepose object
     *
     */
    virtual ~Facepose() override{};
    /**
     * @brief Detect input face images and output 3-DOF features in vector
     *
     * @param images Input images
     * @param results Output 3-DOF features
     */
    void Detect(const std::vector<cv::Mat> &images, std::vector<std::vector<float>> &results);
};

}  // namespace vision
}  // namespace qnn
