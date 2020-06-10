// Copyright 2018 Bitmain Inc.
// License
// Author
/**
 * @file re_id.hpp
 * @ingroup NETWORKS_PR
 */
#pragma once
#include "core/image_net.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace qnn {
namespace vision {

class ReID : public ImageNet {
    public:
    ReID(const std::string &model_path, QNNCtx *qnn_ctx = nullptr);
    virtual ~ReID() override {}
    void Detect(const std::vector<cv::Mat> &images, std::vector<std::vector<float>> &results);
};

}  // namespace vision
}  // namespace qnn
