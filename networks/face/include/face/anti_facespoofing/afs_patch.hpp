// Copyright 2018 Bitmain Inc.
// License
// Author Yangwen Huang <yangwen.huang@bitmain.com>
/**
 * @file anti_facespoofingpatch.hpp
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

class AntiFaceSpoofingPatch : public ImageNet {
    public:
    /**
     * @brief Construct a new AntiFaceSpoofingPatch object.
     *
     * @param model_path Bmodel file path
     * @param qnn_ctx The QNNCtx is an internal cross-model memory manager
     */
    AntiFaceSpoofingPatch(const std::string& model_path, QNNCtx* qnn_ctx = nullptr);
    /**
     * @brief Destroy the AntiFaceSpoofingPatch object
     *
     */
    virtual ~AntiFaceSpoofingPatch() override{};
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

    const size_t GetCounterVal();

    private:
    size_t m_counter_result = 0;
};

}  // namespace vision
}  // namespace qnn
