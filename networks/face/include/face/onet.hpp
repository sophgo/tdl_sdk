// Copyright 2018 Bitmain Inc.
// License
// Author
/**
 * @file onet.hpp
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

class ONet : public ImageNet {
    public:
    ONet(const std::string &model_path, QNNCtx *qnn_ctx = nullptr);
    virtual ~ONet() override {}
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
     * @brief Find face landmark and regress bbox (regress is optional)
     *
     * @param images The input image
     * @param squared_bboxes The squared roi of the input image
     * @param pad_bboxes The padded bbox of squared box
     * @param face_infos The output face info (w landmark)
     * @param do_regress Turn on / off bbox regression
     * @param use_threshold Turn on / off face info threshold
     */
    void Classify(const cv::Mat &images, const std::vector<face_detect_rect_t> &squared_bboxes,
                  const std::vector<face_detect_rect_t> &pad_bboxes,
                  std::vector<face_info_t> *face_infos, bool do_regress, bool use_threshold);

    protected:
    void PrepareImage(const cv::Mat &image, cv::Mat &prepared, float &ratio) override;

    private:
    float m_bias[10] = {0};  /**< Thd input bias for bm self-trained onet model */
    float m_confidence = 1.; /**< The confidence threshold */
    bool is_nchw = false;
    std::vector<std::string> m_layer_name; /**< The layer name in string */
};

}  // namespace vision
}  // namespace qnn
