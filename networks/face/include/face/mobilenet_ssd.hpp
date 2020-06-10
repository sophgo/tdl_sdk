/**
 * @file mobilenet_ssd.hpp
 * @ingroup NETWORKS_OD
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
 * @brief MobileNetSSD anchor shape structure.
 *
 */
struct AnchorShape {
    float width;
    float height;
    float center_x;
    float center_y;
    float variance[4];
};

/**
 * @brief MobileNetSSD anchor with layer name structure.
 *
 */
struct AnchorSSD {
    std::string name;
    std::vector<AnchorShape> shapes;
};

class MobileNetSSD : public ImageNet {
    public:
    /**
     * @brief Construct a new MobileNetSSD object.
     *
     * @param model_path Bmodel file path
     * @param anchor_path Anchor json file path
     * @param qnn_ctx The QNNCtx is an internal cross-model memory manager
     */
    MobileNetSSD(const std::string &model_path, const std::string &anchor_path,
                 QNNCtx *qnn_ctx = nullptr);

    /**
     * @brief Destroy the MobileNetSSD object
     *
     */
    virtual ~MobileNetSSD() override {}

    /**
     * @brief Detect input face images and output human rects.
     *
     * @param images Input images
     * @param results Output human rects
     */
    void Detect(const std::vector<cv::Mat> &images,
                std::vector<std::vector<face_detect_rect_t>> &results);

    private:
    int m_total_anchor_num;           /**< Total output anchor num. */
    std::vector<AnchorSSD> m_anchors; /**< Shape of anchors for each layer. */
    std::unique_ptr<float[], malloc_delete<float>> m_confidence_data; /**< Softmax buffer. */
};

}  // namespace vision
}  // namespace qnn
