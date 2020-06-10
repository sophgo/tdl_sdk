// Copyright 2018 Bitmain Inc.
// License
// Author Yangwen Huang <yangwen.huang@bitmain.com>
/**
 * @file mtcnn.hpp
 * @ingroup NETWORKS_FD
 */
#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "core/image_net.hpp"
#include "onet.hpp"
#include "pnet.hpp"
#include "rnet.hpp"

namespace qnn {
namespace vision {

/**
 * @brief The MTCNN class
 *
 */
class Mtcnn : public FaceDetector {
    public:
    /**
     * @brief Construct a new Mtcnn object
     *
     * @param models_path The filepath of p net, r net, onet bmodel
     * @param type Set the Tensor Sequence
     * @param qnn_ctx The QNNCtx is an internal cross-model memory manager.
     */
    explicit Mtcnn(const std::vector<std::string> &models_path, TENSORSEQ type = TENSORSEQ::NCHW,
                   QNNCtx *qnn_ctx = nullptr);

    /**
     * @brief Destroy the Mtcnn object
     *
     */
    virtual ~Mtcnn() {}

    /**
     * @brief Set the Confidence Threshold
     *
     * @param p_value The threshold for pnet
     * @param r_value The threshold for rnet
     */
    void SetConfidenceThreshold(float p_value, float r_value);

    void SetConfidenceThreshold(float p_value, float r_value, float o_value);

    /**
     * @brief Set the Tensor Sequence
     *
     * @param type sequence type
     * @param qnn_ctx The QNNCtx is an internal cross-model memory manager.
     */
    void SetTensorSequence(TENSORSEQ type, QNNCtx *qnn_ctx);
    void SetFastExp(bool fast);

    /**
     * @brief Detect face and pts
     *
     * @param images The input images, must be rgb
     * @param results The output result
     */
    void Detect(const std::vector<cv::Mat> &images,
                std::vector<std::vector<face_info_t>> &results) override;
    void Detect(const cv::Mat &image, std::vector<face_info_t> &results) override;

    private:
    std::unique_ptr<PNet> mp_pnet; /**< The pnet smart ptr container */
    std::string m_pnet_model_path;
    std::unique_ptr<RNet> mp_rnet;         /**< The rnet smart ptr container */
    std::unique_ptr<ONet> mp_onet;         /**< The onet smart ptr container */
    TENSORSEQ m_seqtype = TENSORSEQ::NCWH; /**< The tensor seq select */
};

}  // namespace vision
}  // namespace qnn