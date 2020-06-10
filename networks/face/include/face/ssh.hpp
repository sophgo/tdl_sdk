/**
 * @file tiny_ssh.hpp
 * @ingroup NETWORKS_FD
 */
#pragma once

#include "core/image_net.hpp"
#include "face.hpp"
#include "proposal_layer_nocaffe.hpp"
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace qnn {
namespace vision {

class SSH : public ImageNet {
    public:
    SSH(const std::string &model_path, const SSH_TYPE &type, int input_width, int input_height,
        const PADDING_POLICY padding, QNNCtx *qnn_ctx = nullptr);
    virtual ~SSH() override {}
    void Detect(const cv::Mat &image, std::vector<face_detect_rect_t> &results);
    void Detect(const std::vector<cv::Mat> &image,
                std::vector<std::vector<face_detect_rect_t>> &results);
    void SetThreshold(float threshold);

    protected:
    void PrepareImage(const cv::Mat &image, cv::Mat &prepared, float &ratio) override;

    private:
    void do_proposal(OutTensors &output_tensors, std::vector<face_detect_rect_t> &boxes);
    void handle_proposal(std::vector<face_detect_rect_t> &boxes, std::vector<size_t> &layer_sizes);
    int *get_output_shape();

    std::vector<std::unique_ptr<ProposalLayer>> proposal_layers;
    std::unique_ptr<float[]> post_probs_buf;
    int m_input_width;
    int m_input_height;
    PADDING_POLICY padding_policy;
    float m_threshold;
};

}  // namespace vision
}  // namespace qnn
