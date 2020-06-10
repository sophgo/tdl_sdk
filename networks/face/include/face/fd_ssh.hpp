// Copyright 2018 Bitmain Inc.
// License
// Author
/**
 * @file fd_tiny_ssh.hpp
 * @ingroup NETWORKS_FD
 */
#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "face.hpp"
#include "mtcnn_helperfunc.hpp"
#include "onet.hpp"
#include "ssh.hpp"

namespace qnn {
namespace vision {

class FdSSH : public FaceDetector {
    public:
    FdSSH(const std::vector<std::string> &models, const SSH_TYPE &type, int input_width,
          int input_height, PADDING_POLICY padding, QNNCtx *qnn_ctx = nullptr);
    virtual ~FdSSH() override {}
    void Detect(const cv::Mat &image, std::vector<face_info_t> &results) override;
    void Detect(const std::vector<cv::Mat> &images,
                std::vector<std::vector<face_info_t>> &results) override;
    void SetThreshold(float threshold);

    private:
    void Classify(const cv::Mat &image, const std::vector<face_detect_rect_t> &squared_bboxes,
                  const std::vector<face_detect_rect_t> &pad_bboxes,
                  std::vector<face_info_t> &face_infos);

    std::unique_ptr<SSH> ssh;
    std::unique_ptr<ONet> onet;
};

}  // namespace vision
}  // namespace qnn
