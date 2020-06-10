// Copyright 2018 Bitmain Inc.
// License
// Author
#include "fd_ssh.hpp"
#include "utils/log_common.h"
#include "utils/function_tracer.h"
#include <cassert>
#include <memory>

namespace qnn {
namespace vision {

using namespace std;
using cv::Mat;

FdSSH::FdSSH(const vector<string> &models, const SSH_TYPE &type, int input_width, int input_height,
             PADDING_POLICY padding, QNNCtx *qnn_ctx) {
    assert(models.size() == 2);
    if (models.size() != 2) {
        return;
    }
    ssh = make_unique<SSH>(models[0], type, input_width, input_height, padding, qnn_ctx);
    onet = make_unique<ONet>(models[1], qnn_ctx);
    onet->SetNCHW(true);
}

void FdSSH::Detect(const Mat &image, vector<face_info_t> &results) {
    if (image.empty()) {
        assert(false);
        return;
    }
    vector<Mat> images = {image};
    vector<vector<face_info_t>> all_results;
    Detect(images, all_results);
    results.swap(all_results.at(0));
}

void FdSSH::Detect(const vector<Mat> &images, vector<vector<face_info_t>> &results) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    vector<vector<face_detect_rect_t>> all_bboxes;
    ssh->Detect(images, all_bboxes);

    assert(images.size() == all_bboxes.size());

    for (size_t i = 0; i < images.size(); ++i) {
        vector<face_detect_rect_t> &origin_bboxes(all_bboxes.at(i));
        const Mat &image(images.at(i));
        vector<face_detect_rect_t> squared_bboxes;
        vector<face_detect_rect_t> pad_bboxes;
        squared_bboxes.assign(origin_bboxes.begin(), origin_bboxes.end());
        Padding(origin_bboxes, squared_bboxes, pad_bboxes, image);

        vector<face_info_t> face_infos;
        vector<face_info_t> result_face_infos;
        onet->Classify(image, squared_bboxes, pad_bboxes, &face_infos, false, false);
        RestoreBbox(origin_bboxes, face_infos, result_face_infos);
        results.emplace_back(result_face_infos);
    }
}

void FdSSH::SetThreshold(float threshold) { ssh->SetThreshold(threshold); }

}  // namespace vision
}  // namespace qnn
