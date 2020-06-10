// Copyright 2018 Bitmain Inc.
// License
// Author Yangwen Huang <yangwen.huang@bitmain.com>
#include "mtcnn.hpp"
#include "mtcnn_helperfunc.hpp"
#include "opencv2/imgproc.hpp"
#include "utils/face_debug.hpp"
#include "utils/simd_wrapper.h"
#include "utils/function_tracer.h"

namespace qnn {
namespace vision {

using cv::Mat;
using std::string;
using std::vector;

// clang-format off
static const vector<NetShape> PNet_kPossibleInputShapes = {
    NetShape(1, 3, 324, 576), NetShape(1, 3, 162, 288), NetShape(1, 3, 81, 144), NetShape(1, 3, 41, 72),
    NetShape(1, 3, 21, 36)};

static const vector<NetShape> PNet_kPossibleInputShapes_ncwh = {
    NetShape(1, 3, 576, 324), NetShape(1, 3, 288, 162), NetShape(1, 3, 144, 81), NetShape(1, 3, 72, 41),
    NetShape(1, 3, 36, 21)};
// clang-format on
// TODO: FIXME: Change PNet's constructor when bmtap2 can read input from
// bmodel.

Mtcnn::Mtcnn(const std::vector<std::string> &models_path, TENSORSEQ type, QNNCtx *qnn_ctx) {
    m_pnet_model_path = models_path[0];
    mp_rnet = std::make_unique<RNet>(models_path[1], qnn_ctx);
    mp_onet = std::make_unique<ONet>(models_path[2], qnn_ctx);
    SetTensorSequence(type, qnn_ctx);
    LOGD << "MTCNN init complete!";
}

void Mtcnn::SetConfidenceThreshold(float p_value, float r_value) {
    mp_pnet->SetConfidenceThreshold(p_value);
    mp_rnet->SetConfidenceThreshold(r_value);
}

void Mtcnn::SetConfidenceThreshold(float p_value, float r_value, float o_value) {
    SetConfidenceThreshold(p_value, r_value);
    mp_onet->SetConfidenceThreshold(o_value);
}

void Mtcnn::SetTensorSequence(TENSORSEQ type, QNNCtx *qnn_ctx) {
    bool is_nchw = false;
    std::vector<NetShape> pnet_shape;
    float threshold_pnet = 0.9;
    float threshold_rnet = 0.9;
    float threshold_onet = 0.9;
    switch (type) {
        case TENSORSEQ::NCHW: {
            is_nchw = true;
            m_seqtype = type;
            pnet_shape = PNet_kPossibleInputShapes;
            threshold_pnet = 0.9;
            threshold_rnet = 0.9;
            threshold_onet = 0.9;
        } break;
        case TENSORSEQ::NCWH: {
            is_nchw = false;
            m_seqtype = type;
            pnet_shape = PNet_kPossibleInputShapes_ncwh;
            threshold_pnet = 0.6;
            threshold_rnet = 0.6;
            threshold_onet = 0.7;
        } break;
        default:
            LOGE << "Please choose a supported type.";
    }
    if (mp_pnet) {
        mp_pnet.reset();
    }
    mp_pnet = std::make_unique<PNet>(m_pnet_model_path, pnet_shape, qnn_ctx);

    mp_pnet->SetNCHW(is_nchw);
    mp_rnet->SetNCHW(is_nchw);
    mp_onet->SetNCHW(is_nchw);
    SetConfidenceThreshold(threshold_pnet, threshold_rnet, threshold_onet);
}

void Mtcnn::SetFastExp(bool use) { mp_pnet->SetFastExp(use); }

void Mtcnn::Detect(const Mat &image, vector<face_info_t> &results) {
    if (image.empty()) {
        assert(false);
        return;
    }
    vector<Mat> images = {image};
    vector<vector<face_info_t>> all_results;
    Detect(images, all_results);
    results.swap(all_results.at(0));
}

void Mtcnn::Detect(const vector<Mat> &images, vector<vector<face_info_t>> &results) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    if (images.empty()) {
        return;
    }

    results.clear();
    results.reserve(images.size());
    for (size_t i = 0; i < images.size(); i++) {
        vector<face_info_t> cur_results;
        const cv::Mat &image(images[i]);

        LOGD << "Stage 1";
        vector<face_detect_rect_t> origin_bboxes;
        float out_ratio = 1;
        cv::Mat zoomed;
        mp_pnet->Classify(image, &zoomed, &out_ratio, &origin_bboxes);

        if (!origin_bboxes.empty()) {
            LOGD << "Stage 2";
            vector<face_detect_rect_t> squared_bboxes;
            vector<face_detect_rect_t> pad_bboxes;
            squared_bboxes.assign(origin_bboxes.begin(), origin_bboxes.end());
            // Pad image to square for rnet
            AdjustBbox2Square(squared_bboxes);
            Padding(origin_bboxes, squared_bboxes, pad_bboxes, zoomed);
            mp_rnet->Classify(zoomed, squared_bboxes, pad_bboxes, &origin_bboxes);

            LOGD << "Stage 3";
            squared_bboxes.clear();
            pad_bboxes.clear();
            squared_bboxes.assign(origin_bboxes.begin(), origin_bboxes.end());
            // Pad image to square for onet
            AdjustBbox2Square(squared_bboxes);
            Padding(origin_bboxes, squared_bboxes, pad_bboxes, zoomed);
            vector<face_info_t> face_output, face_nms;
            mp_onet->Classify(zoomed, squared_bboxes, pad_bboxes, &face_output, true, true);

            // Post processing of the result from onet
            NonMaximumSuppression(face_output, face_nms, 0.7, 'm');
            RestoreBboxWithRatio(face_nms, out_ratio, cur_results);
            BITMAIN_DRAWFDINFO_SAVE("mtcnn_results.jpg", image, cur_results);
            LOGD << "MTCNN End";
        }
        results.emplace_back(cur_results);
    }
}

}  // namespace vision
}  // namespace qnn