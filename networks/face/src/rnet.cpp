// Copyright 2018 Bitmain Inc.
// License
// Author Yangwen Huang <yangwen.huang@bitmain.com>
#include "rnet.hpp"
#include "mtcnn_helperfunc.hpp"
#include "utils/face_debug.hpp"
#include "utils/function_tracer.h"
#include "utils/image_utils.hpp"

namespace qnn {
namespace vision {

using cv::Mat;
using std::string;
using std::vector;

#define RNET_WIDTH 24
#define RNET_HEIGHT 24
#define RNET_DEFAULT_THRESHOLD (0.6)
#define RNET_OURS_THRESHOLD (0.9)
// Not yet used
#define RNET_MEAN -128
#define RNET_C 3
#define RNET_SCALE_FACTOR float(1) / 1.02049

static const vector<NetShape> kPossibleInputShapes = {
    NetShape(128, 3, 24, 24), NetShape(1, 3, 24, 24),  NetShape(16, 3, 24, 24),
    NetShape(2, 3, 24, 24),   NetShape(32, 3, 24, 24), NetShape(4, 3, 24, 24),
    NetShape(64, 3, 24, 24),  NetShape(8, 3, 24, 24)};

RNet::RNet(const string &model_path, QNNCtx *qnn_ctx)
    : ImageNet(model_path, kPossibleInputShapes, RNET_C, RNET_WIDTH, RNET_HEIGHT, false,
               cv::INTER_LINEAR, qnn_ctx) {
    // float scale_threshold = GetInPutThreshold();
    SetQuanParams(
        {std::vector<float>{RNET_MEAN, RNET_MEAN, RNET_MEAN}, std::vector<float>{RNET_SCALE_FACTOR}});

    m_confidence = is_nchw ? RNET_OURS_THRESHOLD : RNET_DEFAULT_THRESHOLD;
    SetNCHW(is_nchw);
}

void RNet::SetConfidenceThreshold(const float value) { m_confidence = value; }

void RNet::SetNCHW(bool value) {
    is_nchw = value;
    m_layer_name.resize(2);
    if (is_nchw) {
        m_layer_name[0] = "conv5_1";
        m_layer_name[1] = "conv5_2";
    } else {
        m_layer_name[0] = "conv5-1";
        m_layer_name[1] = "conv5-2";
    }
}

void RNet::Classify(const cv::Mat &image, const vector<face_detect_rect_t> &squared_bboxes,
                    const vector<face_detect_rect_t> &pad_bboxes,
                    vector<face_detect_rect_t> *face_rects) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    if (image.empty()) {
        assert(false);
        return;
    }

    int box_number = squared_bboxes.size();
    face_rects->clear();
    face_rects->reserve(box_number);
    int input_width = RNET_WIDTH, input_height = RNET_HEIGHT;
    int face_start_idx = 0;
    std::vector<face_info_regression_t> candidate, candidate_nms;

    // The policy here uses maximum batch selection. This is slightly different
    // from onet but same as bmiva.
    int batch_num = GetBoxPerBatch(box_number);
    box_number -= batch_num;
    while (batch_num > 0) {
        BITMAIN_FUNCTION_TRACE("Rnet While Batch");
        LOGD << "handle batch size: " << batch_num;

        NetShape shape(batch_num, 3, input_height, input_width);
        auto tensor_pair = SelectTensor(shape);
        InputTensor &input_tensor(*tensor_pair.first);
        OutTensors &output_tensor((*tensor_pair.second));

        // We get the pointer at the very first and make offset to put image
        // data into memory buffer.
        char *ptr = input_tensor.data;
        for (int i = 0, face_i = face_start_idx; i < batch_num; i++, ++face_i) {
            int pad_top = std::abs(pad_bboxes[face_i].y1 - squared_bboxes[face_i].y1);
            int pad_left = std::abs(pad_bboxes[face_i].x1 - squared_bboxes[face_i].x1);
            int pad_right = std::abs(pad_bboxes[face_i].x2 - squared_bboxes[face_i].x2);
            int pad_bottom = std::abs(pad_bboxes[face_i].y2 - squared_bboxes[face_i].y2);
#if !Release
            LOGD << "[RNET] image size:" << image.rows << "x" << image.cols;
            LOGD << "[RNET] crop image: idx " << face_i << " (" << pad_bboxes[face_i].x1 - 1 << ", "
                 << pad_bboxes[face_i].y1 - 1 << ", " << pad_bboxes[face_i].x2 << ", "
                 << pad_bboxes[face_i].y2 << ")";
#endif
            vector<cv::Mat> channels;
            ptr = WrapInputLayer(channels, ptr, 3, input_height, input_width);

#ifdef USE_VPP
            cv::Rect roi(pad_bboxes[i].x1-1, pad_bboxes[i].y1-1,
                         pad_bboxes[i].x2-pad_bboxes[i].x1+1, pad_bboxes[i].y2-pad_bboxes[i].y1+1);
            cv::Mat vpp_board = cv::vpp::crop_resize_border(const_cast<Mat&>(image),
                                                            roi.height, roi.width,
                                                            roi.x, roi.y, roi.width, roi.height,
                                                            pad_top, pad_bottom, pad_left, pad_right);

            cv::Mat src_channels[3];
            cv::vpp::resize_border_split(vpp_board, input_height, input_width, 0, 0, 0, 0, src_channels);

            auto &quan_val = GetRefQuanParams();
            NormalizeToU8(src_channels, quan_val.mean_values, quan_val.input_scales, channels);
#else
            cv::Mat proccessed(input_height, input_width, CV_8UC3);

            Mat crop_img = image(cv::Range(pad_bboxes[face_i].y1 - 1, pad_bboxes[face_i].y2),
                                 cv::Range(pad_bboxes[face_i].x1 - 1, pad_bboxes[face_i].x2));
            // TODO: FIXME: The resize policy here is not the same as the base
            cv::copyMakeBorder(crop_img, proccessed, pad_left, pad_right, pad_top, pad_bottom,
                               cv::BORDER_CONSTANT, cv::Scalar(0));
            cv::resize(proccessed, proccessed, cv::Size(input_width, input_height), 0, 0,
                       cv::INTER_LINEAR);

            if (!is_nchw) {
                cv::transpose(proccessed, proccessed);
            }

            auto &quan_val = GetRefQuanParams();
            NormalizeAndSplitToU8(proccessed, quan_val.mean_values, quan_val.input_scales, channels);
#endif
        }
        Inference(shape);
        DequantizeOutputTensors(output_tensor);

        OutputTensor &reg(output_tensor[m_layer_name[1].c_str()]);
        OutputTensor &confidence(output_tensor[m_layer_name[0].c_str()]);
        float *conf_data = confidence.data;

        for (int i = 0, face_i = face_start_idx; i < batch_num; i++, ++face_i) {
            // Compute the score
            float x0 = *(conf_data + i * 2);
            float x1 = *(conf_data + i * 2 + 1);
            float max = std::max(x0, x1);
            float f0 = std::exp((x0 - max));
            float f1 = std::exp((x1 - max));
            float score = f1 / (f0 + f1);
            // If larger than threshold confidence, get the bbox and regression
            // info
            if (score > m_confidence) {
                face_detect_rect_t faceRect;
                faceRect.score = score;
                faceRect.x1 = squared_bboxes[face_i].x1;
                faceRect.y1 = squared_bboxes[face_i].y1;
                faceRect.x2 = squared_bboxes[face_i].x2;
                faceRect.y2 = squared_bboxes[face_i].y2;
                faceRect.id = squared_bboxes[face_i].id;

                std::array<float, 4> regression;
                regression[0] = ((float *)reg.data)[4 * i + 0];
                regression[1] = ((float *)reg.data)[4 * i + 1];
                regression[2] = ((float *)reg.data)[4 * i + 2];
                regression[3] = ((float *)reg.data)[4 * i + 3];
                candidate.emplace_back(face_info_regression_t(faceRect, regression));
            }
        }
        // Update the rest unpredicted image index.
        face_start_idx += batch_num;
        batch_num = GetBoxPerBatch(box_number);
        box_number -= batch_num;
    }
    // Do nms and apply regression for ouput
    NonMaximumSuppression(candidate, candidate_nms, 0.7, 'u');
    BoxRegress(candidate_nms, 2, face_rects);
    BITMAIN_DRAWFDRECT_SAVE("rnet_output.jpg", image, *face_rects);
}

}  // namespace vision
}  // namespace qnn
