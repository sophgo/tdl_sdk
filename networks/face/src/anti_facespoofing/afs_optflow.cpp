// Copyright 2018 Bitmain Inc.
// License
// Author
#include "anti_facespoofing/afs_optflow.hpp"
#include "opencv2/imgproc.hpp"
#include "utils/face_debug.hpp"
#include "utils/function_tracer.h"
#include "utils/image_utils.hpp"
#include "utils/math_utils.hpp"

#define AANTIFACESPOOFINGOPTFLOW_B_MEAN -127.5
#define AANTIFACESPOOFINGOPTFLOW_G_MEAN -127.5
#define AANTIFACESPOOFINGOPTFLOW_R_MEAN -127.5
#define AANTIFACESPOOFINGOPTFLOW_SZ 224
#define AANTIFACESPOOFINGOPTFLOW_C 12
namespace qnn {
namespace vision {

using cv::Mat;
using std::string;
using std::vector;

static const std::vector<NetShape> kPossibleInputShapes = {NetShape(
    1, AANTIFACESPOOFINGOPTFLOW_C, AANTIFACESPOOFINGOPTFLOW_SZ, AANTIFACESPOOFINGOPTFLOW_SZ)};

AntiFaceSpoofingOpticalFlow::AntiFaceSpoofingOpticalFlow(const string &model_path, QNNCtx *qnn_ctx)
    : ImageNet(model_path, kPossibleInputShapes, AANTIFACESPOOFINGOPTFLOW_C,
               AANTIFACESPOOFINGOPTFLOW_SZ, AANTIFACESPOOFINGOPTFLOW_SZ, true, cv::INTER_LINEAR,
               qnn_ctx) {
    Init(0);
}

AntiFaceSpoofingOpticalFlow::AntiFaceSpoofingOpticalFlow(const std::string &model_path,
                                                         float threshold, QNNCtx *qnn_ctx)
    : ImageNet(model_path, kPossibleInputShapes, AANTIFACESPOOFINGOPTFLOW_C,
               AANTIFACESPOOFINGOPTFLOW_SZ, AANTIFACESPOOFINGOPTFLOW_SZ, true, cv::INTER_LINEAR,
               qnn_ctx) {
    Init(threshold);
}

void AntiFaceSpoofingOpticalFlow::Init(float threshold) {
    float model_threshold = GetInPutThreshold();
    printf("threshold=%f\n", GetInPutThreshold());
    SetQuanParams(
        {std::vector<float>{AANTIFACESPOOFINGOPTFLOW_B_MEAN, AANTIFACESPOOFINGOPTFLOW_G_MEAN,
                            AANTIFACESPOOFINGOPTFLOW_R_MEAN},
         std::vector<float>{1 / model_threshold}});
    EnableDequantize(true);
    m_realface_threshold = threshold;
}

void AntiFaceSpoofingOpticalFlow::SetROIScale(float scale) { m_odq.roi_scale = scale; }

void AntiFaceSpoofingOpticalFlow::SetThreshold(float threshold) {
    m_realface_threshold = threshold;
}

const uint AntiFaceSpoofingOpticalFlow::GetThreshold() { return m_realface_threshold; }

const float AntiFaceSpoofingOpticalFlow::GetConfidence() { return m_currconfidence; }

void AntiFaceSpoofingOpticalFlow::PrepareImage(const cv::Mat &image, cv::Mat &prepared,
                                               float &ratio) {
    prepared = image;
}

// WARNING: Very slow
cv::Mat GetPaddedROI(const cv::Mat &input, int top_left_x, int top_left_y, int width, int height,
                     cv::Scalar paddingColor) {
    int bottom_right_x = top_left_x + width;
    int bottom_right_y = top_left_y + height;

    cv::Mat output;
    if (top_left_x < 0 || top_left_y < 0 || bottom_right_x > input.cols ||
        bottom_right_y > input.rows) {
        // border padding will be required
        int border_left = 0, border_right = 0, border_top = 0, border_bottom = 0;

        if (top_left_x < 0) {
            width = width + top_left_x;
            border_left = -1 * top_left_x;
            top_left_x = 0;
        }
        if (top_left_y < 0) {
            height = height + top_left_y;
            border_top = -1 * top_left_y;
            top_left_y = 0;
        }
        if (bottom_right_x > input.cols) {
            width = width - (bottom_right_x - input.cols);
            border_right = bottom_right_x - input.cols;
        }
        if (bottom_right_y > input.rows) {
            height = height - (bottom_right_y - input.rows);
            border_bottom = bottom_right_y - input.rows;
        }

        cv::Rect R(top_left_x, top_left_y, width, height);
        cv::copyMakeBorder(input(R), output, border_top, border_bottom, border_left, border_right,
                           cv::BORDER_CONSTANT, paddingColor);
    } else {
        // no border padding required
        cv::Rect R(top_left_x, top_left_y, width, height);
        output = input(R);
    }
    return output;
}

bool AntiFaceSpoofingOpticalFlow::GetNImages(const cv::Mat &image, const cv::Rect &face_roi,
                                             const uint queue_num) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    cv::Mat gray_scale;
    cv::cvtColor(image, gray_scale, cv::COLOR_BGR2GRAY);
    if (m_odq.img_queue.empty()) {
        int max_length = std::max(face_roi.width, face_roi.height);
        if (m_odq.roi_scale != 1.f) {
            max_length *= m_odq.roi_scale;
        }
        int cx = face_roi.x + face_roi.width / 2;
        int cy = face_roi.y + face_roi.height / 2;
        int new_x = cx - max_length / 2;
        int new_y = cy - max_length / 2;
        m_odq.roi = cv::Rect(new_x, new_y, max_length, max_length);
        cv::Mat cropped =
            GetPaddedROI(gray_scale, new_x, new_y, max_length, max_length, cv::Scalar(0, 0, 0));
        cv::Mat resized;
        cv::resize(cropped, resized,
                   cv::Size(AANTIFACESPOOFINGOPTFLOW_SZ, AANTIFACESPOOFINGOPTFLOW_SZ),
                   cv::INTER_LINEAR);
        m_odq.img_queue.push_back(cropped);
        return false;
    } else if (m_odq.img_queue.size() < queue_num) {
        if (m_odq.cur_interval >= m_odq.frame_interval) {
            cv::Mat cropped = GetPaddedROI(gray_scale, m_odq.roi.x, m_odq.roi.y, m_odq.roi.width,
                                           m_odq.roi.height, cv::Scalar(0, 0, 0));
            cv::Mat resized;
            cv::resize(cropped, resized,
                       cv::Size(AANTIFACESPOOFINGOPTFLOW_SZ, AANTIFACESPOOFINGOPTFLOW_SZ),
                       cv::INTER_LINEAR);
            m_odq.img_queue.push_back(resized);
            m_odq.cur_interval = 0;
        } else {
            m_odq.cur_interval += 1;
        }
        return false;
    }

    // return true if m_odq.img_queue.size() >= queue_num
    return true;
}

cv::Mat VizFlow(const cv::Mat &flow) {
    cv::Mat hsv1(cv::Size(flow.rows, flow.cols), CV_8UC1, cv::Scalar(255));
    std::vector<cv::Mat> flow_split;
    cv::split(flow, flow_split);
    cv::Mat magnitude, angle;
    cv::cartToPolar(flow_split[0], flow_split[1], magnitude, angle);  //调用库函数
    cv::Mat hsv0 = angle * 180 / M_PI / 2;
    hsv0.convertTo(hsv0, CV_8UC1);
    cv::Mat hsv2;
    cv::normalize(magnitude, hsv2, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    std::vector<cv::Mat> viz_vec = {hsv0, hsv1, hsv2};
    cv::Mat hsv, bgr;
    cv::merge(viz_vec, hsv);
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    return bgr;
}

void CalcBatchOptFlow(const std::vector<cv::Mat> &input, std::vector<cv::Mat> &output,
                      const bool is_color) {
    output.clear();
    for (size_t i = 0; i < input.size() - 1; i++) {
        cv::Mat flow;
        cv::calcOpticalFlowFarneback(input[i], input[i + 1], flow, 0.5, 5, 15, 3, 7, 1.5,
                                     cv::OPTFLOW_FARNEBACK_GAUSSIAN);
        if (is_color) {
            output.push_back(VizFlow(flow));
        } else {
            output.push_back(flow);
        }
    }
}

void AntiFaceSpoofingOpticalFlow::DoOptFlow(std::vector<cv::Mat> &output_vec) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    std::vector<cv::Mat> flow_color, flow_raw;
    CalcBatchOptFlow(m_odq.img_queue, flow_color, true);
    m_odq.img_queue.clear();
}

void AntiFaceSpoofingOpticalFlow::Detect(const cv::Mat &img, const cv::Rect &face_roi,
                                         bool &is_real) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    is_real = false;
    if (img.empty()) {
        return;
    }
    if (!GetNImages(img, face_roi, 5)) {
        return;
    }
    std::vector<cv::Mat> images;
    DoOptFlow(images);

    ImageNet::Detect(images, [&](OutTensors &out, vector<float> &ratios, int start, int end) {
        assert(start >= 0 && start < end && end <= int(images.size()));
        // DequantizeOutputTensors(out_tensors);
        auto &out_tensor = out["logits_MatMul"];
        float *data = reinterpret_cast<float *>(out_tensor.data);
        float prob[2];
        qnn::math::SoftMax(data, prob, 2, 1, 1);
        LOGI << "Prob(0, 1) = ( " << prob[0] << ", " << prob[1]
             << "). Threshold = " << m_realface_threshold;
        m_currconfidence = prob[0];
        if (prob[0] > m_realface_threshold) {
            is_real = true;
        }
    });
}

}  // namespace vision
}  // namespace qnn
