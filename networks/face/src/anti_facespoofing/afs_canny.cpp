// Copyright 2019 Bitmain Inc.
// License
// Author
#include "anti_facespoofing/afs_canny.hpp"
#include "utils/image_utils.hpp"
#include <algorithm>

namespace qnn {
namespace vision {

#define CANNY_C 1
#define CANNY_WIDTH 112
#define CANNY_HEIGHT 112
#define CANNY_MEAN (-127.5040)
#define CANNY_SCALE_FACTOR (1.0034)

using cv::Mat;
using std::string;
using std::vector;

AntiFaceSpoofingCanny::AntiFaceSpoofingCanny(const std::string &model_path, QNNCtx *qnn_ctx) :
    ImageNet(model_path, {NetShape(1, CANNY_C, CANNY_WIDTH, CANNY_HEIGHT)},
             CANNY_C, CANNY_WIDTH, CANNY_HEIGHT, false, cv::INTER_LINEAR, qnn_ctx) {
    SetQuanParams({vector<float>{CANNY_MEAN},
                   vector<float>{CANNY_SCALE_FACTOR}});
}

void AntiFaceSpoofingCanny::Detect(const cv::Mat &srcImg, cv::Rect face_rect, bool &is_real) {
    m_face_rect = face_rect;

    ImageNet::Detect({srcImg},
                     [&](OutTensors &out, vector<float> &ratios, size_t start, size_t end){
                         float data = *(out["fc_blob1"].data);
                         float possibility = 1.0 / (1 + exp(-data));

                         m_confidence = possibility;
                         is_real = (possibility > m_threshold);

                         LOGD << "data: " << data;
                         LOGD << "possibility: " << possibility;
                     });
}

void AntiFaceSpoofingCanny::SetThreshold(float threshold) { m_threshold = threshold; }

const float AntiFaceSpoofingCanny::GetThreshold() { return m_threshold; }

const float AntiFaceSpoofingCanny::GetConfidence() { return m_confidence; }

void AntiFaceSpoofingCanny::PrepareImage(const cv::Mat &image, cv::Mat &prepared, float &ratio) {
    // in image coordinate
    int crop_x1 = m_face_rect.x - m_face_rect.width * m_crop_expand_ratio;
    int crop_y1 = m_face_rect.y - m_face_rect.height * m_crop_expand_ratio;
    int crop_x2 = m_face_rect.x + m_face_rect.width * (m_crop_expand_ratio+1);
    int crop_y2 = m_face_rect.y + m_face_rect.height * (m_crop_expand_ratio+1);

    crop_x1 = std::max(0, crop_x1);
    crop_y1 = std::max(0, crop_y1);
    crop_x2 = std::min(image.cols-1, crop_x2);
    crop_y2 = std::min(image.rows-1, crop_y2);

    auto crop_rect = cv::Rect(crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1);
    auto crop_img = image(crop_rect);

    ResizeImage(crop_img, crop_img, CANNY_WIDTH, CANNY_HEIGHT, m_ibuf, m_resize_policy, preserve_ratio);

    // in resized crop_img coordinate
    auto resize_w_ratio = CANNY_WIDTH / float(crop_x2 - crop_x1);
    auto resize_h_ratio = CANNY_HEIGHT / float(crop_y2 - crop_y1);

    int new_face_x1 = (m_face_rect.x - crop_x1) * resize_w_ratio;
    int new_face_y1 = (m_face_rect.y - crop_y1) * resize_h_ratio;
    int new_face_x2 = (m_face_rect.x + m_face_rect.width - crop_x1) * resize_w_ratio;
    int new_face_y2 = (m_face_rect.y + m_face_rect.height - crop_y1) * resize_h_ratio;

    // top, bottom, left, right
    vector<int> len_thresholds{int(m_face_rect.width * resize_w_ratio),
                               int(m_face_rect.width * resize_w_ratio),
                               int(m_face_rect.height * resize_h_ratio),
                               int(m_face_rect.height * resize_h_ratio)};
    vector<cv::Rect> detect_rects;
    detect_rects.emplace_back(cv::Rect(0, 0, CANNY_WIDTH, new_face_y1));
    detect_rects.emplace_back(cv::Rect(0, new_face_y2, CANNY_WIDTH, CANNY_HEIGHT - new_face_y2));
    detect_rects.emplace_back(cv::Rect(0, 0, new_face_x1, CANNY_HEIGHT));
    detect_rects.emplace_back(cv::Rect(new_face_x2, 0, CANNY_WIDTH - new_face_x2, CANNY_HEIGHT));

    prepared = cv::Mat(CANNY_WIDTH, CANNY_HEIGHT, CV_8UC1, cv::Scalar(0, 0, 0));

    cv::rectangle(prepared, cv::Point(new_face_x1, new_face_y1), cv::Point(new_face_x2, new_face_y2), cv::Scalar(255, 255, 255), 1);

    auto it_rect = detect_rects.begin();
    auto it_threshold = len_thresholds.begin();
    for (; it_rect != detect_rects.end() && it_threshold != len_thresholds.end(); it_rect++, it_threshold++) {
        DetectAndDrawLine(crop_img, prepared, *it_rect, *it_threshold);
    }
}

void AntiFaceSpoofingCanny::DetectAndDrawLine(const cv::Mat &src_img, cv::Mat &dst_img, cv::Rect &rect, int len_threshold) {
    if (rect.width <= 0 || rect.height <= 0) {
        return;
    }

    LOGD << "detect rect: " << rect.x << ", " << rect.y << ", " << rect.width << ", " << rect.height;

    auto region_img = src_img(rect);

    cv::cvtColor(region_img, region_img, CV_RGB2GRAY);
    cv::GaussianBlur(region_img, region_img, cv::Size(3, 3), 0, 0);

    auto sigma = 0.33;
    auto median = GetMedian(region_img);
    auto low_threshold = std::max(0, int((1.0 - sigma) * median));
    auto high_threshold = std::min(255, int((1.0 + sigma) * median));

    cv::Canny(region_img, region_img, low_threshold, high_threshold);

    LOGD << "median: " << median;
    LOGD << "low_threshold: " << low_threshold;
    LOGD << "high_threshold: " << high_threshold;

    auto rho = 1.0;
    auto theta = CV_PI / 180.0;
    auto threshold = 30;
    auto min_line_length = len_threshold * 1.0;
    auto max_line_gap = len_threshold * 0.2;

    vector<cv::Vec4i> lines;
    cv::HoughLinesP(region_img, lines, rho, theta, threshold, min_line_length, max_line_gap);

    for (const auto line: lines) {
        cv::Point p1(line[0] + rect.x, line[1] + rect.y);
        cv::Point p2(line[2] + rect.x, line[3] + rect.y);

        cv::line(dst_img, p1, p2, cv::Scalar(255, 255, 255), 1);
    }
}

int AntiFaceSpoofingCanny::GetMedian(const cv::Mat &img) {
    vector<int> vec_from_mat;

    for (int i = 0; i < img.rows; ++i) {
        cv::Mat row_mat = img.row(i);
        std::vector<int> tmpv;
        row_mat.copyTo(tmpv);
        vec_from_mat.insert(vec_from_mat.end(), tmpv.begin(), tmpv.end());
    }

    std::nth_element(vec_from_mat.begin(), vec_from_mat.begin() + vec_from_mat.size() / 2, vec_from_mat.end());

    return vec_from_mat[vec_from_mat.size() / 2];
}

}
}
