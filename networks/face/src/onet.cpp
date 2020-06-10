#include "onet.hpp"

#include <cassert>

#include "mtcnn_helperfunc.hpp"
#include "utils/face_debug.hpp"
#include "utils/function_tracer.h"
#include "utils/log_common.h"

namespace qnn {
namespace vision {

#define ONET_WIDTH 48
#define ONET_HEIGHT 48
#define ONET_DEFAULT_THRESHOLD (0.7)
#define ONET_OURS_THRESHOLD (0.9)
#define ONET_MEAN -128
#define ONET_C 3
#define ONET_SCALE_FACTOR float(1) / 0.9965

using cv::Mat;
using std::string;
using std::vector;

static const vector<NetShape> kPossibleInputShapes = {
    NetShape(128, 3, 48, 48), NetShape(1, 3, 48, 48),  NetShape(16, 3, 48, 48),
    NetShape(2, 3, 48, 48),   NetShape(32, 3, 48, 48), NetShape(4, 3, 48, 48),
    NetShape(64, 3, 48, 48),  NetShape(8, 3, 48, 48)};

ONet::ONet(const string &model_path, QNNCtx *qnn_ctx)
    : ImageNet(model_path, kPossibleInputShapes, ONET_C, ONET_WIDTH, ONET_HEIGHT, false,
               cv::INTER_LINEAR, qnn_ctx) {
    SetQuanParams({std::vector<float>{ONET_MEAN, ONET_MEAN, ONET_MEAN},
                   std::vector<float>{ONET_SCALE_FACTOR}});
    m_confidence = is_nchw ? ONET_OURS_THRESHOLD : ONET_DEFAULT_THRESHOLD;
    SetNCHW(is_nchw);
    m_ibuf.push_back({"transpose"});
}

void ONet::SetConfidenceThreshold(float value) { m_confidence = value; }

void ONet::SetNCHW(bool value) {
    is_nchw = value;
    m_layer_name.resize(3);
    if (is_nchw) {
        m_bias[0] = 0.33987370133399963;
        m_bias[1] = 0.6888470649719238;
        m_bias[2] = 0.5172482132911682;
        m_bias[3] = 0.3650050461292267;
        m_bias[4] = 0.6614519953727722;
        m_bias[5] = 0.3818088173866272;
        m_bias[6] = 0.3750465214252472;
        m_bias[7] = 0.5871966481208801;
        m_bias[8] = 0.754140317440033;
        m_bias[9] = 0.746856689453125;
        m_layer_name[0] = "conv6_1";
        m_layer_name[1] = "conv6_2";
        m_layer_name[2] = "conv6_3";
    } else {
        for (int i = 0; i < 10; i++) {
            m_bias[i] = 0.;
        }
        m_layer_name[0] = "conv6-1";
        m_layer_name[1] = "conv6-2";
        m_layer_name[2] = "conv6-3";
    }
}

void ONet::PrepareImage(const cv::Mat &image, cv::Mat &prepared, float &ratio) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    cv::Mat resized;
    ratio =
        ResizeImage(image, resized, net_width, net_height, m_ibuf, m_resize_policy, preserve_ratio);
    if (is_nchw) {
        prepared = resized;
    } else {
        cv::transpose(resized, m_ibuf[2].img);
        prepared = m_ibuf[2].img;
    }
}

void ONet::Classify(const Mat &image, const vector<face_detect_rect_t> &squared_bboxes,
                    const vector<face_detect_rect_t> &pad_bboxes, vector<face_info_t> *face_infos,
                    bool do_regress, bool use_threshold) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    if (image.empty()) {
        assert(false);
        return;
    }
    int box_number = squared_bboxes.size();
    face_infos->clear();
    face_infos->reserve(box_number);
    std::vector<face_info_regression_t> o_results;
    vector<Mat> face_images;
    face_images.reserve(box_number);
    for (int i = 0; i < box_number; ++i) {
        int pad_top = std::abs(pad_bboxes[i].y1 - squared_bboxes[i].y1);
        int pad_left = std::abs(pad_bboxes[i].x1 - squared_bboxes[i].x1);
        int pad_right = std::abs(pad_bboxes[i].x2 - squared_bboxes[i].x2);
        int pad_bottom = std::abs(pad_bboxes[i].y2 - squared_bboxes[i].y2);
#if !Release
        LOGD << "[ONET] image size:" << image.rows << "x" << image.cols;
        LOGD << "[ONET] crop image: idx " << i << " (" << pad_bboxes[i].x1 - 1 << ", "
             << pad_bboxes[i].y1 - 1 << ", " << pad_bboxes[i].x2 << ", " << pad_bboxes[i].y2 << ")";
#endif
#ifdef USE_VPP
        {
        BITMAIN_FUNCTION_TRACE("Onet Vpp Crop & Border");

        cv::Rect roi(pad_bboxes[i].x1-1, pad_bboxes[i].y1-1,
                     pad_bboxes[i].x2-pad_bboxes[i].x1+1, pad_bboxes[i].y2-pad_bboxes[i].y1+1);
        cv::Mat vpp_board = cv::vpp::crop_resize_border(const_cast<Mat&>(image),
                                                        roi.height, roi.width,
                                                        roi.x, roi.y, roi.width, roi.height,
                                                        pad_top, pad_bottom, pad_left, pad_right);

        face_images.emplace_back(vpp_board);
        }
#else
        {
        BITMAIN_FUNCTION_TRACE("Onet CPU Crop & Border");
        Mat crop_img = image(cv::Range(pad_bboxes[i].y1 - 1, pad_bboxes[i].y2),
                             cv::Range(pad_bboxes[i].x1 - 1, pad_bboxes[i].x2));
        cv::copyMakeBorder(crop_img, crop_img, pad_left, pad_right, pad_top, pad_bottom,
                           cv::BORDER_CONSTANT, cv::Scalar(0));
        face_images.emplace_back(crop_img);
        }
#endif
    }
    if (face_images.size() == 0) {
        return;
    }

    ImageNet::Detect(face_images,
                     [&](OutTensors &out, vector<float> &ratios, size_t start, size_t end) {
                         assert(start >= 0 && start < end && end <= face_images.size());
                         OutputTensor &points_offset(out[m_layer_name[2].c_str()]);
                         OutputTensor &reg(out[m_layer_name[1].c_str()]);
                         OutputTensor &confidence(out[m_layer_name[0].c_str()]);

                         float *conf_data = confidence.data;
                         float *reg_data = reg.data;
                         float *pts_x = points_offset.data;
                         float *pts_y = points_offset.data + 5;

                         for (size_t i = start, offset = 0; i < end; ++i, ++offset) {
                             face_detect_rect_t rect(squared_bboxes[i]);

                             float x0 = *(conf_data + offset * 2);
                             float x1 = *(conf_data + offset * 2 + 1);
                             float max = std::max(x0, x1);
                             float f0 = std::exp(x0 - max);
                             float f1 = std::exp(x1 - max);
                             float score = f1 / (f0 + f1);
                             rect.score = score;
                             if (score < m_confidence && use_threshold) {
                                 pts_x += 10;
                                 pts_y += 10;
                                 continue;
                             }
                             face_pts_t pts;
                             float h = rect.y2 - rect.y1 + 1;
                             float w = rect.x2 - rect.x1 + 1;
                             for (int j = 0, k = 5; j < 5; j++, ++k) {
                                 pts.x[j] = rect.x1 + (*pts_x + m_bias[j]) * w - 1;
                                 pts.y[j] = rect.y1 + (*pts_y + m_bias[k]) * h - 1;
                                 ++pts_x;
                                 ++pts_y;
                             }
                             pts_x += 5;
                             pts_y += 5;

                             std::array<float, 4> regression;
                             regression[0] = ((float *)reg_data)[4 * offset + 0];
                             regression[1] = ((float *)reg_data)[4 * offset + 1];
                             regression[2] = ((float *)reg_data)[4 * offset + 2];
                             regression[3] = ((float *)reg_data)[4 * offset + 3];
                             o_results.emplace_back(face_info_regression_t(rect, pts, regression));
                         }
                     });
    LOGD << "ONet End.";
    if (do_regress) {
        BoxRegress(o_results, 3, face_infos);
    } else {
        for (size_t i = 0; i < o_results.size(); i++) {
            face_infos->emplace_back(face_info_t{o_results[i].bbox, o_results[i].face_pts});
        }
    }
    BITMAIN_DRAWFDINFO_SAVE("onet.png", image, *face_infos);
}

}  // namespace vision
}  // namespace qnn
