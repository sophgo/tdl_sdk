// Copyright 2018 Bitmain Inc.
// License
// Author Yangwen Huang <yangwen.huang@bitmain.com>
#include "pnet.hpp"

#include "mtcnn_helperfunc.hpp"
#include "utils/face_debug.hpp"
#include "utils/function_tracer.h"
#include "utils/image_utils.hpp"
#include "utils/math_utils.hpp"

namespace qnn {
namespace vision {

#define PNET_WIDTH 12
#define PNET_HEIGHT 12
#define PNET_DEFAULT_THRESHOLD (0.6)
#define PNET_OURS_THRESHOLD (0.9)
#define PNET_DEFAULT_FACTOR (0.5)
#define PNET_DEFAULT_MINSIZE (40)

#define PNET_MEAN -128
#define PNAT_C 3
#define PNET_SCALE_FACTOR float(1) / 0.9965

using cv::Mat;
using qnn::math::SoftMax;
using std::string;
using std::vector;

PNet::PNet(const string &model_path, const std::vector<NetShape> &shapes, QNNCtx *qnn_ctx)
    : ImageNet(model_path, shapes, PNAT_C, PNET_WIDTH, PNET_HEIGHT, true, cv::INTER_LINEAR,
               qnn_ctx) {
    m_supported_shapes = shapes;
    SetQuanParams({std::vector<float>{PNET_MEAN, PNET_MEAN, PNET_MEAN},
                   std::vector<float>{PNET_SCALE_FACTOR}});
    m_confidence = is_nchw ? PNET_OURS_THRESHOLD : PNET_DEFAULT_THRESHOLD;
    SetNCHW(is_nchw);
}

void PNet::SetConfidenceThreshold(float value) { m_confidence = value; }

void PNet::SetNCHW(bool value) {
    is_nchw = value;
    m_layer_name.resize(2);
    if (is_nchw) {
        m_layer_name[0] = "conv4_1";
        m_layer_name[1] = "conv4_2";
    } else {
        m_layer_name[0] = "conv4-1";
        m_layer_name[1] = "conv4-2";
    }
}

void PNet::SetFastExp(bool fast) { m_fast_exp = fast; }

void PNet::PrePadding(const cv::Mat &src, cv::Mat *dst, int *nearest_width, int *nearest_height) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    int dis = std::numeric_limits<int>::max();
    int size_wh[2] = {0};
    if (is_nchw) {
        for (size_t i = 0; i < m_supported_shapes.size(); i++) {
            const int &w = m_supported_shapes[i].w;
            const int &h = m_supported_shapes[i].h;
            // Caution! mtcnn origin uses wh instead of hw
            int &&dis_h = h - src.rows;
            int &&dis_w = w - src.cols;
            int min_dis = std::abs(std::max(dis_w, dis_h));
            if (min_dis < dis) {
                size_wh[0] = w;
                size_wh[1] = h;
                dis = min_dis;
            }
        }
    } else {
        for (size_t i = 0; i < m_supported_shapes.size(); i++) {
            const int &w = m_supported_shapes[i].h;
            const int &h = m_supported_shapes[i].w;
            // Caution! mtcnn origin uses wh instead of hw
            int &&dis_h = h - src.rows;
            int &&dis_w = w - src.cols;
            int min_dis = std::abs(std::max(dis_w, dis_h));
            if (min_dis < dis) {
                size_wh[0] = w;
                size_wh[1] = h;
                dis = min_dis;
            }
        }
    }

    float ratio = std::max((float)src.cols / size_wh[0], (float)src.rows / size_wh[1]);

    int pad_bottom = size_wh[1] * ratio - src.rows;
    int pad_right = size_wh[0] * ratio - src.cols;

    LOGD << "[Prepadded] w, h: " << size_wh[0] << " " << size_wh[1];
    LOGD << "[Origin] w h: " << src.cols << " " << src.rows;
    LOGD << "Padded value: " << pad_right << " " << pad_bottom;

#ifdef USE_VPP
    if (pad_bottom || pad_right) {
        *dst = cv::vpp::border(const_cast<Mat&>(src), 0, pad_bottom, 0, pad_right);
    } else {
        *dst = src;
    }
#else
    if (pad_bottom || pad_right) {
        cv::copyMakeBorder(src, *dst, 0, pad_bottom, 0, pad_right, cv::BORDER_CONSTANT,
                           cv::Scalar(0));
    } else {
        *dst = src;
    }
#endif
    *nearest_width = size_wh[0];
    *nearest_height = size_wh[1];
    return;
}

void PNet::Classify(const cv::Mat &image, cv::Mat *dst, float *out_ratio,
                    std::vector<face_detect_rect_t> *face_rects) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    if (image.empty()) {
        return;
    }
    int batch_num = 1;
    face_rects->clear();
    face_rects->reserve(batch_num);
    // For each image starts here
    *out_ratio = 1.f;
    cv::Mat base_img;
    int init_width, init_height;
    float w_scale, h_scale;
    if (is_nchw) {
        // Ours uses 1920 x 1080
        *out_ratio = 1.f;
        PrePadding(image, dst, &init_width, &init_height);
        base_img = (*dst);
        w_scale = init_width;
        h_scale = init_height;
    } else {
        // Python ver uses 960 * 540 + transpose
        cv::Mat zoomed;
        cv::cvtColor(image, zoomed, cv::COLOR_BGR2RGB);
        PrePadding(zoomed, dst, &init_width, &init_height);
        cv::transpose(*dst, base_img);
        w_scale = init_height;
        h_scale = init_width;
    }

    // Calculate scale factor
    vector<std::vector<float>> scales_wh;
    float minWH = std::min(w_scale, h_scale);
    while (minWH > 12) {  // The pnet pyramid scales
        std::vector<float> val;
        val.emplace_back(w_scale);
        val.emplace_back(h_scale);
        scales_wh.emplace_back(val);
        minWH *= 0.5;
        w_scale *= 0.5;
        h_scale *= 0.5;
    }

    std::vector<face_info_regression_t> total_bbox;
    // TODO: Here we did not use Detect from base class cause the different
    // between the resize policy and quantize formula.
    for (size_t i = 0; i < (uint32_t)scales_wh.size(); i++) {
        double scale = scales_wh[i][0] / base_img.cols;
        int ws = ceil(scales_wh[i][0]);
        int hs = ceil(scales_wh[i][1]);
        /////////////////////////////////////////////////////////////////////
        // Detect starts here

        auto tensor_pair = SelectTensor(NetShape(1, 3, hs, ws));
        InputTensor &input_tensor(*tensor_pair.first);
        OutTensors &output_tensor((*tensor_pair.second));

        LOGD << scales_wh[i][0] << " " << scales_wh[i][1];
        LOGD << "Success select input shape " << input_tensor.shape.n << " " << input_tensor.shape.c
             << " " << input_tensor.shape.h << " " << input_tensor.shape.w;

        char *ptr = input_tensor.data;
        vector<Mat> channels;
        ptr = WrapInputLayer(channels, ptr, 3, hs, ws);  // NOLINT // FIXME: ptr is not used

        // resize policy diff from base class
        {
        BITMAIN_FUNCTION_TRACE("Pnet resize");
#ifdef USE_VPP
        cv::Mat src_channels[3];
        cv::vpp::resize_border_split(base_img, hs, ws, 0, 0, 0, 0, src_channels);

        auto &quan_val = GetRefQuanParams();
        NormalizeToU8(src_channels, quan_val.mean_values, quan_val.input_scales, channels);
#else
        cv::Mat prepared;
        cv::resize(base_img, prepared, cv::Size(ws, hs), 0, 0, cv::INTER_LINEAR);

        auto &quan_val = GetRefQuanParams();
        NormalizeAndSplitToU8(prepared, quan_val.mean_values, quan_val.input_scales, channels);
#endif
        }

        // FIXME: Temporarily added to align result with Python
        // This is caused by OpenCV convertTo, all zeros on bm1880 is set to 1
        for (int i = 0; i < input_tensor.count; i++) {
            if (input_tensor.data[i] == 0) input_tensor.data[i] = 1;
        }
        NetShape shape(1, 3, hs, ws);
        Inference(shape);
        DequantizeOutputTensors(output_tensor);

        // Detect ends here
        /////////////////////////////////////////////////////////////////////
        std::vector<face_info_regression_t> draw;
        for (int j = 0; j < batch_num; j++) {
            // return result
            std::vector<face_info_regression_t> boxes;
            OutputTensor *reg = &output_tensor[m_layer_name[1].c_str()];
            OutputTensor *confidence = &output_tensor[m_layer_name[0].c_str()];
            std::vector<OutputTensor> tens;
            tens.emplace_back(*reg);
            tens.emplace_back(*confidence);

            bmnet_output_info_t output_info;
            GetOutputInfo(&output_info);
            GenerateBoundingBox(output_info, (*confidence), (*reg), scale, m_confidence, ws, hs, j,
                                &boxes);
            std::vector<face_info_regression_t> bboxes_nms;
            NonMaximumSuppression(boxes, bboxes_nms, 0.5, 'u');
            if (!is_nchw) {
                for (auto &reg_face_info : bboxes_nms) {
                    auto &box(reg_face_info.bbox);
                    std::swap(box.x1, box.y1);
                    std::swap(box.x2, box.y2);
                }
            }
            total_bbox.insert(total_bbox.end(), bboxes_nms.begin(), bboxes_nms.end());
            draw.insert(draw.end(), bboxes_nms.begin(), bboxes_nms.end());
        }

        std::string ss = "pnet_output_" + std::to_string(ws) + ".jpg";
        BITMAIN_DRAWFDINFO_SAVE(ss.c_str(), *dst, draw);
    }
    std::vector<face_info_regression_t> factor_nms;
    NonMaximumSuppression(total_bbox, factor_nms, 0.7, 'u');
    BoxRegress(factor_nms, 1, face_rects);
    BITMAIN_DRAWFDRECT_SAVE("pnet_output.jpg", *dst, *face_rects);
    LOGD << "PNet end";
}

void PNet::GenerateBoundingBox(const bmnet_output_info_t &output_info,
                               const OutputTensor &confidence, const OutputTensor &reg,
                               const float scale, const float thresh, const int image_width,
                               const int image_height, const int image_id,
                               std::vector<face_info_regression_t> *candidate) {
    int stride = 2;
    int cellSize = 12;

    int curr_feature_map_w_ = ceil((image_width - cellSize) * 1.0 / stride) + 1;
    int curr_feature_map_h_ = ceil((image_height - cellSize) * 1.0 / stride) + 1;

    int regOffset = curr_feature_map_w_ * curr_feature_map_h_;

    // the first count numbers are confidence of face
    int count = confidence.count / 2;

    float *conf_ptr, *reg_ptr;
    if (is_nchw) {
        conf_ptr = confidence.data;
        reg_ptr = reg.data;
    } else {
        // TODO: FIXME: Need a better way to do transpose.
        cv::Mat a(cv::Size(confidence.shape.w, confidence.shape.h), CV_32FC2, confidence.data);
        cv::Mat b(cv::Size(reg.shape.w, reg.shape.h), CV_32FC4, reg.data);
        cv::Mat aa = a.clone();
        cv::Mat bb = b.clone();
        cv::transpose(aa, aa);
        cv::transpose(bb, bb);
        aa.copyTo(a);
        bb.copyTo(b);

        conf_ptr = confidence.data;
        reg_ptr = reg.data;
    }

    std::unique_ptr<float[]> confidence_data(
        new float[confidence.size * sizeof(float) / sizeof(int8_t)]);
    SoftMax(conf_ptr, confidence_data.get(), confidence.shape.n, confidence.shape.c,
            confidence.shape.h * confidence.shape.w, m_fast_exp);

    // Get the head of the pointer buffer
    float *confidence_data_ptr = confidence_data.get();
    confidence_data_ptr += count;
    for (int i = 0; i < count; i++) {
        // If larger than threshold confidence, get the bbox and regression
        // info
        if (*(confidence_data_ptr + i) >= thresh) {
            int y = i / curr_feature_map_w_;
            int x = i - curr_feature_map_w_ * y;

            float xTop = static_cast<int>((x * stride + 1) / scale);
            float yTop = static_cast<int>((y * stride + 1) / scale);
            float xBot = static_cast<int>((x * stride + cellSize - 1 + 1) / scale);
            float yBot = static_cast<int>((y * stride + cellSize - 1 + 1) / scale);
            face_detect_rect_t faceRect;
            faceRect.x1 = xTop;
            faceRect.y1 = yTop;
            faceRect.x2 = xBot;
            faceRect.y2 = yBot;
            faceRect.id = image_id;
            faceRect.score = *(confidence_data_ptr + i);
            std::array<float, 4> regression;
            regression[0] = reg_ptr[i + 0 * regOffset];
            regression[1] = reg_ptr[i + 1 * regOffset];
            regression[2] = reg_ptr[i + 2 * regOffset];
            regression[3] = reg_ptr[i + 3 * regOffset];
            candidate->emplace_back(face_info_regression_t(faceRect, regression));
        }
    }
}

}  // namespace vision
}  // namespace qnn
