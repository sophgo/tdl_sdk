// Copyright 2018 Bitmain Inc.
// License
// Author
#include "anti_facespoofing/afs_classify.hpp"
#include "opencv2/imgproc.hpp"
#include "utils/face_debug.hpp"
#include "utils/function_tracer.h"
#include "utils/image_utils.hpp"
#include "utils/math_utils.hpp"

#define ANTIFACESPOOFINGCLASSIFY_B_MEAN -127.5
#define ANTIFACESPOOFINGCLASSIFY_G_MEAN -127.5
#define ANTIFACESPOOFINGCLASSIFY_R_MEAN -127.5
#define ANTIFACESPOOFINGCLASSIFY_SZ 224
#define ANTIFACESPOOFINGCLASSIFY_C 3
namespace qnn {
namespace vision {

using cv::Mat;
using std::string;
using std::vector;

static const std::vector<NetShape> kPossibleInputShapes = {
    NetShape(1, ANTIFACESPOOFINGCLASSIFY_C, ANTIFACESPOOFINGCLASSIFY_SZ, ANTIFACESPOOFINGCLASSIFY_SZ)};

AntiFaceSpoofingClassify::AntiFaceSpoofingClassify(const string &model_path, QNNCtx *qnn_ctx)
    : ImageNet(model_path, kPossibleInputShapes, ANTIFACESPOOFINGCLASSIFY_C,
               ANTIFACESPOOFINGCLASSIFY_SZ, ANTIFACESPOOFINGCLASSIFY_SZ, false, cv::INTER_LINEAR,
               qnn_ctx) {
    Init(0);
}

AntiFaceSpoofingClassify::AntiFaceSpoofingClassify(const std::string &model_path, float threshold,
                                                   QNNCtx *qnn_ctx)
    : ImageNet(model_path, kPossibleInputShapes, ANTIFACESPOOFINGCLASSIFY_C,
               ANTIFACESPOOFINGCLASSIFY_SZ, ANTIFACESPOOFINGCLASSIFY_SZ, false, cv::INTER_LINEAR,
               qnn_ctx) {
    Init(threshold);
}

void AntiFaceSpoofingClassify::Init(float threshold) {
    float model_threshold = GetInPutThreshold();
    printf("threshold=%f\n", GetInPutThreshold());
    SetQuanParams(
        {std::vector<float>{ANTIFACESPOOFINGCLASSIFY_B_MEAN, ANTIFACESPOOFINGCLASSIFY_G_MEAN,
                            ANTIFACESPOOFINGCLASSIFY_R_MEAN},
         std::vector<float>{1 / model_threshold}});
    EnableDequantize(true);
    m_cvt_code.push_back(cv::COLOR_BGR2RGB);
    m_realface_threshold = threshold;
}

void AntiFaceSpoofingClassify::SetThreshold(float threshold) { m_realface_threshold = threshold; }

const uint AntiFaceSpoofingClassify::GetThreshold() { return m_realface_threshold; }

const float AntiFaceSpoofingClassify::GetConfidence() { return m_currconfidence; }

void AntiFaceSpoofingClassify::Preprocess(const cv::Mat &crop_img, std::vector<cv::Mat> &images) {
    for (size_t i = 0; i < m_cvt_code.size(); i++) {
        cv::Mat cvt_img;
        cv::cvtColor(crop_img, cvt_img, m_cvt_code[i]);
        images.emplace_back(cvt_img);
    }
}

void AntiFaceSpoofingClassify::Detect(const cv::Mat &crop_img, bool &is_real) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    is_real = false;
    if (crop_img.empty()) {
        return;
    }
    std::vector<cv::Mat> images;
    Preprocess(crop_img, images);
    ImageNet::Detect(images, [&](OutTensors &out, vector<float> &ratios, int start, int end) {
        assert(start >= 0 && start < end && end <= int(images.size()));
        // DequantizeOutputTensors(out_tensors);
        auto &out_tensor = out["logits_MatMul"];
        float *data = reinterpret_cast<float *>(out_tensor.data);
        float prob[2];
        qnn::math::SoftMax(data, prob, 1, 2, 1);
        LOGI << "Prob(0, 1) = ( " << prob[0] << ", " << prob[1]
             << "). Threshold = " << m_realface_threshold;
        m_currconfidence = prob[1];
        if (prob[1] > m_realface_threshold) {
            is_real = true;
        }
    });
}

}  // namespace vision
}  // namespace qnn
