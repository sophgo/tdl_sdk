// Copyright 2018 Bitmain Inc.
// License
// Author
#include "anti_facespoofing/afs_classify_hsv_ycbcr.hpp"
#include "opencv2/imgproc.hpp"
#include "utils/face_debug.hpp"
#include "utils/function_tracer.h"
#include "utils/image_utils.hpp"
#include "utils/math_utils.hpp"

#define ANTIFACESPOOFING_CLASSIFY_HSV_B_MEAN -127.5
#define ANTIFACESPOOFING_CLASSIFY_HSV_G_MEAN -127.5
#define ANTIFACESPOOFING_CLASSIFY_HSV_R_MEAN -127.5
#define ANTIFACESPOOFING_CLASSIFY_HSV_SZ 224
#define ANTIFACESPOOFING_CLASSIFY_HSV_C 9
namespace qnn {
namespace vision {

using cv::Mat;
using std::string;
using std::vector;

static const std::vector<NetShape> kPossibleInputShapes = {
    NetShape(1, ANTIFACESPOOFING_CLASSIFY_HSV_C, ANTIFACESPOOFING_CLASSIFY_HSV_SZ, ANTIFACESPOOFING_CLASSIFY_HSV_SZ)};

AntiFaceSpoofingClassifyHSVYCbCr::AntiFaceSpoofingClassifyHSVYCbCr(const string &model_path, QNNCtx *qnn_ctx)
    : ImageNet(model_path, kPossibleInputShapes, ANTIFACESPOOFING_CLASSIFY_HSV_C,
               ANTIFACESPOOFING_CLASSIFY_HSV_SZ, ANTIFACESPOOFING_CLASSIFY_HSV_SZ, false, cv::INTER_LINEAR,
               qnn_ctx) {
    Init(0);
}

AntiFaceSpoofingClassifyHSVYCbCr::AntiFaceSpoofingClassifyHSVYCbCr(const std::string &model_path, float threshold,
                                                   QNNCtx *qnn_ctx)
    : ImageNet(model_path, kPossibleInputShapes, ANTIFACESPOOFING_CLASSIFY_HSV_C,
               ANTIFACESPOOFING_CLASSIFY_HSV_SZ, ANTIFACESPOOFING_CLASSIFY_HSV_SZ, false, cv::INTER_LINEAR,
               qnn_ctx) {
    Init(threshold);
}

void AntiFaceSpoofingClassifyHSVYCbCr::Init(float threshold) {
    float model_threshold = GetInPutThreshold();
    printf("threshold=%f\n", GetInPutThreshold());
    SetQuanParams(
        {std::vector<float>{ANTIFACESPOOFING_CLASSIFY_HSV_B_MEAN, ANTIFACESPOOFING_CLASSIFY_HSV_G_MEAN,
                            ANTIFACESPOOFING_CLASSIFY_HSV_R_MEAN},
         std::vector<float>{1 / model_threshold}});
    EnableDequantize(true);
    m_cvt_code.push_back(cv::COLOR_BGR2RGB);
    m_cvt_code.push_back(cv::COLOR_BGR2HSV);
    m_cvt_code.push_back(cv::COLOR_BGR2YCrCb);
    m_realface_threshold = threshold;
}

void AntiFaceSpoofingClassifyHSVYCbCr::SetThreshold(float threshold) { m_realface_threshold = threshold; }

const uint AntiFaceSpoofingClassifyHSVYCbCr::GetThreshold() { return m_realface_threshold; }

const float AntiFaceSpoofingClassifyHSVYCbCr::GetConfidence() { return m_currconfidence; }

void AntiFaceSpoofingClassifyHSVYCbCr::Preprocess(const cv::Mat &crop_img, std::vector<cv::Mat> &images) {
    for (size_t i = 0; i < m_cvt_code.size(); i++) {
        cv::Mat cvt_img;
        cv::cvtColor(crop_img, cvt_img, m_cvt_code[i]);
        images.emplace_back(cvt_img);
    }
}

void AntiFaceSpoofingClassifyHSVYCbCr::DetectInference(const std::vector<cv::Mat> &images, bool &is_real) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    assert(!images.empty());

    vector<Mat> prepared = images;
    assert(images.size() == prepared.size());

    int remainings = images.size() / m_cvt_code.size();
    int idx = 0;
    while (true) {
        int batch = GetNumberPerBatch(remainings);
        if (batch == 0) {
            break;
        }
        assert(batch > 0 && remainings >= batch);
        remainings -= batch;
        auto tensor_pair = SelectTensor(NetShape(batch, ANTIFACESPOOFING_CLASSIFY_HSV_C, net_height, net_width));
        InputTensor &in_tensor(*tensor_pair.first);
        OutTensors &out_tensors(*tensor_pair.second);

        LOGD << "Input tensor shape " << in_tensor.shape.n << " " << in_tensor.shape.c << " "
             << in_tensor.shape.h << " " << in_tensor.shape.w;

        auto &quan_parms = GetRefQuanParams();
        char *ptr = in_tensor.data;
        for (int i = 0; i < in_tensor.shape.c / 3; ++i) {
            Mat &img(prepared[i]);
#if DEBUG_NETWORK_IMAGENET
#    ifndef __ARM_ARCH
            SaveMat2Txt("prepared.txt", prepared[i]);
#    else
            SaveMat2Txt("/mnt/prepared.txt", prepared[i]);
#    endif
#endif
            vector<Mat> channels;
            ptr = WrapInputLayer(channels, ptr, 3, net_height, net_width);
#if defined(NPU_INT8)
            NormalizeAndSplitToU8(img, quan_parms.mean_values, quan_parms.input_scales, channels);
#elif defined(NPU_FLOAT32)
            AverageAndSplitToF32(img, channels, quan_parms.r_mean, quan_parms.g_mean,
                                 quan_parms.b_mean, net_width, net_height);
#else
            assert(false);
#endif
        }
        NetShape shape(batch, in_tensor.shape.c, net_height, net_width);
        Inference(shape);

        std::vector<OutputTensor> o_ten;
        for (auto &x : out_tensors) {
            o_ten.push_back(x.second);
        }
#if DEBUG_NETWORK_IMAGENET
#    ifndef __ARM_ARCH
        SaveTensor2Txt("tensor_raw.txt", in_tensor, o_ten);
#    else
        SaveTensor2Txt("/mnt/tensor_raw.txt", in_tensor, o_ten);
#    endif
#endif
        DequantizeOutputTensors(out_tensors);
        auto &out_tensor = out_tensors["logits_MatMul"];
        float *data = reinterpret_cast<float *>(out_tensor.data);
        float prob[2];
        qnn::math::SoftMax(data, prob, 1, 2, 1);
        LOGI << "Prob(0, 1) = ( " << prob[0] << ", " << prob[1]
             << "). Threshold = " << m_realface_threshold;
        m_currconfidence = prob[1];
        if (prob[1] > m_realface_threshold) {
            is_real = true;
        }
        std::cout << "Confidence = (" << prob[1]
             << ") Threshold = " << m_realface_threshold
             << " is real = " << is_real << std::endl;

        idx += batch;
    }
}

void AntiFaceSpoofingClassifyHSVYCbCr::Detect(const cv::Mat &crop_img, bool &is_real) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    is_real = false;
    if (crop_img.empty()) {
        return;
    }
    std::vector<cv::Mat> images;
    Preprocess(crop_img, images);
    DetectInference(images, is_real);
}

}  // namespace vision
}  // namespace qnn
