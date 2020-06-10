// Copyright 2018 Bitmain Inc.
// License
// Author
#include "anti_facespoofing/afs_depth.hpp"
#include "opencv2/imgproc.hpp"
#include "utils/face_debug.hpp"
#include "utils/function_tracer.h"
#include "utils/image_utils.hpp"
#include "utils/math_utils.hpp"

#define ANTIFACESPOOFINGDEPTH_B_MEAN -127.5
#define ANTIFACESPOOFINGDEPTH_G_MEAN -127.5
#define ANTIFACESPOOFINGDEPTH_R_MEAN -127.5
#define ANTIFACESPOOFINGDEPTH_SZ 224
#define ANTIFACESPOOFINGDEPTH_C 6
namespace qnn {
namespace vision {

using cv::Mat;
using std::string;
using std::vector;

static const std::vector<NetShape> kPossibleInputShapes = {
    NetShape(1, 6, ANTIFACESPOOFINGDEPTH_SZ, ANTIFACESPOOFINGDEPTH_SZ)};

AntiFaceSpoofingDepth::AntiFaceSpoofingDepth(const string &model_path, QNNCtx *qnn_ctx)
    : ImageNet(model_path, kPossibleInputShapes, ANTIFACESPOOFINGDEPTH_C, ANTIFACESPOOFINGDEPTH_SZ,
               ANTIFACESPOOFINGDEPTH_SZ, true, cv::INTER_LINEAR, qnn_ctx) {
    Init(0);
}

AntiFaceSpoofingDepth::AntiFaceSpoofingDepth(const std::string &model_path, size_t threshold,
                                             QNNCtx *qnn_ctx)
    : ImageNet(model_path, kPossibleInputShapes, ANTIFACESPOOFINGDEPTH_C, ANTIFACESPOOFINGDEPTH_SZ,
               ANTIFACESPOOFINGDEPTH_SZ, true, cv::INTER_LINEAR, qnn_ctx) {
    Init(threshold);
}

void AntiFaceSpoofingDepth::Init(size_t threshold) {
    float model_threshold = GetInPutThreshold();
    printf("threshold=%f\n", GetInPutThreshold());
    SetQuanParams({std::vector<float>{ANTIFACESPOOFINGDEPTH_B_MEAN, ANTIFACESPOOFINGDEPTH_G_MEAN,
                                      ANTIFACESPOOFINGDEPTH_R_MEAN},
                   std::vector<float>{1 / model_threshold}});
    EnableDequantize(true);
    m_cvt_code.push_back(cv::COLOR_BGR2RGB);
    m_cvt_code.push_back(cv::COLOR_BGR2HSV);
    m_realface_threshold = threshold;
}

void AntiFaceSpoofingDepth::SetThreshold(size_t threshold) { m_realface_threshold = threshold; }

const uint AntiFaceSpoofingDepth::GetThreshold() { return m_realface_threshold; }

const size_t AntiFaceSpoofingDepth::GetCounterVal() { return m_counter_result; }

void AntiFaceSpoofingDepth::Preprocess(const cv::Mat &crop_img, std::vector<cv::Mat> &images) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    int pad_switch = 0;
    if (crop_img.rows > crop_img.cols) {
        pad_switch = 1;
    } else if (crop_img.rows < crop_img.cols) {
        pad_switch = 2;
    }

    uint pad_length = std::abs(crop_img.rows - crop_img.cols);
    uint pad_left = pad_length / 2;
    uint &&pad_right = pad_length - pad_left;

    cv::Mat resized;
    switch (pad_switch) {
        case 1: {
            cv::copyMakeBorder(crop_img, resized, 0, 0, pad_left, pad_right, cv::BORDER_CONSTANT,
                               cv::Scalar(0));
        } break;
        case 2: {
            cv::copyMakeBorder(crop_img, resized, pad_left, pad_right, 0, 0, cv::BORDER_CONSTANT,
                               cv::Scalar(0));
        } break;
        default:
            resized = crop_img;
            break;
    }

    cv::resize(resized, resized, cv::Size(ANTIFACESPOOFINGDEPTH_SZ, ANTIFACESPOOFINGDEPTH_SZ), 0, 0,
               cv::INTER_LINEAR);
    for (size_t i = 0; i < m_cvt_code.size(); i++) {
        cv::Mat cvt_img;
        cv::cvtColor(resized, cvt_img, m_cvt_code[i]);
        images.emplace_back(cvt_img);
    }
}

void AntiFaceSpoofingDepth::PrepareImage(const cv::Mat &image, cv::Mat &prepared, float &ratio) {
    prepared = image;
}

cv::Mat AntiFaceSpoofingDepth::GetDepthVisualMat() { return m_visual; }

void AntiFaceSpoofingDepth::DetectInference(const std::vector<cv::Mat> &images, bool &is_real) {
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
        auto tensor_pair = SelectTensor(NetShape(batch, 6, net_height, net_width));
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
        if (m_make_visual) {
            DequantizeOutputTensors(out_tensors);
            auto &out_tensor = out_tensors["FaceDepthNet_DepthMapHead_Conv_Conv2D"];
            m_visual = cv::Mat(cv::Size(out_tensor.shape.w, out_tensor.shape.h), CV_32FC1,
                               out_tensor.data);
        }
        auto &out_tensor = out_tensors["FaceDepthNet_DepthMapHead_Conv_Conv2D"];
        char *data = reinterpret_cast<char *>(out_tensor.q_data);
        size_t counter = 0;
        for (int i = 0; i < out_tensor.count; i++) {
            counter += std::abs((int)data[i]);
        }
        LOGI << "Total: " << out_tensor.count << ". counter value " << counter << " threshold "
             << m_realface_threshold;
        if (counter > m_realface_threshold) {
            is_real = true;
        }
        m_counter_result = counter;
        idx += batch;
    }
}

void AntiFaceSpoofingDepth::Detect(const cv::Mat &crop_img, bool &is_real) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    if (crop_img.empty()) {
        return;
    }
    is_real = false;
    std::vector<cv::Mat> images;
    Preprocess(crop_img, images);
    DetectInference(images, is_real);
}

}  // namespace vision
}  // namespace qnn
