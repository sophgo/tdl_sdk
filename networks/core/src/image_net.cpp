// Copyright 2018 Bitmain Inc.
// License
// Author
#include "image_net.hpp"
#include "utils/function_tracer.h"
#include "utils/image_utils.hpp"
#include <cassert>
#if DEBUG_NETWORK_IMAGENET
#    include "utils/face_debug.hpp"
#endif

namespace qnn {
namespace vision {

using cv::Mat;
using std::string;
using std::vector;

ImageNet::ImageNet(const string& model_path, const vector<NetShape>& supported_input_shapes,
                   int channel, int width, int height, bool preserve_img_ratio, int resize_policy,
                   QNNCtx* qnn_ctx)
    : BaseNet(model_path, supported_input_shapes, qnn_ctx),
      net_channel(channel),
      net_width(width),
      net_height(height),
      preserve_ratio(preserve_img_ratio),
      m_resize_policy(resize_policy) {
    m_ibuf.push_back({"resize"});
    m_ibuf.push_back({"copymakeborder"});
}

void ImageNet::Detect(const vector<Mat>& images, ParseFunc parse) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    assert(!images.empty());

    vector<float> ratios;

    int remainings = images.size();
    int idx = 0;
    while (true) {
        int batch = GetNumberPerBatch(remainings);
        if (batch == 0) {
            break;
        }
        assert(batch > 0 && remainings >= batch);
        auto tensor_pair = SelectTensor(NetShape(batch, net_channel, net_height, net_width));
        InputTensor& in_tensor(*tensor_pair.first);
        OutTensors& out_tensors(*tensor_pair.second);

        LOGD << "Input tensor shape " << in_tensor.shape.n << " " << in_tensor.shape.c << " "
             << in_tensor.shape.h << " " << in_tensor.shape.w;

        /* Here I use a quick example to explain the code here. Assume the Netshape is 1, 6, 32, 32,
         * and the image array is idx: 0 (channel 3), idx: 1 (channel 3)....
         * When quantizing the images, it will automatically group idx 0 & 1 into one dataset
         *
         *        1 (tensor n) * 6 (tensor c) = 3 (idx 0) + 3 (idx 1)
         *
         * If the image array is idx: 0 (channel 6), idx: 1 (channel 6)..., it will not group the
         * images into datasets
         * total_img_layer is used to count how many layers is quantized, and (i - idx) gives the
         * info of how many images are used.
         * On ARM, the channel of each image must between 1 to 4 because of the limitation of the
         * NEON instructions. For Netshapes that uses channels > 4, you'll have to overwrite
         * PrepareImages for performance.
         */
        char* ptr = in_tensor.data;
        int total_img_layer = 0;
        const int presume_img_used = in_tensor.shape.n * in_tensor.shape.c;
        int i = idx;
        for (; total_img_layer < presume_img_used; ++i) {
            Mat out_img;
            float ratio;
            PrepareImage(images[i], out_img, ratio);
#if DEBUG_NETWORK_IMAGENET
#    ifndef __ARM_ARCH
            SaveMat2Txt("prepared.txt", out_img);
#    else
            SaveMat2Txt("/mnt/prepared.txt", out_img);
#    endif
#endif
            ratios.emplace_back(ratio);
            vector<Mat> channels;
            ptr = WrapInputLayer(channels, ptr, in_tensor.shape.c, in_tensor.shape.h,
                                 in_tensor.shape.w);
            PreProcessImage(out_img, channels);
            total_img_layer += out_img.channels();
        }
        if (total_img_layer != presume_img_used) {
            LOGE << "Error! image size is not match to the Netshape requirements.";
            exit(-1);
        }
        NetShape shape(in_tensor.shape.n, in_tensor.shape.c, in_tensor.shape.h, in_tensor.shape.w);
        Inference(shape);

        std::vector<OutputTensor> o_ten;
        for (auto& x : out_tensors) {
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
        parse(out_tensors, ratios, idx, idx + batch);
        int real_batch = i - idx;
        remainings -= real_batch;
        idx = i;
    }
}

void ImageNet::PreProcessImage(Mat& image, vector<Mat>& channels) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
#if defined(NPU_INT8)
    NormalizeAndSplitToU8(image, quan_parms.mean_values, quan_parms.input_scales, channels);
#elif defined(NPU_FLOAT32)
    AverageAndSplitToF32(image, channels, R_MEAN, G_MEAN, B_MEAN, net_width, net_height);
#else
    assert(false);
#endif
}

void ImageNet::PrepareImage(const cv::Mat& image, cv::Mat& prepared, float& ratio) {
    ratio = ResizeImage(image, prepared, net_width, net_height, m_ibuf, m_resize_policy,
                        preserve_ratio);
}

float ImageNet::ResizeImage(const Mat& src, Mat& dst, int width, int height,
                            std::vector<MatBuffer>& buf, int resize_policy, bool preserve_ratio) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);

    if (src.rows == height && src.cols == width) {
        dst = src;
        return 1.0;
    }
    float ratio = std::max(1.0 * src.rows / height, 1.0 * src.cols / width);
    cv::Size dstSize;
    if (preserve_ratio) {
        dstSize.width = src.cols / ratio;
        dstSize.height = src.rows / ratio;
    } else {
        dstSize.width = width;
        dstSize.height = height;
    }
    int pad_bottom = height - dstSize.height;
    int pad_right = width - dstSize.width;
    // Using MatBuffer in resize and copyMakeBorder avoids cv::Mat::create to acquire new space from
    // ion

#ifdef USE_VPP
    buf[1].img = cv::vpp::crop_resize_border(const_cast<Mat&>(src), dstSize.height, dstSize.width,
                                             0, 0, src.cols, src.rows,
                                             0, pad_bottom, 0, pad_right);

    dst = buf[1].img;
#else
    bool isPadding = preserve_ratio && (pad_bottom || pad_right);
    cv::resize(src, buf[0].img, dstSize, 0, 0, resize_policy);

    if (isPadding) {
        cv::copyMakeBorder(buf[0].img, buf[1].img, 0, pad_bottom, 0, pad_right, cv::BORDER_CONSTANT,
                           cv::Scalar(0));
        dst = buf[1].img;
    } else {
        dst = buf[0].img;
    }
#endif

    LOGD << "resize from " << src.cols << ", " << src.rows << " to " << width << ", " << height;
    LOGD << "pad from " << src.cols / ratio << ", " << src.rows / ratio << " to " << width << ", "
         << height;

    return ratio;
}

float ImageNet::ResizeImageCenter(const Mat& src, Mat& dst, int width, int height,
                                  std::vector<MatBuffer>& buf, int resize_policy,
                                  bool preserve_ratio, cv::Scalar scalar) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);

    float ratio;

    if (src.rows == width && src.cols == height) {
        dst = src;
        ratio = 1.0;
    } else {
        ratio = std::max(1.0 * src.rows / width, 1.0 * src.cols / height);
        cv::Size dstSize;
        if (preserve_ratio) {
            dstSize.width = src.cols / ratio;
            dstSize.height = src.rows / ratio;
        } else {
            dstSize.width = width;
            dstSize.height = height;
        }

        int pad_left = int((width - dstSize.width) / 2);
        int pad_top = int((height - dstSize.height) / 2);
        int pad_right = width - dstSize.width - pad_left;
        int pad_bottom = height - dstSize.height - pad_top;

#ifdef USE_VPP
        dst = cv::vpp::crop_resize_border(const_cast<Mat&>(src), dstSize.height, dstSize.width,
                                          0, 0, src.cols, src.rows,
                                          pad_top, pad_bottom, pad_left, pad_right);
#else
        cv::resize(src, buf[0].img, dstSize, 0, 0, resize_policy);

        bool bPadding = pad_top || pad_bottom || pad_right || pad_left;

        if (bPadding) {
            cv::copyMakeBorder(buf[0].img, buf[1].img, pad_top, pad_bottom, pad_left, pad_right,
                               cv::BORDER_CONSTANT, scalar);
            dst = buf[1].img;
        } else {
            dst = buf[0].img;
        }
#endif

        LOGD << "Resize from " << src.cols << ", " << src.rows << " to " << dstSize.width << ", "
             << dstSize.height;
        LOGD << "Padding " << pad_bottom << " " << pad_top << " " << pad_left << " " << pad_right;
    }

    return ratio;
}

}  // namespace vision
}  // namespace qnn
