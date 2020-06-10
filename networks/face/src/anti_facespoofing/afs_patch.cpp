// Copyright 2018 Bitmain Inc.
// License
// Author
#include "anti_facespoofing/afs_patch.hpp"
#include "opencv2/imgproc.hpp"
#include "utils/function_tracer.h"

namespace qnn {
namespace vision {

using cv::Mat;
using std::string;
using std::vector;

static const std::vector<NetShape> kPossibleInputShapes = {NetShape(1, 3, 96, 96)};

AntiFaceSpoofingPatch::AntiFaceSpoofingPatch(const string &model_path, QNNCtx *qnn_ctx)
    : ImageNet(model_path, kPossibleInputShapes, 3, 96, 96, true, cv::INTER_LINEAR, qnn_ctx) {
    float threshold = GetInPutThreshold();
    printf("threshold=%f\n", GetInPutThreshold());
    SetQuanParams({std::vector<float>{0, 0, 0}, std::vector<float>{float(128) / threshold}});
    EnableDequantize(false);
    srand(time(NULL));
}

void AntiFaceSpoofingPatch::PrepareImage(const cv::Mat &image, cv::Mat &prepared, float &ratio) {
    prepared = image;
}

void AntiFaceSpoofingPatch::Detect(const cv::Mat &crop_img, bool &is_real) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    if (crop_img.empty()) {
        return;
    }
    is_real = false;
    if (crop_img.cols <= 96 || crop_img.rows <= 96) {
        return;
    }

    std::vector<cv::Mat> images;

    int rand_x_li = crop_img.cols - 96;
    int rand_y_li = crop_img.rows - 96;

    for (int j = 0; j < 5; j++) {
        int &&rand_x = rand() % rand_x_li;
        int &&rand_y = rand() % rand_y_li;
        cv::Mat patch = crop_img(cv::Rect(rand_x, rand_y, 96, 96));
        cv::Mat hsv_img;
        cv::cvtColor(patch, hsv_img, cv::COLOR_BGR2HSV);
        images.emplace_back(hsv_img);
    }

    if (images.size() == 0) {
        return;
    }

    size_t counter = 0;
    ImageNet::Detect(images, [&](OutTensors &out, vector<float> &ratios, int start, int end) {
        assert(start >= 0 && start < end && end <= int(images.size()));
        // float *data = (float *)out.at(0).data;
        float *data = out["score"].data;
        assert(out["score"].count == 2);
        std::cout << data[0] << " " << data[1] << std::endl;
        if (data[0] < data[1]) counter++;
    });
    // std::cout << counter << std::endl;
    if (counter > images.size() / 2) is_real = true;
    m_counter_result = counter;
}

const size_t AntiFaceSpoofingPatch::GetCounterVal() { return m_counter_result; }

}  // namespace vision
}  // namespace qnn
