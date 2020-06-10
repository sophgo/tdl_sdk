#include "re_id.hpp"
#include "utils/log_common.h"

#define FEATURES_DIM (128)

namespace qnn {
namespace vision {

using cv::Mat;
using std::string;
using std::vector;

static const vector<NetShape> kPossibleInputShapes = {
    NetShape(1, 3, 128, 64), NetShape(2, 3, 128, 64), NetShape(3, 3, 128, 64),
    NetShape(4, 3, 128, 64)};

ReID::ReID(const string &model_path, QNNCtx *qnn_ctx)
    : ImageNet(model_path, kPossibleInputShapes, 3, 64, 128, true, cv::INTER_LINEAR, qnn_ctx) {
    float threshold = GetInPutThreshold();
    printf("threshold: %f\n", threshold);
    SetQuanParams({std::vector<float>{0, 0, 0}, std::vector<float>{float(128) / threshold}});
}

void ReID::Detect(const vector<Mat> &images, vector<vector<float>> &results) {
    if (images.size() == 0) {
        return;
    }
    results.clear();
    results.reserve(images.size());

    ImageNet::Detect(images, [&](OutTensors &out, vector<float> &ratios, int start, int end) {
        assert(start >= 0 && start < end && end <= (int)images.size());
        float *data = (float *)out["reid-features"].data;
        for (int i = start; i < end; ++i) {
            vector<float> extract_feature(data, data + FEATURES_DIM);
            data += FEATURES_DIM;
            results.emplace_back(extract_feature);
        }
    });
}

}  // namespace vision
}  // namespace qnn