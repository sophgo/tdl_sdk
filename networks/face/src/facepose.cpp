// Copyright 2018 Bitmain Inc.
// License
// Author
#include "facepose.hpp"
#include "opencv2/imgproc.hpp"

// NCHW, N=1
#define EXTRACTOR_DEFAULT_IC 3
#define EXTRACTOR_DEFAULT_IH 128
#define EXTRACTOR_DEFAULT_IW 128
#define EXTRACTOR_DEFAULT_OC 139

#define FACEPOSE_R_MEAN -123.0
#define FACEPOSE_G_MEAN -117.0
#define FACEPOSE_B_MEAN -104.0

namespace qnn {
namespace vision {

using cv::Mat;
using std::string;
using std::vector;

static const std::vector<NetShape> kPossibleInputShapes = {NetShape(1, 3, 128, 128)};

Facepose::Facepose(const string &model_path, QNNCtx *qnn_ctx)
    : ImageNet(model_path, kPossibleInputShapes, EXTRACTOR_DEFAULT_IC, EXTRACTOR_DEFAULT_IW,
               EXTRACTOR_DEFAULT_IH, true, cv::INTER_LINEAR, qnn_ctx) {
    float threshold = 151.073761;  // GetInPutThreshold();
    printf("threshold=%f\n", GetInPutThreshold());
    SetQuanParams({std::vector<float>{FACEPOSE_B_MEAN, FACEPOSE_G_MEAN, FACEPOSE_R_MEAN},
                   std::vector<float>{float(128) / threshold}});
}

void Facepose::Detect(const vector<Mat> &images, vector<vector<float>> &results) {
    if (images.size() == 0) {
        return;
    }
    results.clear();
    results.reserve(images.size());

    ImageNet::Detect(images, [&](OutTensors &out, vector<float> &ratios, int start, int end) {
        assert(start >= 0 && start < end && end <= int(images.size()));
        // float *data = (float *)out.at(0).data;
        float *data = out["lmp_out"].data;
        for (int i = start, offset = 0; i < end; ++i, ++offset) {
            // vector<float> extract_feature(data, data + FEATURES_DIM);
            // data += FEATURES_DIM;
            // results.emplace_back(extract_feature);
            float *begin;
            begin = reinterpret_cast<float *>(data + (EXTRACTOR_DEFAULT_OC * offset));

            vector<float> feature(begin, begin + EXTRACTOR_DEFAULT_OC);
            results.emplace_back(feature);
        }
    });
}

}  // namespace vision
}  // namespace qnn
