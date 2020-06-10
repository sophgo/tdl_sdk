// Copyright 2018 Bitmain Inc.
// License
// Author Tim Ho <tim.ho@bitmain.com>
/**
 * @file face_attribute.hpp
 * @ingroup NETWORKS_ATTR
 */
#pragma once

#if defined FACE_ATTRIBUTE_112x112
#    define FACE_ATTRIBUTE_INPUT_HEIGHT (112)
#    define FACE_ATTRIBUTE_INPUT_WIDTH (112)
#    define FACE_ATTRIBUTE_INPUT_CHANNELS (3)

#elif defined FACE_ATTRIBUTE_224x224
#    define FACE_ATTRIBUTE_INPUT_HEIGHT (224)
#    define FACE_ATTRIBUTE_INPUT_WIDTH (224)
#    define FACE_ATTRIBUTE_INPUT_CHANNELS (3)
#endif

#define FACE_ATTRIBUTE_MEAN (-127.5)
#define FACE_ATTRIBUTE_INPUT_THRESHOLD (0.99658036232)

#include "core/image_net.hpp"
#include "face.hpp"
#include "utils/math_utils.hpp"
#include <string>
#include <vector>

namespace qnn {
namespace vision {

class FaceAttribute : public ImageNet, public FaceRecognizer {
    public:
    FaceAttribute(const std::string &model_path, QNNCtx *qnn_ctx = nullptr);
    virtual ~FaceAttribute() override;

    /**
     * Input is a single or list of RGB images, each with shape
     * {FACE_ATTRIBUTE_INPUT_HEIGHT, FACE_ATTRIBUTE_INPUT_WIDTH,
     * FACE_ATTRIBUTE_INPUT_CHANNELS}
     *
     * Each face image need to be aligned using MTCNN's 5 landmarks,
     * follow the location used in insightface's paper.
     *
     * Results will be returned in list of FaceAttributeInfo, same order as
     * input order.
     */
    void Detect(const cv::Mat &image, FaceAttributeInfo &result) override;
    void Detect(const std::vector<cv::Mat> &images,
                std::vector<FaceAttributeInfo> &results) override;

    void SetRequiredOutputs(bool is_extract_face_feature, bool is_extract_emotion,
                            bool is_extract_age, bool is_extract_gender, bool is_extract_race);

    private:
    void DecideOutputs();

    /**
     * Get attribute predictions from dequantized raw output buffer
     */
    void ExtractAttrInfo(OutTensors &out_tensors, std::vector<FaceAttributeInfo> &results,
                         size_t offset);

    FaceFeature ExtractFaceFeature(float *out);
    std::pair<float, AgeFeature> ExtractAge(float *out);

    template <typename U, typename V>
    std::pair<U, V> ExtractFeatures(float *out, int size) {
        qnn::math::SoftMaxForBuffer(out, out, size);
        size_t max_idx = 0;
        float max_val = -1e3;
        for (int i = 0; i < size; i++) {
            LOGD << out[i];
            if (out[i] > max_val) {
                max_val = out[i];
                max_idx = i;
            }
        }
        LOGD << std::endl;

        V features;
        memcpy(features.features.data(), out, sizeof(float) * size);
        return std::make_pair(U(max_idx + 1), features);
    }

    private:
    bool _is_extract_face_feature;
    bool _is_extract_emotion;
    bool _is_extract_age;
    bool _is_extract_gender;
    bool _is_extract_race;

    size_t single_batch_output_buffer_size = 0;
    size_t face_feature_size = 0;
    size_t emotion_prob_size = 0;
    size_t age_prob_size = 0;
    size_t gender_prob_size = 0;
    size_t race_prob_size = 0;
};

}  // namespace vision
}  // namespace qnn
