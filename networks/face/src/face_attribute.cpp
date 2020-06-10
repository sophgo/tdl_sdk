// Copyright 2018 Bitmain Inc.
// License
// Author Tim Ho <tim.ho@bitmain.com>

#include "face_attribute.hpp"
#include "net_face.hpp"
#include "utils/face_attribute_utils.hpp"
#include "utils/function_tracer.h"
#include "utils/math_utils.hpp"
#include "utils/string_utils.hpp"
#include <exception>
#include <utility>
#define FACE_ATTRIBUTE_INPUT_CHANNEL 3

namespace qnn {
namespace vision {

using cv::Mat;
using qnn::math::SoftMaxForBuffer;
using qnn::utils::FindString;
using std::string;
using std::vector;

static const vector<NetShape> kPossibleInputShapes{
    NetShape(1, 3, 112, 112), NetShape(2, 3, 112, 112), NetShape(4, 3, 112, 112),
    NetShape(8, 3, 112, 112), NetShape(16, 3, 112, 112)};

FaceAttribute::FaceAttribute(const string &model_path, QNNCtx *qnn_ctx)
    : ImageNet(model_path, kPossibleInputShapes, FACE_ATTRIBUTE_INPUT_CHANNEL,
               FACE_ATTRIBUTE_INPUT_WIDTH, FACE_ATTRIBUTE_INPUT_HEIGHT, true, cv::INTER_NEAREST,
               qnn_ctx) {
    SetQuanParams(
        {std::vector<float>{FACE_ATTRIBUTE_MEAN, FACE_ATTRIBUTE_MEAN, FACE_ATTRIBUTE_MEAN},
         std::vector<float>{FACE_ATTRIBUTE_INPUT_THRESHOLD}});
    DecideOutputs();
}

FaceAttribute::~FaceAttribute() {}

void FaceAttribute::DecideOutputs() {
    auto in_out_tensor = SelectTensor(kPossibleInputShapes[0]);
    OutTensors *out_tensors = in_out_tensor.second;
    size_t output_size = out_tensors->size();
    if (output_size == 1) {
        SetRequiredOutputs(true, false, false, false, false);
    } else {
        SetRequiredOutputs(true, true, true, true, true);
    }

    for (auto it = out_tensors->begin(); it != out_tensors->end(); it++) {
        OutputTensor &tensor = it->second;
        single_batch_output_buffer_size += tensor.count;

        if ((FindString(tensor.name, "feature") || FindString(tensor.name, "fc1")) ||
            (FindString(tensor.name, "dense"))) {
            face_feature_size = tensor.count;
        } else if (FindString(tensor.name, "age")) {
            age_prob_size = tensor.count;
        } else if (FindString(tensor.name, "emotion")) {
            emotion_prob_size = tensor.count;
        } else if (FindString(tensor.name, "race")) {
            race_prob_size = tensor.count;
        } else if (FindString(tensor.name, "gender")) {
            gender_prob_size = tensor.count;
        } else {
            LOGD << "Attribute " << tensor.name << " is not supported!";
        }
    }
}

void FaceAttribute::SetRequiredOutputs(bool is_extract_face_feature, bool is_extract_emotion,
                                       bool is_extract_age, bool is_extract_gender,
                                       bool is_extract_race) {
    _is_extract_face_feature = is_extract_face_feature;
    _is_extract_emotion = is_extract_emotion;
    _is_extract_age = is_extract_age;
    _is_extract_gender = is_extract_gender;
    _is_extract_race = is_extract_race;
}

std::pair<float, AgeFeature> FaceAttribute::ExtractAge(float *out) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    LOGD << "Extract age" << std::endl;
    float expect_age = 0;
    if (age_prob_size == 1) {
        expect_age = out[0];
    } else if (age_prob_size == 101) {
        SoftMaxForBuffer(out, out, age_prob_size);
        for (size_t i = 0; i < 101; i++) {
            expect_age += (i * out[i]);
        }
    } else {
        expect_age = -1.0;
    }

    AgeFeature features;
    memcpy(features.features.data(), out, sizeof(float) * age_prob_size);
    return std::make_pair(expect_age, features);
}

FaceFeature FaceAttribute::ExtractFaceFeature(float *out) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    LOGD << "Extract face feature" << std::endl;
    FaceFeature features;
    memcpy(features.features.data(), out, sizeof(float) * face_feature_size);
    return features;
}

void FaceAttribute::ExtractAttrInfo(OutTensors &out_tensors,
                                    std::vector<FaceAttributeInfo> &results, size_t offset) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);

    FaceAttributeInfo result;
    for (auto it = out_tensors.begin(); it != out_tensors.end(); it++) {
        OutputTensor &tensor = it->second;

        if (_is_extract_face_feature &&
            (FindString(tensor.name, "feature") || FindString(tensor.name, "fc1") ||
             FindString(tensor.name, "dense"))) {
            if (_is_extract_face_feature) {
                auto t = ExtractFaceFeature(tensor.data + offset * face_feature_size);
                result.face_feature = std::move(t);
            }
        } else if (_is_extract_age && FindString(tensor.name, "age")) {
            auto t = ExtractAge(tensor.data + offset * age_prob_size);
            result.age = t.first;
            result.age_prob = std::move(t.second);
        } else if (_is_extract_emotion && FindString(tensor.name, "emotion")) {
            auto t = ExtractFeatures<FaceEmotion, EmotionFeature>(
                tensor.data + offset * emotion_prob_size, emotion_prob_size);
            result.emotion = t.first;
            result.emotion_prob = std::move(t.second);
        } else if (_is_extract_race && FindString(tensor.name, "race")) {
            auto t = ExtractFeatures<FaceRace, RaceFeature>(tensor.data + offset * race_prob_size,
                                                            race_prob_size);
            result.race = t.first;
            result.race_prob = std::move(t.second);
        } else if (_is_extract_gender && FindString(tensor.name, "gender")) {
            auto t = ExtractFeatures<FaceGender, GenderFeature>(
                tensor.data + offset * gender_prob_size, gender_prob_size);
            result.gender = t.first;
            result.gender_prob = std::move(t.second);
        } else {
            LOGD << "Tensor " << tensor.name << " has no rule to extract";
        }
    }
    results.emplace_back(std::move(result));

    LOGD << "Emotion   |" << GetEmotionString(result.emotion) << std::endl;
    LOGD << "Age       |" << result.age << std::endl;
    LOGD << "Gender    |" << GetGenderString(result.gender) << std::endl;
    LOGD << "Race      |" << GetRaceString(result.race) << std::endl;
}

void FaceAttribute::Detect(const Mat &image, FaceAttributeInfo &result) {
    if (image.empty()) {
        assert(false);
        return;
    }
    vector<Mat> images = {image};
    vector<FaceAttributeInfo> results;
    Detect(images, results);
    result = std::move(results[0]);
}

void FaceAttribute::Detect(const vector<Mat> &images, vector<FaceAttributeInfo> &results) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    if (images.empty()) {
        assert(false);
        return;
    }

    results.clear();
    results.reserve(images.size());
    ImageNet::Detect(images, [&](OutTensors &out, vector<float> &ratios, int start, int end) {
        assert(start >= 0 && start < end && end <= (int)images.size());
        LOGD << "start: " << start << " end: " << end;
        for (int i = start, offset = 0; i < end; ++i, ++offset) {
            ExtractAttrInfo(out, results, offset);
            LOGD << "Extract " << offset + 1 << " image in current batch" << std::endl;
        }
    });
}

}  // namespace vision
}  // namespace qnn
