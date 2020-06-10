// Copyright 2018 Bitmain Inc.
// License
// Author
/**
 * @file face.hpp
 * @ingroup NETWORKS_FACE
 */
#pragma once

#include <array>
#include <vector>

namespace qnn {
namespace vision {

#define FACE_PTS 5

#define NUM_FACE_FEATURE_DIM (512)
#define NUM_EMOTION_FEATURE_DIM (7)
#define NUM_GENDER_FEATURE_DIM (2)
#define NUM_RACE_FEATURE_DIM (3)
#define NUM_AGE_FEATURE_DIM (101)

typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int id;
} face_detect_rect_t;

typedef struct {
    std::array<float, FACE_PTS> x;
    std::array<float, FACE_PTS> y;
} face_pts_t;

typedef struct {
    face_detect_rect_t bbox;
    face_pts_t face_pts;
} face_info_t;

typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
    int classes;
    float prob;
    std::string label;
} object_detect_rect_t;

class face_info_regression_t {
    public:
    face_info_regression_t() {}
    explicit face_info_regression_t(const face_detect_rect_t &other) : bbox(other) {}
    face_info_regression_t(const face_detect_rect_t &rect, const face_pts_t &pts)
        : bbox(rect), face_pts(pts) {}
    face_info_regression_t(const face_detect_rect_t &rect, const std::array<float, 4> &reg)
        : bbox(rect), regression(reg) {}
    face_info_regression_t(const face_detect_rect_t &rect, const face_pts_t &pts,
                           const std::array<float, 4> &reg)
        : bbox(rect), face_pts(pts), regression(reg) {}

    face_detect_rect_t bbox;
    face_pts_t face_pts;
    std::array<float, 4> regression;
};

enum SSH_TYPE { ORIGIN = 0, TINY };

enum PADDING_POLICY { CENTER = 0, UP_LEFT };

typedef enum {
    EMOTION_UNKNOWN = 0,
    EMOTION_HAPPY,
    EMOTION_SURPRISE,
    EMOTION_FEAR,
    EMOTION_DISGUST,
    EMOTION_SAD,
    EMOTION_ANGER,
    EMOTION_NEUTRAL,
} FaceEmotion;

typedef enum {
    GENDER_UNKNOWN = 0,
    GENDER_MALE,
    GENDER_FEMALE,
} FaceGender;

typedef enum { RACE_UNKNOWN = 0, RACE_CAUCASIAN, RACE_BLACK, RACE_ASIAN } FaceRace;

template <int DIM>
class FeatureVector final {
    public:
    std::vector<float> features;

    FeatureVector() : features(DIM, 0) {}
    FeatureVector(const FeatureVector &other) { *this = other; }

    inline size_t size() { return DIM; }

    FeatureVector &operator=(const FeatureVector &other) {
        features = std::vector<float>(other.features.begin(), other.features.end());
        return *this;
    }

    FeatureVector &operator=(FeatureVector &&other) {
        features.swap(other.features);
        return *this;
    }

    FeatureVector &operator+=(const FeatureVector &other) {
        for (size_t i = 0; i < DIM; i++) {
            features[i] += other.features[i];
        }
        return *this;
    }

    FeatureVector &operator-=(const FeatureVector &other) {
        for (size_t i = 0; i < DIM; i++) {
            features[i] -= other.features[i];
        }
        return *this;
    }

    FeatureVector &operator*=(const FeatureVector &other) {
        for (size_t i = 0; i < DIM; i++) {
            features[i] *= other.features[i];
        }
        return *this;
    }

    FeatureVector &operator/=(const float num) {
        for (size_t i = 0; i < DIM; i++) {
            features[i] /= num;
        }
        return *this;
    }

    inline bool operator==(const FeatureVector &rhs) const {
        for (int i = 0; i < DIM; i++) {
            if (this->features[i] != rhs.features[i]) {
                return false;
            }
        }
        return true;
    }

    inline bool operator!=(const FeatureVector &rhs) const { return !(*this == rhs); }
    inline float &operator[](int x) { return features[x]; }

    int max_idx() const {
        float tmp = features[0];
        int res = 0;
        for (int i = 0; i < DIM; i++) {
            if (features[i] > tmp) {
                tmp = features[i];
                res = i;
            }
        }
        return res;
    }
};

typedef FeatureVector<NUM_FACE_FEATURE_DIM> FaceFeature;
typedef FeatureVector<NUM_EMOTION_FEATURE_DIM> EmotionFeature;
typedef FeatureVector<NUM_GENDER_FEATURE_DIM> GenderFeature;
typedef FeatureVector<NUM_RACE_FEATURE_DIM> RaceFeature;
typedef FeatureVector<NUM_AGE_FEATURE_DIM> AgeFeature;

class FaceAttributeInfo final {
    public:
    FaceAttributeInfo() = default;
    FaceAttributeInfo(FaceAttributeInfo &&other) { *this = std::move(other); }

    FaceAttributeInfo(const FaceAttributeInfo &other) { *this = other; }

    FaceAttributeInfo &operator=(const FaceAttributeInfo &other) {
        this->face_feature = other.face_feature;
        this->emotion_prob = other.emotion_prob;
        this->gender_prob = other.gender_prob;
        this->race_prob = other.race_prob;
        this->age_prob = other.age_prob;
        this->emotion = other.emotion;
        this->gender = other.gender;
        this->race = other.race;
        this->age = other.age;
        return *this;
    }

    FaceAttributeInfo &operator=(FaceAttributeInfo &&other) {
        this->face_feature = std::move(other.face_feature);
        this->emotion_prob = std::move(other.emotion_prob);
        this->gender_prob = std::move(other.gender_prob);
        this->race_prob = std::move(other.race_prob);
        this->age_prob = std::move(other.age_prob);
        this->emotion = other.emotion;
        this->gender = other.gender;
        this->race = other.race;
        this->age = other.age;
        return *this;
    }

    inline bool operator==(const FaceAttributeInfo &rhs) const {
        if (this->face_feature != rhs.face_feature || this->emotion_prob != rhs.emotion_prob ||
            this->gender_prob != rhs.gender_prob || this->race_prob != rhs.race_prob ||
            this->age_prob != rhs.age_prob || this->emotion != rhs.emotion ||
            this->gender != rhs.gender || this->race != rhs.race || this->age != rhs.age) {
            return false;
        }
        return true;
    }

    inline bool operator!=(const FaceAttributeInfo &rhs) const { return !(*this == rhs); }

    FaceFeature face_feature;
    EmotionFeature emotion_prob;
    GenderFeature gender_prob;
    RaceFeature race_prob;
    AgeFeature age_prob;
    FaceEmotion emotion = EMOTION_UNKNOWN;
    FaceGender gender = GENDER_UNKNOWN;
    FaceRace race = RACE_UNKNOWN;
    float age = -1.0;
};

}  // namespace vision
}  // namespace qnn