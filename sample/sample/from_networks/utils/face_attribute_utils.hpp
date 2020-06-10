// Copyright 2018 Bitmain Inc.
// License
// Author Tim Ho <tim.ho@bitmain.com>
/**
 * @file face_attribute_utils.hpp
 * @ingroup NETWORKS_UTILS
 */
#pragma once

#include "face/face.hpp"
#include <string>

namespace qnn {
namespace vision {

inline std::string GetEmotionString(FaceEmotion emotion) {
    switch (emotion) {
        case EMOTION_UNKNOWN:
            return std::string("Unknown");
        case EMOTION_HAPPY:
            return std::string("Happy");
        case EMOTION_SURPRISE:
            return std::string("Surprise");
        case EMOTION_FEAR:
            return std::string("Fear");
        case EMOTION_DISGUST:
            return std::string("Disgust");
        case EMOTION_SAD:
            return std::string("Sad");
        case EMOTION_ANGER:
            return std::string("Anger");
        case EMOTION_NEUTRAL:
            return std::string("Neutral");
        default:
            return std::string("Unknown");
    }
}

inline std::string GetGenderString(FaceGender gender) {
    switch (gender) {
        case GENDER_UNKNOWN:
            return std::string("Unknown");
        case GENDER_MALE:
            return std::string("Male");
        case GENDER_FEMALE:
            return std::string("Female");
        default:
            return std::string("Unknown");
    }
}

inline std::string GetRaceString(FaceRace race) {
    switch (race) {
        case RACE_UNKNOWN:
            return std::string("Unknown");
        case RACE_CAUCASIAN:
            return std::string("Caucasian");
        case RACE_BLACK:
            return std::string("Black");
        case RACE_ASIAN:
            return std::string("Asian");
        default:
            return std::string("Unknown");
    }
}

}  // namespace vision
}  // namespace qnn
