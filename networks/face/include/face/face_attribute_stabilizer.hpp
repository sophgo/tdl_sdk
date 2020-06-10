// Copyright 2018 Bitmain Inc.
// License
// Author Tim Ho <tim.ho@bitmain.com>
/**
 * @file face_attribute_stabilizer.hpp
 * @ingroup NETWORKS_ATTR
 */

#pragma once

#include "face.hpp"
#include "utils/log_common.h"
#include <chrono>
#include <map>
#include <math.h>
#include <queue>
#include <string>
#include <vector>

#define EMOTION_THRESHOLD (0.6)
#define GENDER_THRESHOLD (0.5)
#define RACE_THRESHOLD (0.8)

#define EMOTION_QUEUE_SIZE (5)
#define INFO_QUEUE_SIZE (30)

#define RESET_EMOTION_TIME (3)
#define RESET_INFO_TIME (20)

namespace qnn {
namespace vision {

typedef std::chrono::steady_clock::time_point t_timepoint;
typedef std::chrono::duration<float> t_duration;

enum class AggregateMethod {
    Sum,
    Vote,
};

template <size_t DIM>
struct TemporalAttributeQueue {
    private:
    std::deque<FeatureVector<DIM>> queue;
    AggregateMethod method;
    FeatureVector<DIM> agg;
    t_timepoint last_update_time;
    size_t count = 0;

    public:
    TemporalAttributeQueue(AggregateMethod agg_method) { method = agg_method; }
    TemporalAttributeQueue(TemporalAttributeQueue &&) = default;
    TemporalAttributeQueue &operator=(const TemporalAttributeQueue &other) = delete;
    TemporalAttributeQueue &operator=(TemporalAttributeQueue &&other) = default;

    void clear() {
        queue.clear();
        for (auto &f : agg.features) {
            f = 0;
        }
        count = 0;
    }

    void push(const FeatureVector<DIM> &feature, size_t capacity, const t_duration &expire_time) {
        auto now = std::chrono::steady_clock::now();
        auto duration = now - last_update_time;
        if (duration > expire_time) {
            clear();
        }

        while ((capacity > 0) && (queue.size() >= capacity)) {
            pop();
        }

        if (capacity > 0) {
            queue.emplace_back(feature);
        }
        count++;
        aggregate(feature);
        last_update_time = now;
    }

    void pop() {
        if (!queue.empty()) {
            deaggregate(queue.front());
            queue.pop_front();
            count--;
        }
    }

    FeatureVector<DIM> aggregated() const { return agg; }

    size_t size() const { return count; }

    private:
    void aggregate(const FeatureVector<DIM> &feature) {
        if (method == AggregateMethod::Sum) {
            agg += feature;
        } else {
            agg[feature.max_idx()] += 1;
        }
    }

    void deaggregate(const FeatureVector<DIM> &feature) {
        if (method == AggregateMethod::Sum) {
            agg -= feature;
        } else {
            agg[feature.max_idx()] -= 1;
        }
    }
};

typedef enum {
    STABILIZE_METHOD_AVERAGE = 0,
    STABILIZE_METHOD_VOTE,
    STABILIZE_METHOD_WEIGHTED_SUM
} StabilizeMethod;

template <typename IdType, size_t DIM>
class AttributeStabilizer {
    private:
    std::map<IdType, TemporalAttributeQueue<DIM>> queues;
    StabilizeMethod stab_method;
    AggregateMethod agg_method;
    t_duration expire_time;
    size_t capacity;
    double threshold;

    public:
    AttributeStabilizer(StabilizeMethod method, float expire_time, size_t capacity,
                        double threshold)
        : stab_method(method),
          agg_method(method == STABILIZE_METHOD_VOTE ? AggregateMethod::Vote
                                                     : AggregateMethod::Sum),
          expire_time(expire_time),
          capacity(capacity),
          threshold(threshold) {}

    template <typename T>
    bool Stabilize(const IdType &id, const FeatureVector<DIM> &feature, FeatureVector<DIM> &prob,
                   T &value) {
        if (!queues.count(id)) {
            queues.emplace(id, TemporalAttributeQueue<DIM>(agg_method));
        }

        auto &queue = queues.at(id);

        queue.push(feature, capacity, expire_time);
        auto agg = queue.aggregated();
        bool valid = false;

        if (stab_method == STABILIZE_METHOD_VOTE || stab_method == STABILIZE_METHOD_AVERAGE) {
            size_t i = agg.max_idx();
            if (agg.features[i]/float(queue.size()) > threshold) {
                value = i;
                valid = true;
            }
        } else if (stab_method == STABILIZE_METHOD_WEIGHTED_SUM) {
            double sum = 0;
            for (size_t i = 0; i < agg.features.size(); i++) {
                sum += agg[i] * i;
            }
            value = sum / queue.size();
            valid = true;
        }

        return valid;
    }

    void remove(const IdType &id) { queues.erase(id); }

    size_t size() const { return queues.size(); }
};

template <typename idType>
class FaceAttributeStabilizer {
    public:
    FaceAttributeStabilizer() : FaceAttributeStabilizer(STABILIZE_METHOD_AVERAGE) {}

    FaceAttributeStabilizer(StabilizeMethod method)
        : FaceAttributeStabilizer(method, INFO_QUEUE_SIZE, EMOTION_QUEUE_SIZE, RESET_INFO_TIME,
                                  RESET_EMOTION_TIME) {}

    FaceAttributeStabilizer(StabilizeMethod method, size_t info_queue_size,
                            size_t emotion_queue_size, float reset_info_time,
                            float reset_emotion_time)
        : emotion(method, reset_emotion_time, emotion_queue_size,
                  method == STABILIZE_METHOD_VOTE ? 0.5 : EMOTION_THRESHOLD),
          gender(method, reset_info_time, info_queue_size,
                 method == STABILIZE_METHOD_VOTE ? 0.5 : GENDER_THRESHOLD),
          race(method, reset_info_time, info_queue_size,
               method == STABILIZE_METHOD_VOTE ? 0.5 : RACE_THRESHOLD),
          age(STABILIZE_METHOD_WEIGHTED_SUM, reset_info_time, info_queue_size, 0) {}

    FaceAttributeStabilizer(StabilizeMethod gender_method, size_t gender_queue_size,
                            float gender_reset_time, StabilizeMethod race_method,
                            size_t race_queue_size, float race_reset_time,
                            StabilizeMethod age_method, size_t age_queue_size, float age_reset_time,
                            StabilizeMethod emotion_method, size_t emotion_queue_size,
                            float emotion_reset_time)
        : emotion(emotion_method, emotion_reset_time, emotion_queue_size,
                  emotion_method == STABILIZE_METHOD_VOTE ? 0.5 : EMOTION_THRESHOLD),
          gender(gender_method, gender_reset_time, gender_queue_size,
                 gender_method == STABILIZE_METHOD_VOTE ? 0.5 : GENDER_THRESHOLD),
          race(race_method, race_reset_time, race_queue_size,
               race_method == STABILIZE_METHOD_VOTE ? 0.5 : RACE_THRESHOLD),
          age(age_method, age_reset_time, age_queue_size, 0) {}

    void StabilizeFaceAttribute(const idType &id, const FaceAttributeInfo &src,
                                FaceAttributeInfo *dst) {
        size_t emotion_idx;
        size_t gender_idx;
        size_t race_idx;

        if (emotion.Stabilize(id, src.emotion_prob, dst->emotion_prob, emotion_idx)) {
            dst->emotion = (FaceEmotion)(emotion_idx + 1);
        } else {
            dst->emotion = EMOTION_NEUTRAL;
        }

        if (gender.Stabilize(id, src.gender_prob, dst->gender_prob, gender_idx)) {
            dst->gender = (FaceGender)(gender_idx + 1);
        } else {
            dst->gender = GENDER_UNKNOWN;
        }

        if (race.Stabilize(id, src.race_prob, dst->race_prob, race_idx)) {
            dst->race = (FaceRace)(race_idx + 1);
        } else {
            dst->race = RACE_UNKNOWN;
        }

        age.Stabilize(id, src.age_prob, dst->age_prob, dst->age);
    }

    void remove(const idType &id) {
        emotion.remove(id);
        gender.remove(id);
        race.remove(id);
        age.remove(id);
    }

    size_t GetTrackingListSize() { return emotion.size(); }

    private:
    AttributeStabilizer<idType, NUM_EMOTION_FEATURE_DIM> emotion;
    AttributeStabilizer<idType, NUM_GENDER_FEATURE_DIM> gender;
    AttributeStabilizer<idType, NUM_RACE_FEATURE_DIM> race;
    AttributeStabilizer<idType, NUM_AGE_FEATURE_DIM> age;
};

}  // namespace vision
}  // namespace qnn
