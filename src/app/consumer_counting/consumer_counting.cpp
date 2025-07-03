#include "consumer_counting.hpp"
#include <json.hpp>
#include "app/app_data_types.hpp"

#include "utils/tdl_log.hpp"

ConsumerCounting::ConsumerCounting(int A_x, int A_y, int B_x, int B_y,
                                   int mode) {
  A_x_ = A_x;
  A_y_ = A_y;
  B_x_ = B_x;
  B_y_ = B_y;
  mode_ = mode;

  // 对于竖直线，从左到右为进入，对于非竖直线，从上到下为进入

  assert(mode == 0 || mode == 1);

  int dx = B_x_ - A_x_;
  int dy = B_y_ - A_y_;

  if (dx == 0) {
    if (mode == 0) {
      normal_vector_x_ = 1.0;
      normal_vector_y_ = 0;
    } else {
      normal_vector_x_ = -1.0;
      normal_vector_y_ = 0;
    }
  } else {
    if ((dx > 0 && mode == 0) || (dx < 0 && mode == 1)) {
      normal_vector_x_ = (float)-dy;
      normal_vector_y_ = (float)dx;
    } else {
      normal_vector_x_ = (float)dy;
      normal_vector_y_ = (float)-dx;
    }
  }

  LOGI("normal_vector: x: %f. y: %f\n", normal_vector_x_, normal_vector_y_);
}

float ConsumerCounting::crossProduct(float A_x, float A_y, float B_x, float B_y,
                                     float C_x, float C_y) {
  return (B_x - A_x) * (C_y - A_y) - (B_y - A_y) * (C_x - A_x);
}

bool ConsumerCounting::isLineIntersect(float old_x, float old_y, float cur_x,
                                       float cur_y) {
  double cp1 = crossProduct(A_x_, A_y_, B_x_, B_y_, old_x, old_y);
  double cp2 = crossProduct(A_x_, A_y_, B_x_, B_y_, cur_x, cur_y);

  double cp3 = crossProduct(old_x, old_y, cur_x, cur_y, A_x_, A_y_);
  double cp4 = crossProduct(old_x, old_y, cur_x, cur_y, B_x_, B_y_);
  // positive and negative symbols
  if ((cp1 * cp2 <= 0) && (cp3 * cp4 <= 0)) return true;
  return false;
}

int32_t ConsumerCounting::consumer_counting(float old_x, float old_y,
                                            float cur_x, float cur_y,
                                            Consumer &it) {
  if (isLineIntersect(old_x, old_y, cur_x, cur_y)) {
    float tmp_x = cur_x - old_x;
    float tmp_y = cur_y - old_y;
    if ((tmp_x * normal_vector_x_ + tmp_y * normal_vector_y_ < 0) &&
        it.counting_gap == counting_gap_ && !it.miss) {
      miss_num_++;
      it.counting_gap = 0;
      it.miss = true;
    } else if ((tmp_x * normal_vector_x_ + tmp_y * normal_vector_y_ > 0) &&
               it.counting_gap == counting_gap_ && !it.enter) {
      enter_num_++;
      it.counting_gap = 0;
      it.enter = true;
    }
  }

  return 0;
}

int32_t ConsumerCounting::update_state(
    const std::vector<TrackerInfo> &track_results) {
  std::map<uint64_t, int> head_index;
  std::map<uint64_t, int> person_index;
  std::vector<int> new_index;

  for (size_t i = 0; i < track_results.size(); i++) {
    const TrackerInfo &t = track_results[i];

    uint64_t track_id = t.track_id_;

    if (t.status_ == TrackStatus::NEW) {
      new_index.push_back(i);
    } else if (t.box_info_.object_type == TDLObjectType::OBJECT_TYPE_HEAD) {
      head_index[track_id] = i;
    } else {
      if (t.box_info_.object_type != TDLObjectType::OBJECT_TYPE_PERSON) {
        LOGE("unexpected object_type: %d\n", t.box_info_.object_type);
        assert(false);
        return -1;
      }
      person_index[track_id] = i;
    }
  }

  for (auto it = heads_.begin(); it != heads_.end();) {
    if (head_index.count(it->first) == 0) {
      it->second.unmatched_times += 1;
      it->second.counting_gap =
          std::min(counting_gap_, it->second.counting_gap + 1);

      if (it->second.unmatched_times == MAX_UNMATCHED_TIME) {
        it = heads_.erase(it);
      }

    } else {
      const TrackerInfo &track_info = track_results[head_index[it->first]];
      uint64_t pair_id = track_info.pair_track_idx_;

      if (pair_id != -1 && persons_.count(pair_id) != 0 &&
          persons_[pair_id].counting_gap > 0) {
        it->second.enter = persons_[pair_id].enter;
        it->second.miss = persons_[pair_id].miss;

      } else if (it->second.enter && it->second.miss) {
        LOGI("track id %ld has been counted", it->first);
      } else if (it->second.counting_gap == counting_gap_) {
        ObjectBoxInfo box_info = track_info.box_info_;
        float cur_x = (box_info.x1 + box_info.x2) / 2.0;
        float cur_y = (box_info.y1 + box_info.y2) / 2.0;

        consumer_counting(it->second.old_x, it->second.old_y, cur_x, cur_y,
                          it->second);
        it->second.old_x = cur_x;
        it->second.old_y = cur_y;
      }

      it->second.unmatched_times = 0;
      it->second.counting_gap =
          std::min(counting_gap_, it->second.counting_gap + 1);
      it++;
    }
  }

  for (auto it = persons_.begin(); it != persons_.end();) {
    if (person_index.count(it->first) == 0) {
      it->second.unmatched_times += 1;
      it->second.counting_gap =
          std::min(counting_gap_, it->second.counting_gap + 1);

      if (it->second.unmatched_times == MAX_UNMATCHED_TIME) {
        it = persons_.erase(it);
      }

    } else {
      const TrackerInfo &track_info = track_results[person_index[it->first]];
      uint64_t pair_id = track_info.pair_track_idx_;

      if (pair_id != -1 && heads_.count(pair_id) != 0) {
        it->second.counting_gap =
            std::max(it->second.counting_gap, heads_[pair_id].counting_gap);
      }
      if (head_index.count(pair_id) != 0 &&
          track_results[head_index[pair_id]].obj_idx_ == -1) {
        ObjectBoxInfo box_info = track_info.box_info_;
        float cur_x = (box_info.x1 + box_info.x2) / 2.0;
        float cur_y = box_info.y1 * 1.2;

        consumer_counting(it->second.old_x, it->second.old_y, cur_x, cur_y,
                          it->second);
        it->second.old_x = cur_x;
        it->second.old_y = cur_y;
      }

      it->second.unmatched_times = 0;
      it->second.counting_gap =
          std::min(counting_gap_, it->second.counting_gap + 1);
      it++;
    }
  }

  for (uint32_t i = 0; i < new_index.size(); i++) {
    const TrackerInfo &t = track_results[new_index[i]];
    ObjectBoxInfo box_info = t.box_info_;

    uint64_t track_id = t.track_id_;

    Consumer new_consumer = {0};
    new_consumer.old_x = (box_info.x1 + box_info.x2) / 2.0;

    if (t.box_info_.object_type == TDLObjectType::OBJECT_TYPE_HEAD) {
      new_consumer.old_y = (box_info.y1 + box_info.y2) / 2.0;
      heads_[track_id] = new_consumer;
    } else if (t.box_info_.object_type == TDLObjectType::OBJECT_TYPE_PERSON) {
      new_consumer.old_y = box_info.y1 * 1.2;
      persons_[track_id] = new_consumer;
    }
  }

  return 0;
}