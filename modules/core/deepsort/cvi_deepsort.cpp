#include "cvi_deepsort.hpp"
#include "core/cviai_types_mem_internal.h"
#include "cvi_deepsort_utils.hpp"
#include "cviai_log.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>

Deepsort::Deepsort() {
  id_counter = 0;
  kf_ = KalmanFilter();
}

Deepsort::Deepsort(int feature_size) {
  id_counter = 0;
  kf_ = KalmanFilter();
  // this->feature_size = feature_size;
}

int Deepsort::track(cvai_object_t *obj, cvai_tracker_t *tracker_t) {
  std::vector<BBOX> bboxes;
  std::vector<FEATURE> features;
  uint32_t bbox_num = obj->size;
  for (uint32_t i = 0; i < bbox_num; i++) {
    BBOX bbox_;
    uint32_t feature_size = obj->info[i].feature.size;
    FEATURE feature_(feature_size);
    bbox_(0, 0) = obj->info[i].bbox.x1;
    bbox_(0, 1) = obj->info[i].bbox.y1;
    bbox_(0, 2) = obj->info[i].bbox.x2 - obj->info[i].bbox.x1;
    bbox_(0, 3) = obj->info[i].bbox.y2 - obj->info[i].bbox.y1;
    int type_size = size_of_feature_type(obj->info[i].feature.type);
    for (uint32_t d = 0; d < feature_size; d++) {
      feature_(d) = static_cast<float>(obj->info[i].feature.ptr[d * type_size]);
    }
    bboxes.push_back(bbox_);
    features.push_back(feature_);
  }

  auto result_ = track(bboxes, features);

  CVI_AI_MemAlloc(bbox_num, tracker_t);

  assert(result_.size() == static_cast<size_t>(bbox_num));
  for (size_t i = 0; i < result_.size(); i++) {
    bool &matched = std::get<0>(result_[i]);
    uint64_t &t_id = std::get<1>(result_[i]);
    TRACKER_STATE &t_state = std::get<2>(result_[i]);
    // BBOX &t_bbox = std::get<3>(result_[i]);
    if (!matched) {
      tracker_t->info[i].state = cvai_trk_state_type_t::CVI_TRACKER_NEW;
    } else if (t_state == TRACKER_STATE::PROBATION) {
      tracker_t->info[i].state = cvai_trk_state_type_t::CVI_TRACKER_UNSTABLE;
    } else if (t_state == TRACKER_STATE::ACCREDITATION) {
      tracker_t->info[i].state = cvai_trk_state_type_t::CVI_TRACKER_STABLE;
    } else {
      LOGE("Tracker State Unknow.\n");
      assert(0);
    }
    obj->info[i].unique_id = t_id;
  }

  return 0;
}

/* Result Format: [i] <is_matched, tracker_id, tracker_state, tracker_bbox> */
std::vector<std::tuple<bool, uint64_t, TRACKER_STATE, BBOX>> Deepsort::track(
    const std::vector<BBOX> &BBoxes, const std::vector<FEATURE> &Features) {
  std::stringstream ss_LOG_;
  std::vector<int> tracker_ids;
  for (size_t i = 0; i < k_trackers.size(); i++) {
    tracker_ids.push_back(k_trackers[i].id);
  }

  LOGI("Kalman Trackers predict\n");
  for (KalmanTracker &tracker_ : k_trackers) {
    kf_.predict(tracker_.kalman_state_, tracker_.x_, tracker_.P_);
  }

  std::vector<std::pair<int, int>> matched_pairs;
  std::vector<int> unmatched_bbox_idxes;
  for (size_t i = 0; i < BBoxes.size(); i++) {
    unmatched_bbox_idxes.push_back(i);
  }
  std::vector<int> unmatched_tracker_idxes = accreditation_tracker_idxes;
  std::vector<int> tmp_tracker_idxes; /* will be assign to tmp_tracker_idxes */

  ss_LOG_.str("");
  ss_LOG_ << "\n"
          << std::setw(32)
          << "Unmatched Tracker IDXes: " << get_INFO_Vector_Int(unmatched_tracker_idxes, 3)
          << std::endl
          << std::setw(32)
          << "Unmatched BBox IDXes: " << get_INFO_Vector_Int(unmatched_bbox_idxes, 3) << std::endl;
  LOGI("%s", ss_LOG_.str().c_str());

  LOGI("Cascade Match\n");
  /* Match accreditation trackers */
  /* - Cascade Match */
  /* - Feature Consine Distance */
  for (int t = 0; t < MAX_UNMATCHED_NUM; t++) {
    if (unmatched_bbox_idxes.empty()) {
      break;
    }
    std::vector<int> tmp_tracker_idxes;
    for (size_t tmp_i = 0; tmp_i < unmatched_tracker_idxes.size(); tmp_i++) {
      if (k_trackers[unmatched_tracker_idxes[tmp_i]].unmatched_times == t) {
        tmp_tracker_idxes.push_back(unmatched_tracker_idxes[tmp_i]);
      }
    }
    if (tmp_tracker_idxes.empty()) {
      continue;
    }
    // LOG(INFO) << "  [" << t << "] match";
    MatchResult match_result = match(BBoxes, Features, tmp_tracker_idxes, unmatched_bbox_idxes,
                                     "Feature_CosineDistance", MAX_DISTANCE_CONSINE);
    if (match_result.matched_pairs.empty()) {
      continue;
    }
    /* Remove matched idx from bbox_idxes and unmatched_tracker_idxes */
    matched_pairs.insert(matched_pairs.end(), match_result.matched_pairs.begin(),
                         match_result.matched_pairs.end());

    for (size_t m_i = 0; m_i < match_result.matched_pairs.size(); m_i++) {
      int m_tracker_idx = match_result.matched_pairs[m_i].first;
      int m_bbox_idx = match_result.matched_pairs[m_i].second;
      unmatched_tracker_idxes.erase(std::remove(unmatched_tracker_idxes.begin(),
                                                unmatched_tracker_idxes.end(), m_tracker_idx),
                                    unmatched_tracker_idxes.end());
      unmatched_bbox_idxes.erase(
          std::remove(unmatched_bbox_idxes.begin(), unmatched_bbox_idxes.end(), m_bbox_idx),
          unmatched_bbox_idxes.end());
    }
  }

  LOGI("Feature (Cosine) Match Result:\n");

  ss_LOG_.str("");
  ss_LOG_ << get_INFO_Match_Pair(matched_pairs, tracker_ids, 3) << std::endl;
  LOGI("%s", ss_LOG_.str().c_str());

  /* Remove trackers' idx, which unmatched_times > T, from
   * unmatched_tracker_idxes */
  const int T_ = MAX_UNMATCHED_TIMES_FOR_BBOX_MATCHING;
  for (auto it = unmatched_tracker_idxes.begin(); it != unmatched_tracker_idxes.end();) {
    if (k_trackers[*it].unmatched_times > T_) {
      tmp_tracker_idxes.push_back(*it);
      it = unmatched_tracker_idxes.erase(it);
    } else {
      it++;
    }
  }
  /* Append probation trackers */
  unmatched_tracker_idxes.insert(unmatched_tracker_idxes.end(), probation_tracker_idxes.begin(),
                                 probation_tracker_idxes.end());
  ss_LOG_.str("");
  // ss_GLOG_.clear();
  ss_LOG_ << std::endl
          << std::setw(32)
          << "Unmatched Tracker IDXes: " << get_INFO_Vector_Int(unmatched_tracker_idxes, 3)
          << std::endl
          << std::setw(32)
          << "Unmatched BBox IDXes: " << get_INFO_Vector_Int(unmatched_bbox_idxes, 3) << std::endl;
  LOGI("%s", ss_LOG_.str().c_str());
  /* Match remain trackers */
  /* - BBox IoU Distance */
  MatchResult match_result_bbox = match(BBoxes, Features, unmatched_tracker_idxes,
                                        unmatched_bbox_idxes, "BBox_IoUDistance", MAX_DISTANCE_IOU);

  LOGI("BBOX (IoU) Match Result:\n");
  ss_LOG_.str("");
  ss_LOG_ << get_INFO_Match_Pair(match_result_bbox.matched_pairs, tracker_ids, 3) << std::endl;
  LOGI("%s", ss_LOG_.str().c_str());

  matched_pairs.insert(matched_pairs.end(), match_result_bbox.matched_pairs.begin(),
                       match_result_bbox.matched_pairs.end());
  unmatched_bbox_idxes = match_result_bbox.unmatched_bbox_idxes;
  unmatched_tracker_idxes = tmp_tracker_idxes;
  unmatched_tracker_idxes.insert(unmatched_tracker_idxes.end(),
                                 match_result_bbox.unmatched_tracker_idxes.begin(),
                                 match_result_bbox.unmatched_tracker_idxes.end());

  /* Generate tracker result */
  /*   format: [i] <is_matched, tracker_id, tracker_state, tracker_bbox> */
  std::vector<std::tuple<bool, uint64_t, TRACKER_STATE, BBOX>> result_(BBoxes.size());
  LOGI("Create Result_ (%zu)", result_.size());

  /* Update the kalman trackers (Matched) */
  LOGI("Update the kalman trackers (Matched)");
  for (size_t i = 0; i < matched_pairs.size(); i++) {
    int tracker_idx = matched_pairs[i].first;
    int bbox_idx = matched_pairs[i].second;
    // LOG(INFO) << "update >> tracker idx: " << tracker_idx << ", bbox idx: " << bbox_idx;
    LOGI("update >> tracker idx: %d, bbox idx: %d", tracker_idx, bbox_idx);
    KalmanTracker &tracker_ = k_trackers[tracker_idx];
    const BBOX &bbox_ = BBoxes[bbox_idx];
    const FEATURE &feature_ = Features[bbox_idx];
    kf_.update(tracker_.kalman_state_, tracker_.x_, tracker_.P_, bbox_tlwh2xyah(bbox_));
    tracker_.update_bbox(bbox_);
    tracker_.update_feature(feature_);
    tracker_.update_state(true);
    result_[bbox_idx] =
        std::make_tuple(true, tracker_.id, tracker_.tracker_state_, tracker_.getBBox_TLWH());
  }

  /* Update the kalman trackers (Unmatched) */
  LOGI("Update the kalman trackers (Unmatched)");
  for (size_t i = 0; i < unmatched_tracker_idxes.size(); i++) {
    int tracker_idx = unmatched_tracker_idxes[i];
    LOGI("update >> tracker idx: %d", tracker_idx);
    KalmanTracker &tracker_ = k_trackers[tracker_idx];
    tracker_.update_state(false);
  }

  /* Check kalman trackers state, and remove invalid trackers */
  LOGI("Check kalman trackers state, and remove invalid trackers");
  for (auto it_ = k_trackers.begin(); it_ != k_trackers.end();) {
    if (it_->tracker_state_ == TRACKER_STATE::MISS) {
      it_ = k_trackers.erase(it_);
    } else {
      it_++;
    }
  }

  /* Create new kalman trackers (Unmatched BBoxes) */
  LOGI("Create new kalman trackers (Unmatched BBoxes)");
  for (size_t i = 0; i < unmatched_bbox_idxes.size(); i++) {
    int bbox_idx = unmatched_bbox_idxes[i];
    id_counter += 1;
    const BBOX &bbox_ = BBoxes[bbox_idx];
    const FEATURE &feature_ = Features[bbox_idx];
    LOGI("create >> id: %" PRIu64 ", bbox: [%f,%f,%f,%f]", id_counter, bbox_(0, 0), bbox_(0, 1),
         bbox_(0, 2), bbox_(0, 3));
    KalmanTracker tracker_(id_counter, bbox_, feature_);
    k_trackers.push_back(tracker_);
    result_[bbox_idx] =
        std::make_tuple(false, tracker_.id, tracker_.tracker_state_, tracker_.getBBox_TLWH());
  }

  /* Update accreditation & probation tracker idxes */
  LOGI("Update accreditation & probation tracker idxes");
  accreditation_tracker_idxes.clear();
  probation_tracker_idxes.clear();
  for (size_t i = 0; i < k_trackers.size(); i++) {
    if (k_trackers[i].tracker_state_ == TRACKER_STATE::ACCREDITATION) {
      accreditation_tracker_idxes.push_back(i);
    } else if (k_trackers[i].tracker_state_ == TRACKER_STATE::PROBATION) {
      probation_tracker_idxes.push_back(i);
    } else {
      assert(0);
    }
  }

  // show_INFO_KalmanTrackers();

  return result_;
}

MatchResult Deepsort::match(const std::vector<BBOX> &BBoxes, const std::vector<FEATURE> &Features,
                            const std::vector<int> &Tracker_IDXes,
                            const std::vector<int> &BBox_IDXes, std::string cost_method,
                            float max_distance) {
  MatchResult result_;
  if (Tracker_IDXes.empty() || BBox_IDXes.empty()) {
    result_.unmatched_tracker_idxes = Tracker_IDXes;
    result_.unmatched_bbox_idxes = BBox_IDXes;
    return result_;
  }

  COST_MATRIX cost_matrix;
  if (cost_method == "Feature_CosineDistance") {
    cost_matrix = KalmanTracker::getCostMatrix_Feature(k_trackers, BBoxes, Features, Tracker_IDXes,
                                                       BBox_IDXes);
    LOGI("Feature Cost Matrix (Consine Distance)");
    // std::cout << cost_matrix << std::endl;
    float gate_value = max_distance;
    gateCostMatrix_Mahalanobis(cost_matrix, kf_, k_trackers, BBoxes, Tracker_IDXes, BBox_IDXes,
                               gate_value);
    LOGI("Cost Matrix (before Munkres)");
    // std::cout << cost_matrix << std::endl;

    // gate_cost_matrix(cost_matrix, max_distance);
  } else if (cost_method == "BBox_IoUDistance") {
    cost_matrix =
        KalmanTracker::getCostMatrix_BBox(k_trackers, BBoxes, Features, Tracker_IDXes, BBox_IDXes);
    gate_cost_matrix(cost_matrix, max_distance);
  } else {
    std::cout << "Cost Method: " << cost_method << " not support." << std::endl;
    assert(0);
  }
  CVIMunkres cvi_munkres_solver(&cost_matrix);
  cvi_munkres_solver.solve();

  int bbox_num = BBox_IDXes.size();
  int tracker_num = Tracker_IDXes.size();
  bool *matched_tracker_i = new bool[tracker_num];
  bool *matched_bbox_j = new bool[bbox_num];
  memset(matched_tracker_i, false, tracker_num * sizeof(bool));
  memset(matched_bbox_j, false, bbox_num * sizeof(bool));

  for (int i = 0; i < tracker_num; i++) {
    int bbox_j = cvi_munkres_solver.match_result[i];
    if (bbox_j != -1) {
      if (cost_matrix(i, bbox_j) < max_distance) {
        matched_tracker_i[i] = true;
        matched_bbox_j[bbox_j] = true;
        int tracker_idx = Tracker_IDXes[i];
        int bbox_idx = BBox_IDXes[bbox_j];
        result_.matched_pairs.push_back(std::make_pair(tracker_idx, bbox_idx));
      }
    }
  }

  for (int i = 0; i < tracker_num; i++) {
    if (!matched_tracker_i[i]) {
      int tracker_idx = Tracker_IDXes[i];
      result_.unmatched_tracker_idxes.push_back(tracker_idx);
    }
  }

  for (int j = 0; j < bbox_num; j++) {
    if (!matched_bbox_j[j]) {
      int bbox_idx = BBox_IDXes[j];
      result_.unmatched_bbox_idxes.push_back(bbox_idx);
    }
  }

  delete[] matched_tracker_i;
  delete[] matched_bbox_j;

  return result_;
}

float chi2inv95[10] = {0, 3.8415, 5.9915, 7.8147, 9.4877, 11.070, 12.592, 14.067, 15.507, 16.919};
void Deepsort::gateCostMatrix_Mahalanobis(COST_MATRIX &cost_matrix, const KalmanFilter &KF_,
                                          const std::vector<KalmanTracker> &K_Trackers,
                                          const std::vector<BBOX> &BBoxes,
                                          const std::vector<int> &Tracker_IDXes,
                                          const std::vector<int> &BBox_IDXes, float gate_value) {
  float chi2_threshold = chi2inv95[4];
  BBOXES measurement_bboxes(BBox_IDXes.size(), 4);
  // BBOXES measurement_bboxes(BBox_IDXes.size());
  for (size_t i = 0; i < BBox_IDXes.size(); i++) {
    int bbox_idx = BBox_IDXes[i];
    measurement_bboxes.row(i) = bbox_tlwh2xyah(BBoxes[bbox_idx]);
  }
  for (size_t i = 0; i < Tracker_IDXes.size(); i++) {
    int tracker_idx = Tracker_IDXes[i];
    const KalmanTracker &tracker_ = K_Trackers[tracker_idx];
    ROW_VECTOR maha2_d =
        KF_.mahalanobis(tracker_.kalman_state_, tracker_.x_, tracker_.P_, measurement_bboxes);
    for (int j = 0; j < maha2_d.cols(); j++) {
      if (maha2_d(0, j) > chi2_threshold) {
        cost_matrix(i, j) = gate_value;
      }
    }
  }
}

/* DEBUG CODE*/
void Deepsort::show_INFO_KalmanTrackers() {
  for (size_t i = 0; i < k_trackers.size(); i++) {
    KalmanTracker &tracker_ = k_trackers[i];
    std::cout << "[" << std::setw(3) << i << "] Tracker ID: " << tracker_.id << std::endl;
    std::cout << "\t" << std::setw(20) << "tracker state = " << tracker_.get_INFO_TrackerState()
              << std::endl;
    std::cout << "\t" << std::setw(20) << "kalman state = " << tracker_.get_INFO_KalmanState()
              << std::endl;
    std::cout << "\t" << std::setw(20) << "bbox = " << tracker_.bbox << std::endl;
    std::cout << "\t" << std::setw(20) << "unmatched times = " << tracker_.unmatched_times
              << std::endl;
    std::cout << "\t" << std::setw(20) << "matched counter = " << tracker_.get_MatchedCounter()
              << std::endl;
    std::cout << "\t" << std::setw(30)
              << "feauture update counter = " << tracker_.get_FeatureUpdateCounter() << std::endl;
    std::cout << "\t" << std::setw(30) << "Kalman x_ = " << tracker_.x_.transpose() << std::endl;
    std::cout << "\t" << std::setw(30) << "Kalman P_ = " << tracker_.P_ << std::endl;
  }
}

std::vector<KalmanTracker> Deepsort::get_Trackers_UnmatchedLastTime() const {
  std::vector<KalmanTracker> unmatched_trackers;
  for (const KalmanTracker &tracker_ : k_trackers) {
    if (tracker_.unmatched_times > 0) {
      unmatched_trackers.push_back(tracker_);
    }
  }
  return unmatched_trackers;
}

bool Deepsort::get_Tracker_ByID(uint64_t id, KalmanTracker &tracker) const {
  for (const KalmanTracker &tracker_ : k_trackers) {
    if (tracker_.id == id) {
      tracker = tracker_;
      return true;
    }
  }
  return false;
}