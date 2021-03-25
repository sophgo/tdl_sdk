#include "cvi_deepsort.hpp"
#include "core/cviai_types_mem_internal.h"
#include "cvi_deepsort_utils.hpp"
#include "cviai_log.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <map>
#include <set>

#define DEBUG_DEEPSORT 0

static void show_deepsort_config(cvai_deepsort_config_t &ds_conf);

DeepSORT::DeepSORT() {
  id_counter = 0;
  kf_ = KalmanFilter();

  conf = get_DefaultConfig();
}

DeepSORT::DeepSORT(cvai_deepsort_config_t ds_conf) {
  id_counter = 0;
  kf_ = KalmanFilter();

  conf = ds_conf;
}

int DeepSORT::track(cvai_object_t *obj, cvai_tracker_t *tracker_t, bool use_reid,
                    bool track_by_class) {
  if (track_by_class) {
    std::map<int, int> class_id_counter;
    std::set<int> class_ids_bbox;
    std::set<int> class_ids_trackers;
    for (uint32_t i = 0; i < obj->size; i++) {
      auto iter = class_id_counter.find(obj->info[i].classes);
      if (iter != class_id_counter.end()) {
        iter->second += 1;
      } else {
        class_id_counter.insert(std::pair<int, int>(obj->info[i].classes, 1));
        class_ids_bbox.insert(obj->info[i].classes);
      }
    }

    for (size_t j = 0; j < k_trackers.size(); j++) {
      class_ids_trackers.insert(k_trackers[j].class_id);
    }

    CVI_AI_MemAlloc(obj->size, tracker_t);

    for (auto const &e : class_id_counter) {
      std::vector<BBOX> bboxes;
      std::vector<FEATURE> features;
      std::vector<int> idx_table;
      for (uint32_t i = 0; i < obj->size; i++) {
        if (obj->info[i].classes == e.first) {
          idx_table.push_back(static_cast<int>(i));
          BBOX bbox_;
          uint32_t feature_size = obj->info[i].feature.size;
          FEATURE feature_(feature_size);
          bbox_(0, 0) = obj->info[i].bbox.x1;
          bbox_(0, 1) = obj->info[i].bbox.y1;
          bbox_(0, 2) = obj->info[i].bbox.x2 - obj->info[i].bbox.x1;
          bbox_(0, 3) = obj->info[i].bbox.y2 - obj->info[i].bbox.y1;
          if (obj->info[i].feature.type != TYPE_INT8) {
            printf("Feature Type not support now.\n");
            return CVI_FAILURE;
          }
          int type_size = getFeatureTypeSize(obj->info[i].feature.type);
          for (uint32_t d = 0; d < feature_size; d++) {
            feature_(d) = static_cast<float>(obj->info[i].feature.ptr[d * type_size]);
          }
          bboxes.push_back(bbox_);
          features.push_back(feature_);
        }
      }
      assert(idx_table.size() == static_cast<size_t>(e.second));

      auto result_ = track(bboxes, features, e.first, use_reid);

      for (size_t i = 0; i < result_.size(); i++) {
        int idx = idx_table[i];
        bool &matched = std::get<0>(result_[i]);
        uint64_t &t_id = std::get<1>(result_[i]);
        TRACKER_STATE &t_state = std::get<2>(result_[i]);
        BBOX &t_bbox = std::get<3>(result_[i]);
        if (!matched) {
          tracker_t->info[idx].state = cvai_trk_state_type_t::CVI_TRACKER_NEW;
        } else if (t_state == TRACKER_STATE::PROBATION) {
          tracker_t->info[idx].state = cvai_trk_state_type_t::CVI_TRACKER_UNSTABLE;
        } else if (t_state == TRACKER_STATE::ACCREDITATION) {
          tracker_t->info[idx].state = cvai_trk_state_type_t::CVI_TRACKER_STABLE;
        } else {
          LOGE("Tracker State Unknow.\n");
          return CVI_FAILURE;
        }
        tracker_t->info[idx].bbox.x1 = t_bbox(0);
        tracker_t->info[idx].bbox.y1 = t_bbox(1);
        tracker_t->info[idx].bbox.x2 = t_bbox(0) + t_bbox(2);
        tracker_t->info[idx].bbox.y2 = t_bbox(1) + t_bbox(3);
        obj->info[idx].unique_id = t_id;
      }
    }

    for (std::set<int>::iterator it = class_ids_trackers.begin(); it != class_ids_trackers.end();
         ++it) {
      if (class_ids_bbox.find(*it) == class_ids_bbox.end()) {
        std::vector<BBOX> bboxes;
        std::vector<FEATURE> features;
        auto result_ = track(bboxes, features, *it, use_reid);
      }
    }

  } else {
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
      if (obj->info[i].feature.type != TYPE_INT8) {
        printf("Feature Type not support now.\n");
        return CVI_FAILURE;
      }
      int type_size = getFeatureTypeSize(obj->info[i].feature.type);
      for (uint32_t d = 0; d < feature_size; d++) {
        feature_(d) = static_cast<float>(obj->info[i].feature.ptr[d * type_size]);
      }
      bboxes.push_back(bbox_);
      features.push_back(feature_);
    }

    auto result_ = track(bboxes, features, -1, use_reid);
    CVI_AI_MemAlloc(bbox_num, tracker_t);

    assert(result_.size() == static_cast<size_t>(bbox_num));
    for (size_t i = 0; i < result_.size(); i++) {
      bool &matched = std::get<0>(result_[i]);
      uint64_t &t_id = std::get<1>(result_[i]);
      TRACKER_STATE &t_state = std::get<2>(result_[i]);
      BBOX &t_bbox = std::get<3>(result_[i]);
      if (!matched) {
        tracker_t->info[i].state = cvai_trk_state_type_t::CVI_TRACKER_NEW;
      } else if (t_state == TRACKER_STATE::PROBATION) {
        tracker_t->info[i].state = cvai_trk_state_type_t::CVI_TRACKER_UNSTABLE;
      } else if (t_state == TRACKER_STATE::ACCREDITATION) {
        tracker_t->info[i].state = cvai_trk_state_type_t::CVI_TRACKER_STABLE;
      } else {
        LOGE("Tracker State Unknow.\n");
        return CVI_FAILURE;
      }
      tracker_t->info[i].bbox.x1 = t_bbox(0);
      tracker_t->info[i].bbox.y1 = t_bbox(1);
      tracker_t->info[i].bbox.x2 = t_bbox(0) + t_bbox(2);
      tracker_t->info[i].bbox.y2 = t_bbox(1) + t_bbox(3);
      obj->info[i].unique_id = t_id;
    }
  }

  return CVI_SUCCESS;
}

int DeepSORT::track(cvai_face_t *face, cvai_tracker_t *tracker_t, bool use_reid) {
  std::vector<BBOX> bboxes;
  std::vector<FEATURE> features;
  uint32_t bbox_num = face->size;
  for (uint32_t i = 0; i < bbox_num; i++) {
    BBOX bbox_;
    FEATURE feature_(0);
    bbox_(0, 0) = face->info[i].bbox.x1;
    bbox_(0, 1) = face->info[i].bbox.y1;
    bbox_(0, 2) = face->info[i].bbox.x2 - face->info[i].bbox.x1;
    bbox_(0, 3) = face->info[i].bbox.y2 - face->info[i].bbox.y1;
    bboxes.push_back(bbox_);
    features.push_back(feature_);
  }

  auto result_ = track(bboxes, features, -1, use_reid);
  CVI_AI_MemAlloc(bbox_num, tracker_t);

  assert(result_.size() == static_cast<size_t>(bbox_num));
  for (size_t i = 0; i < result_.size(); i++) {
    bool &matched = std::get<0>(result_[i]);
    uint64_t &t_id = std::get<1>(result_[i]);
    TRACKER_STATE &t_state = std::get<2>(result_[i]);
    BBOX &t_bbox = std::get<3>(result_[i]);
    if (!matched) {
      tracker_t->info[i].state = cvai_trk_state_type_t::CVI_TRACKER_NEW;
    } else if (t_state == TRACKER_STATE::PROBATION) {
      tracker_t->info[i].state = cvai_trk_state_type_t::CVI_TRACKER_UNSTABLE;
    } else if (t_state == TRACKER_STATE::ACCREDITATION) {
      tracker_t->info[i].state = cvai_trk_state_type_t::CVI_TRACKER_STABLE;
    } else {
      LOGE("Tracker State Unknow.\n");
      return CVI_FAILURE;
    }
    tracker_t->info[i].bbox.x1 = t_bbox(0);
    tracker_t->info[i].bbox.y1 = t_bbox(1);
    tracker_t->info[i].bbox.x2 = t_bbox(0) + t_bbox(2);
    tracker_t->info[i].bbox.y2 = t_bbox(1) + t_bbox(3);
    face->info[i].unique_id = t_id;
  }
  return CVI_SUCCESS;
}

/* Result Format: [i] <is_matched, tracker_id, tracker_state, tracker_bbox> */
std::vector<std::tuple<bool, uint64_t, TRACKER_STATE, BBOX>> DeepSORT::track(
    const std::vector<BBOX> &BBoxes, const std::vector<FEATURE> &Features, int class_id,
    bool use_reid) {
  std::stringstream ss_LOG_;
  std::vector<int> tracker_ids;
  for (size_t i = 0; i < k_trackers.size(); i++) {
    tracker_ids.push_back(k_trackers[i].id);
  }

  LOGI("Kalman Trackers predict\n");
  for (KalmanTracker &tracker_ : k_trackers) {
    kf_.predict(tracker_.kalman_state_, tracker_.x_, tracker_.P_, conf.kfilter_conf);
  }

  std::vector<std::pair<int, int>> matched_pairs;
  std::vector<int> unmatched_bbox_idxes;
  for (size_t i = 0; i < BBoxes.size(); i++) {
    unmatched_bbox_idxes.push_back(i);
  }

  std::vector<int> unmatched_tracker_idxes;
  if (class_id != -1) {
    for (std::vector<int>::iterator iter = accreditation_tracker_idxes.begin();
         iter != accreditation_tracker_idxes.end(); iter++) {
      if (k_trackers[*iter].class_id == class_id) {
        unmatched_tracker_idxes.push_back(*iter);
      }
    }
    /* Append probation trackers */
    for (std::vector<int>::iterator iter = probation_tracker_idxes.begin();
         iter != probation_tracker_idxes.end(); iter++) {
      if (k_trackers[*iter].class_id == class_id) {
        unmatched_tracker_idxes.push_back(*iter);
      }
    }
  } else {
    unmatched_tracker_idxes.insert(unmatched_tracker_idxes.end(),
                                   accreditation_tracker_idxes.begin(),
                                   accreditation_tracker_idxes.end());
    unmatched_tracker_idxes.insert(unmatched_tracker_idxes.end(), probation_tracker_idxes.begin(),
                                   probation_tracker_idxes.end());
  }

#if DEBUG_DEEPSORT
  ss_LOG_.str("");
  ss_LOG_ << "\n"
          << std::setw(32)
          << "Unmatched Tracker IDXes: " << get_INFO_Vector_Int(unmatched_tracker_idxes, 3)
          << std::endl
          << std::setw(32)
          << "Unmatched BBox IDXes: " << get_INFO_Vector_Int(unmatched_bbox_idxes, 3) << std::endl;
  LOGI("%s", ss_LOG_.str().c_str());
#endif

  LOGI("Cascade Match\n");
  /* Match accreditation trackers */
  /* - Cascade Match */
  /* - Feature Consine Distance */
  /* - Kalman Mahalanobis Distance */
  for (int t = 0; t < conf.ktracker_conf.max_unmatched_num; t++) {
    if (unmatched_bbox_idxes.empty()) {
      break;
    }
    std::vector<int> t_tracker_idxes;
    for (size_t tmp_i = 0; tmp_i < unmatched_tracker_idxes.size(); tmp_i++) {
      if (k_trackers[unmatched_tracker_idxes[tmp_i]].unmatched_times == t) {
        t_tracker_idxes.push_back(unmatched_tracker_idxes[tmp_i]);
      }
    }
    if (t_tracker_idxes.empty()) {
      continue;
    }
    MatchResult match_result =
        match(BBoxes, Features, t_tracker_idxes, unmatched_bbox_idxes,
              (use_reid) ? "Feature_CosineDistance" : "Kalman_MahalanobisDistance",
              (use_reid) ? conf.max_distance_consine : chi2inv95[4]);
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

#if DEBUG_DEEPSORT
  LOGI("Cascade Match Result:\n");
  std::cout << "Cascade Match Result:\n";
  ss_LOG_.str("");
  ss_LOG_ << std::endl << get_INFO_Match_Pair(matched_pairs, tracker_ids, 3) << std::endl;
  LOGI("%s", ss_LOG_.str().c_str());
  std::cout << ss_LOG_.str();
#endif

  std::vector<int> tmp_tracker_idxes; /* unmatch trackers' index in cascade match */
  /* Remove trackers' idx, which unmatched_times > T, from
   * unmatched_tracker_idxes */
  for (auto it = unmatched_tracker_idxes.begin(); it != unmatched_tracker_idxes.end();) {
    if (k_trackers[*it].unmatched_times > conf.max_unmatched_times_for_bbox_matching) {
      tmp_tracker_idxes.push_back(*it);
      it = unmatched_tracker_idxes.erase(it);
    } else {
      it++;
    }
  }

#if 0
  /* Append probation trackers */
  unmatched_tracker_idxes.insert(unmatched_tracker_idxes.end(),
    probation_tracker_idxes.begin(), probation_tracker_idxes.end());
#endif

#if DEBUG_DEEPSORT
  ss_LOG_.str("");
  ss_LOG_ << std::endl
          << std::setw(32)
          << "Unmatched Tracker IDXes: " << get_INFO_Vector_Int(unmatched_tracker_idxes, 3)
          << std::endl
          << std::setw(32)
          << "Unmatched BBox IDXes: " << get_INFO_Vector_Int(unmatched_bbox_idxes, 3) << std::endl;
  LOGI("%s", ss_LOG_.str().c_str());
  std::cout << ss_LOG_.str();
#endif

  /* Match remain trackers */
  /* - BBox IoU Distance */
  MatchResult match_result_bbox =
      match(BBoxes, Features, unmatched_tracker_idxes, unmatched_bbox_idxes, "BBox_IoUDistance",
            conf.max_distance_iou);

#if DEBUG_DEEPSORT
  LOGI("BBOX (IoU) Match Result:\n");
  ss_LOG_.str("");
  ss_LOG_ << get_INFO_Match_Pair(match_result_bbox.matched_pairs, tracker_ids, 3) << std::endl;
  LOGI("%s", ss_LOG_.str().c_str());
#endif

  /* Match remain trackers */
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
#if DEBUG_DEEPSORT
  LOGI("Create Result_ (%zu)", result_.size());
#endif

  /* Update the kalman trackers (Matched) */
  LOGI("Update the kalman trackers (Matched)");
  for (size_t i = 0; i < matched_pairs.size(); i++) {
    int tracker_idx = matched_pairs[i].first;
    int bbox_idx = matched_pairs[i].second;
    // LOGI("update >> tracker idx: %d, bbox idx: %d", tracker_idx, bbox_idx);
    KalmanTracker &tracker_ = k_trackers[tracker_idx];
    const BBOX &bbox_ = BBoxes[bbox_idx];
    const FEATURE &feature_ = Features[bbox_idx];
    kf_.update(tracker_.kalman_state_, tracker_.x_, tracker_.P_, bbox_tlwh2xyah(bbox_),
               conf.kfilter_conf);
    tracker_.update_bbox(bbox_);
    tracker_.update_feature(feature_, conf.ktracker_conf.feature_budget_size,
                            conf.ktracker_conf.feature_update_interval);
    tracker_.update_state(true, conf.ktracker_conf.max_unmatched_num,
                          conf.ktracker_conf.accreditation_threshold);
    result_[bbox_idx] =
        std::make_tuple(true, tracker_.id, tracker_.tracker_state_, tracker_.getBBox_TLWH());
  }

  /* Update the kalman trackers (Unmatched) */
  LOGI("Update the kalman trackers (Unmatched)");
  for (size_t i = 0; i < unmatched_tracker_idxes.size(); i++) {
    int tracker_idx = unmatched_tracker_idxes[i];
    // LOGI("update >> tracker idx: %d", tracker_idx);
    KalmanTracker &tracker_ = k_trackers[tracker_idx];
    tracker_.update_state(false, conf.ktracker_conf.max_unmatched_num,
                          conf.ktracker_conf.accreditation_threshold);
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
    // LOGI("new >> tracker idx: %d", id_counter);
    // KalmanTracker tracker_(id_counter, bbox_, feature_);
    KalmanTracker tracker_(id_counter, class_id, bbox_, feature_, conf.ktracker_conf);
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

  return result_;
}

MatchResult DeepSORT::match(const std::vector<BBOX> &BBoxes, const std::vector<FEATURE> &Features,
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
  if (cost_method == "Kalman_MahalanobisDistance") {
    LOGI("Kalman Cost Matrix (Mahalanobis Distance)");
    // std::cout << "Kalman Cost Matrix (Mahalanobis Distance)" << std::endl;
    cost_matrix = KalmanTracker::getCostMatrix_Mahalanobis(
        kf_, k_trackers, BBoxes, Tracker_IDXes, BBox_IDXes, conf.kfilter_conf, max_distance);
    // std::cout << cost_matrix << std::endl;
  } else if (cost_method == "Feature_CosineDistance") {
    cost_matrix = KalmanTracker::getCostMatrix_Feature(k_trackers, BBoxes, Features, Tracker_IDXes,
                                                       BBox_IDXes);
    LOGI("Feature Cost Matrix (Consine Distance)");
    // std::cout << "Cost Matrix (before Munkres)" << std::endl;
    // std::cout << cost_matrix << std::endl;
    KalmanTracker::gateCostMatrix_Mahalanobis(cost_matrix, kf_, k_trackers, BBoxes, Tracker_IDXes,
                                              BBox_IDXes, conf.kfilter_conf, max_distance);

    // gate_cost_matrix(cost_matrix, max_distance);
  } else if (cost_method == "BBox_IoUDistance") {
    // std::cout << "BBox Cost Matrix (IoU Distance)" << std::endl;
    cost_matrix =
        KalmanTracker::getCostMatrix_BBox(k_trackers, BBoxes, Features, Tracker_IDXes, BBox_IDXes);
    gate_cost_matrix(cost_matrix, max_distance);
    // std::cout << cost_matrix << std::endl;
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
    int bbox_j = cvi_munkres_solver.m_match_result[i];
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

void DeepSORT::setConfig(cvai_deepsort_config_t ds_conf) {
  memcpy(&conf, &ds_conf, sizeof(cvai_deepsort_config_t));
  show_deepsort_config(conf);
}

cvai_deepsort_config_t DeepSORT::get_DefaultConfig() {
  cvai_deepsort_config_t conf;
  conf.max_distance_consine = 0.05;
  conf.max_distance_iou = 0.7;
  conf.max_unmatched_times_for_bbox_matching = 2;

  conf.ktracker_conf.accreditation_threshold = 3;
  conf.ktracker_conf.feature_budget_size = 8;
  conf.ktracker_conf.feature_update_interval = 1;
  conf.ktracker_conf.max_unmatched_num = 40;

  conf.ktracker_conf.P_std_alpha[0] = 2 * 1 / 20.0;
  conf.ktracker_conf.P_std_alpha[1] = 2 * 1 / 20.0;
  conf.ktracker_conf.P_std_alpha[2] = 0.0;
  conf.ktracker_conf.P_std_alpha[3] = 2 * 1 / 20.0;
  conf.ktracker_conf.P_std_alpha[4] = 10 * 1 / 160.0;
  conf.ktracker_conf.P_std_alpha[5] = 10 * 1 / 160.0;
  conf.ktracker_conf.P_std_alpha[6] = 0.0;
  conf.ktracker_conf.P_std_alpha[7] = 10 * 1 / 160.0;
  conf.ktracker_conf.P_std_x_idx[0] = 3;
  conf.ktracker_conf.P_std_x_idx[1] = 3;
  conf.ktracker_conf.P_std_x_idx[2] = -1;
  conf.ktracker_conf.P_std_x_idx[3] = 3;
  conf.ktracker_conf.P_std_x_idx[4] = 3;
  conf.ktracker_conf.P_std_x_idx[5] = 3;
  conf.ktracker_conf.P_std_x_idx[6] = -1;
  conf.ktracker_conf.P_std_x_idx[7] = 3;
  conf.ktracker_conf.P_std_beta[0] = 0.0;
  conf.ktracker_conf.P_std_beta[1] = 0.0;
  conf.ktracker_conf.P_std_beta[2] = 0.01;
  conf.ktracker_conf.P_std_beta[3] = 0.0;
  conf.ktracker_conf.P_std_beta[4] = 0.0;
  conf.ktracker_conf.P_std_beta[5] = 0.0;
  conf.ktracker_conf.P_std_beta[6] = 1e-5;
  conf.ktracker_conf.P_std_beta[7] = 0.0;

  conf.kfilter_conf.Q_std_alpha[0] = 1 / 20.0;
  conf.kfilter_conf.Q_std_alpha[1] = 1 / 20.0;
  conf.kfilter_conf.Q_std_alpha[2] = 0.0;
  conf.kfilter_conf.Q_std_alpha[3] = 1 / 20.0;
  conf.kfilter_conf.Q_std_alpha[4] = 1 / 160.0;
  conf.kfilter_conf.Q_std_alpha[5] = 1 / 160.0;
  conf.kfilter_conf.Q_std_alpha[6] = 0.0;
  conf.kfilter_conf.Q_std_alpha[7] = 1 / 160.0;
  conf.kfilter_conf.Q_std_x_idx[0] = 3;
  conf.kfilter_conf.Q_std_x_idx[1] = 3;
  conf.kfilter_conf.Q_std_x_idx[2] = -1;
  conf.kfilter_conf.Q_std_x_idx[3] = 3;
  conf.kfilter_conf.Q_std_x_idx[4] = 3;
  conf.kfilter_conf.Q_std_x_idx[5] = 3;
  conf.kfilter_conf.Q_std_x_idx[6] = -1;
  conf.kfilter_conf.Q_std_x_idx[7] = 3;
  conf.kfilter_conf.Q_std_beta[0] = 0.0;
  conf.kfilter_conf.Q_std_beta[1] = 0.0;
  conf.kfilter_conf.Q_std_beta[2] = 0.1;
  conf.kfilter_conf.Q_std_beta[3] = 0.0;
  conf.kfilter_conf.Q_std_beta[4] = 0.0;
  conf.kfilter_conf.Q_std_beta[5] = 0.0;
  conf.kfilter_conf.Q_std_beta[6] = 1e-5;
  conf.kfilter_conf.Q_std_beta[7] = 0.0;

  conf.kfilter_conf.R_std_alpha[0] = 1 / 20.0;
  conf.kfilter_conf.R_std_alpha[1] = 1 / 20.0;
  conf.kfilter_conf.R_std_alpha[2] = 0.0;
  conf.kfilter_conf.R_std_alpha[3] = 1 / 20.0;
  conf.kfilter_conf.R_std_x_idx[0] = 3;
  conf.kfilter_conf.R_std_x_idx[1] = 3;
  conf.kfilter_conf.R_std_x_idx[2] = -1;
  conf.kfilter_conf.R_std_x_idx[3] = 3;
  conf.kfilter_conf.R_std_beta[0] = 0.0;
  conf.kfilter_conf.R_std_beta[1] = 0.0;
  conf.kfilter_conf.R_std_beta[2] = 0.1;
  conf.kfilter_conf.R_std_beta[3] = 0.0;

  return conf;
}

/* DEBUG CODE*/
void DeepSORT::show_INFO_KalmanTrackers() {
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

std::vector<KalmanTracker> DeepSORT::get_Trackers_UnmatchedLastTime() const {
  std::vector<KalmanTracker> unmatched_trackers;
  for (const KalmanTracker &tracker_ : k_trackers) {
    if (tracker_.unmatched_times > 0) {
      unmatched_trackers.push_back(tracker_);
    }
  }
  return unmatched_trackers;
}

bool DeepSORT::get_Tracker_ByID(uint64_t id, KalmanTracker &tracker) const {
  for (const KalmanTracker &tracker_ : k_trackers) {
    if (tracker_.id == id) {
      tracker = tracker_;
      return true;
    }
  }
  return false;
}

std::string DeepSORT::get_TrackersInfo_UnmatchedLastTime(std::string &str_info) const {
  std::stringstream ss_info;
  std::vector<KalmanTracker> unmatched_trackers = get_Trackers_UnmatchedLastTime();
  ss_info << unmatched_trackers.size() << std::endl;
  for (size_t i = 0; i < unmatched_trackers.size(); i++) {
    KalmanTracker &tracker_ = unmatched_trackers[i];
    if (tracker_.tracker_state_ != TRACKER_STATE::ACCREDITATION) {
      assert(0);
    }
    BBOX tracker_bbox = tracker_.getBBox_TLWH();
    ss_info << tracker_.id << ",-1,-1,-1,-1," << tracker_.tracker_state_ << ",";
    ss_info << static_cast<int>(tracker_bbox(0, 0)) << "," << static_cast<int>(tracker_bbox(0, 1))
            << "," << static_cast<int>(tracker_bbox(0, 0) + tracker_bbox(0, 2)) << ","
            << static_cast<int>(tracker_bbox(0, 1) + tracker_bbox(0, 3)) << "\n";
  }

  str_info = ss_info.str();
  return str_info;
}

static void show_deepsort_config(cvai_deepsort_config_t &ds_conf) {
  std::cout << "[DeepSORT] Max Distance Consine : " << ds_conf.max_distance_consine << std::endl;
  std::cout << "[DeepSORT] Max Distance IoU     : " << ds_conf.max_distance_iou << std::endl;
  std::cout << "[DeepSORT] Max Unmatched Times for BBox Matching : "
            << ds_conf.max_unmatched_times_for_bbox_matching << std::endl;
  std::cout << "[Kalman Tracker] Max Unmatched Num       : "
            << ds_conf.ktracker_conf.max_unmatched_num << std::endl;
  std::cout << "[Kalman Tracker] Accreditation Threshold : "
            << ds_conf.ktracker_conf.accreditation_threshold << std::endl;
  std::cout << "[Kalman Tracker] Feature Budget Size     : "
            << ds_conf.ktracker_conf.feature_budget_size << std::endl;
  std::cout << "[Kalman Tracker] Feature Update Interval : "
            << ds_conf.ktracker_conf.feature_update_interval << std::endl;
  std::cout << "[Kalman Tracker] P-alpha : " << std::endl;
  std::cout << std::setw(6) << ds_conf.ktracker_conf.P_std_alpha[0];
  for (int i = 1; i < 8; i++) {
    std::cout << "," << std::setw(6) << ds_conf.ktracker_conf.P_std_alpha[i];
  }
  std::cout << std::endl;
  std::cout << "[Kalman Tracker] P-beta : " << std::endl;
  std::cout << std::setw(6) << ds_conf.ktracker_conf.P_std_beta[0];
  for (int i = 1; i < 8; i++) {
    std::cout << "," << std::setw(6) << ds_conf.ktracker_conf.P_std_beta[i];
  }
  std::cout << std::endl;
  std::cout << "[Kalman Tracker] P-x_idx : " << std::endl;
  std::cout << std::setw(6) << ds_conf.ktracker_conf.P_std_x_idx[0];
  for (int i = 1; i < 8; i++) {
    std::cout << "," << std::setw(6) << ds_conf.ktracker_conf.P_std_x_idx[i];
  }
  std::cout << std::endl;

  std::cout << "[Kalman Filter] Q-alpha : " << std::endl;
  std::cout << std::setw(6) << ds_conf.kfilter_conf.Q_std_alpha[0];
  for (int i = 1; i < 8; i++) {
    std::cout << "," << std::setw(6) << ds_conf.kfilter_conf.Q_std_alpha[i];
  }
  std::cout << std::endl;
  std::cout << "[Kalman Filter] Q-beta : " << std::endl;
  std::cout << std::setw(6) << ds_conf.kfilter_conf.Q_std_beta[0];
  for (int i = 1; i < 8; i++) {
    std::cout << "," << std::setw(6) << ds_conf.kfilter_conf.Q_std_beta[i];
  }
  std::cout << std::endl;
  std::cout << "[Kalman Filter] Q-x_idx : " << std::endl;
  std::cout << std::setw(6) << ds_conf.kfilter_conf.Q_std_x_idx[0];
  for (int i = 1; i < 8; i++) {
    std::cout << "," << std::setw(6) << ds_conf.kfilter_conf.Q_std_x_idx[i];
  }
  std::cout << std::endl;
  std::cout << "[Kalman Filter] R-alpha : " << std::endl;
  std::cout << std::setw(6) << ds_conf.kfilter_conf.R_std_alpha[0];
  for (int i = 1; i < 4; i++) {
    std::cout << "," << std::setw(6) << ds_conf.kfilter_conf.R_std_alpha[i];
  }
  std::cout << std::endl;
  std::cout << "[Kalman Filter] R-beta : " << std::endl;
  std::cout << std::setw(6) << ds_conf.kfilter_conf.R_std_beta[0];
  for (int i = 1; i < 4; i++) {
    std::cout << "," << std::setw(6) << ds_conf.kfilter_conf.R_std_beta[i];
  }
  std::cout << std::endl;
  std::cout << "[Kalman Filter] R-x_idx : " << std::endl;
  std::cout << std::setw(6) << ds_conf.kfilter_conf.R_std_x_idx[0];
  for (int i = 1; i < 4; i++) {
    std::cout << "," << std::setw(6) << ds_conf.kfilter_conf.R_std_x_idx[i];
  }
  std::cout << std::endl;
}