#include "cvi_kalman_tracker.hpp"
#include "cvi_deepsort_utils.hpp"

#include <math.h>
#include <iostream>

KalmanTracker::KalmanTracker() {
  // assert(0);
  id = -1;
  bbox = Eigen::MatrixXf::Zero(1, DIM_Z);
  tracker_state_ = TRACKER_STATE::MISS;

  kalman_state_ = KALMAN_STAGE::UPDATED;
  x_ = Eigen::MatrixXf::Zero(DIM_X, 1);
  P_ = Eigen::MatrixXf::Identity(DIM_X, DIM_X);
}

KalmanTracker::KalmanTracker(const uint64_t &id, const BBOX &bbox, const FEATURE &feature) {
  this->id = id;
  this->bbox = bbox;
  if (USE_COSINE_DISTANCE_FOR_FEATURE) {
    FEATURE tmp_feature = feature;
    normalize_feature(tmp_feature);
    features_.push_back(tmp_feature);
  } else {
    assert(0);
    features_.push_back(feature);
  }
  feature_update_counter = 0;

  tracker_state_ = TRACKER_STATE::PROBATION;
  matched_counter = 1;
  unmatched_times = 0;

  kalman_state_ = KALMAN_STAGE::UPDATED;
  BBOX bbox_xyah = bbox_tlwh2xyah(bbox);
  x_.block(0, 0, DIM_Z, 1) = bbox_xyah.transpose();
  x_.block(DIM_Z, 0, DIM_Z, 1) = Eigen::MatrixXf::Zero(DIM_Z, 1);
  P_ = Eigen::MatrixXf::Zero(DIM_X, DIM_X);
  for (int i = 0; i < DIM_Z; i++) {
    P_(i, i) = pow(2 * STD_XP_0 * bbox_xyah(3), 2);
  }
  for (int i = DIM_Z; i < DIM_X; i++) {
    P_(i, i) = pow(10 * STD_XP_1 * bbox_xyah(3), 2);
  }
  P_(DIM_Z - 2, DIM_Z - 2) = pow(1e-2, 2);
  P_(DIM_X - 2, DIM_X - 2) = pow(1e-5, 2);
}

KalmanTracker::KalmanTracker(const uint64_t &id, const BBOX &bbox, const FEATURE &feature,
                             const cvai_kalman_tracker_config_t &ktracker_conf) {
  this->id = id;
  this->bbox = bbox;
  if (USE_COSINE_DISTANCE_FOR_FEATURE) {
    FEATURE tmp_feature = feature;
    normalize_feature(tmp_feature);
    features_.push_back(tmp_feature);
  } else {
    assert(0);
    features_.push_back(feature);
  }
  feature_update_counter = 0;

  tracker_state_ = TRACKER_STATE::PROBATION;
  matched_counter = 1;
  unmatched_times = 0;

  kalman_state_ = KALMAN_STAGE::UPDATED;
  BBOX bbox_xyah = bbox_tlwh2xyah(bbox);
  x_.block(0, 0, DIM_Z, 1) = bbox_xyah.transpose();
  x_.block(DIM_Z, 0, DIM_Z, 1) = Eigen::MatrixXf::Zero(DIM_Z, 1);
  P_ = Eigen::MatrixXf::Zero(DIM_X, DIM_X);
  for (int i = 0; i < DIM_X; i++) {
    if (ktracker_conf.P_std_x_idx[i] != -1) {
      P_(i, i) = pow(ktracker_conf.P_std_alpha[i] * x_[ktracker_conf.P_std_x_idx[i]] +
                         ktracker_conf.P_std_beta[i],
                     2);
    } else {
      P_(i, i) = pow(ktracker_conf.P_std_beta[i], 2);
    }
  }
}

void KalmanTracker::update_bbox(const BBOX &bbox) { this->bbox = bbox; }

void KalmanTracker::update_feature(const FEATURE &feature) {
  feature_update_counter += 1;
  if (feature_update_counter >= FEATURE_UPDATE_INTERVAL) {
    if (USE_COSINE_DISTANCE_FOR_FEATURE) {
      FEATURE tmp_feature = feature;
      normalize_feature(tmp_feature);
      features_.push_back(tmp_feature);
    } else {
      assert(0);
      features_.push_back(feature);
    }
    feature_update_counter = 0;
    if (features_.size() > FEATURE_BUDGET_SIZE) {
      features_.erase(features_.begin());
    }
  }
}

void KalmanTracker::update_state(bool is_matched) {
  if (is_matched) {
    if (tracker_state_ == TRACKER_STATE::PROBATION) {
      matched_counter += 1;
      if (matched_counter >= ACCREDITATION_THRESHOLD) {
        tracker_state_ = TRACKER_STATE::ACCREDITATION;
      }
    }
    unmatched_times = 0;
  } else {
    if (tracker_state_ == TRACKER_STATE::PROBATION) {
      tracker_state_ = TRACKER_STATE::MISS;
      return;
    }
    unmatched_times += 1;
    kalman_state_ = KALMAN_STAGE::UPDATED;
    if (unmatched_times > MAX_UNMATCHED_NUM) {
      tracker_state_ = TRACKER_STATE::MISS;
    }
  }
}

COST_MATRIX KalmanTracker::getCostMatrix_Feature(const std::vector<KalmanTracker> &KTrackers,
                                                 const std::vector<BBOX> &BBoxes,
                                                 const std::vector<FEATURE> &Features,
                                                 const std::vector<int> &Tracker_IDXes,
                                                 const std::vector<int> &BBox_IDXes) {
  assert(!Tracker_IDXes.empty() && !BBox_IDXes.empty());
  COST_MATRIX cost_m(Tracker_IDXes.size(), BBox_IDXes.size());
  uint32_t feature_size = Features[0].cols();
  FEATURES features_m_(BBox_IDXes.size(), feature_size);
  for (size_t i = 0; i < BBox_IDXes.size(); i++) {
    int bbox_idx = BBox_IDXes[i];
    FEATURE tmp_feature_ = Features[bbox_idx];
    if (USE_COSINE_DISTANCE_FOR_FEATURE) {
      normalize_feature(tmp_feature_);
    }
    features_m_.row(i) = tmp_feature_;
  }
  for (size_t i = 0; i < Tracker_IDXes.size(); i++) {
    int tracker_idx = Tracker_IDXes[i];
    const std::vector<FEATURE> &t_features = KTrackers[tracker_idx].features_;
    assert(t_features.size() > 0);

    FEATURES tracker_features(t_features.size(), feature_size);
    for (size_t t = 0; t < t_features.size(); t++) {
      tracker_features.row(t) = t_features[t];
    }
    COST_MATRIX distance_m = cosine_distance(tracker_features, features_m_);
    ROW_VECTOR distance_v = get_min_colwise(distance_m);

    cost_m.row(i) = distance_v;
  }

  return cost_m;
}

COST_MATRIX KalmanTracker::getCostMatrix_BBox(const std::vector<KalmanTracker> &KTrackers,
                                              const std::vector<BBOX> &BBoxes,
                                              const std::vector<FEATURE> &Features,
                                              const std::vector<int> &Tracker_IDXes,
                                              const std::vector<int> &BBox_IDXes) {
  assert(!Tracker_IDXes.empty() && !BBox_IDXes.empty());
  COST_MATRIX cost_m(Tracker_IDXes.size(), BBox_IDXes.size());

  BBOXES bbox_m_(BBox_IDXes.size(), 4);
  for (size_t i = 0; i < BBox_IDXes.size(); i++) {
    int bbox_idx = BBox_IDXes[i];
    bbox_m_.row(i) = BBoxes[bbox_idx];
  }
  for (size_t i = 0; i < Tracker_IDXes.size(); i++) {
    int tracker_idx = Tracker_IDXes[i];
    assert(KTrackers[tracker_idx].unmatched_times <= MAX_UNMATCHED_TIMES_FOR_BBOX_MATCHING);

    BBOX tracker_bbox = KTrackers[tracker_idx].getBBox_TLWH();
    COST_VECTOR distance_v = iou_distance(tracker_bbox, bbox_m_);

    cost_m.row(i) = distance_v;
  }

  return cost_m;
}

float chi2inv95[10] = {0, 3.8415, 5.9915, 7.8147, 9.4877, 11.070, 12.592, 14.067, 15.507, 16.919};
void KalmanTracker::gateCostMatrix_Mahalanobis(COST_MATRIX &cost_matrix, const KalmanFilter &KF_,
                                               const std::vector<KalmanTracker> &K_Trackers,
                                               const std::vector<BBOX> &BBoxes,
                                               const std::vector<int> &Tracker_IDXes,
                                               const std::vector<int> &BBox_IDXes,
                                               float gate_value) {
  float chi2_threshold = chi2inv95[4];
  BBOXES measurement_bboxes(BBox_IDXes.size(), 4);
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

void KalmanTracker::gateCostMatrix_Mahalanobis(COST_MATRIX &cost_matrix, const KalmanFilter &KF_,
                                               const std::vector<KalmanTracker> &K_Trackers,
                                               const std::vector<BBOX> &BBoxes,
                                               const std::vector<int> &Tracker_IDXes,
                                               const std::vector<int> &BBox_IDXes,
                                               const cvai_kalman_filter_config_t &kfilter_conf,
                                               float gate_value) {
  float chi2_threshold = chi2inv95[4];
  BBOXES measurement_bboxes(BBox_IDXes.size(), 4);
  for (size_t i = 0; i < BBox_IDXes.size(); i++) {
    int bbox_idx = BBox_IDXes[i];
    measurement_bboxes.row(i) = bbox_tlwh2xyah(BBoxes[bbox_idx]);
  }
  for (size_t i = 0; i < Tracker_IDXes.size(); i++) {
    int tracker_idx = Tracker_IDXes[i];
    const KalmanTracker &tracker_ = K_Trackers[tracker_idx];
    ROW_VECTOR maha2_d = KF_.mahalanobis(tracker_.kalman_state_, tracker_.x_, tracker_.P_,
                                         measurement_bboxes, kfilter_conf);
    for (int j = 0; j < maha2_d.cols(); j++) {
      if (maha2_d(0, j) > chi2_threshold) {
        cost_matrix(i, j) = gate_value;
      }
    }
  }
}

BBOX KalmanTracker::getBBox_TLWH() const {
  BBOX bbox_tlwh;
  bbox_tlwh(2) = x_(2) * x_(3);  // H
  bbox_tlwh(3) = x_(3);          // W
  bbox_tlwh(0) = x_(0) - 0.5 * bbox_tlwh(2);
  bbox_tlwh(1) = x_(1) - 0.5 * bbox_tlwh(3);
  return bbox_tlwh;
}

/* DEBUG CODE */
int KalmanTracker::get_FeatureUpdateCounter() const { return feature_update_counter; }

int KalmanTracker::get_MatchedCounter() const { return matched_counter; }

std::string KalmanTracker::get_INFO_KalmanState() const {
  switch (kalman_state_) {
    case KALMAN_STAGE::UPDATED:
      return std::string("UPDATED");
    case KALMAN_STAGE::PREDICTED:
      return std::string("PREDICTED");
    default:
      assert(0);
  }
  return "ERROR";
}

std::string KalmanTracker::get_INFO_TrackerState() const {
  switch (tracker_state_) {
    case TRACKER_STATE::PROBATION:
      return std::string("PROBATION");
    case TRACKER_STATE::ACCREDITATION:
      return std::string("ACCREDITATION");
    case TRACKER_STATE::MISS:
      return std::string("MISS");
    default:
      assert(0);
  }
  return "ERROR";
}