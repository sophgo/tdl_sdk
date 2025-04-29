#include "utils/cost_matrix_helper.hpp"
#include "utils/mot_box_helper.hpp"

CostMatrixHelper::CostMatrixHelper() {}

COST_MATRIX CostMatrixHelper::getCostMatrixFeature(
    const std::vector<std::shared_ptr<KalmanTracker>> &trackers,
    const std::vector<ObjectBoxInfo> &detections,
    const std::vector<ModelFeatureInfo> &features,
    const std::vector<int> &tracker_idx,
    const std::vector<int> &detection_idx) {
  assert(!tracker_idx.empty() && !detection_idx.empty());
  COST_MATRIX cost_m(tracker_idx.size(), detection_idx.size());
  //   uint32_t feature_size = Features[0].cols();
  //   FEATURES features_m_(BBox_IDXes.size(), feature_size);
  //   for (size_t i = 0; i < BBox_IDXes.size(); i++) {
  //     int bbox_idx = BBox_IDXes[i];
  //     FEATURE tmp_feature_ = Features[bbox_idx];
  //     if (USE_COSINE_DISTANCE_FOR_FEATURE) {
  //       normalize_feature(tmp_feature_);
  //     }
  //     features_m_.row(i) = tmp_feature_;
  //   }
  //   for (size_t i = 0; i < Tracker_IDXes.size(); i++) {
  //     int tracker_idx = Tracker_IDXes[i];
  //     const std::vector<FEATURE> &t_features =
  //     KTrackers[tracker_idx].features; assert(t_features.size() > 0);

  //     FEATURES tracker_features(t_features.size(), feature_size);
  //     for (size_t t = 0; t < t_features.size(); t++) {
  //       tracker_features.row(t) = t_features[t];
  //     }
  //     COST_MATRIX distance_m = cosine_distance(tracker_features,
  //     features_m_); ROW_VECTOR distance_v = get_min_colwise(distance_m);

  //     cost_m.row(i) = distance_v;
  //   }

  return cost_m;
}

COST_MATRIX CostMatrixHelper::getCostMatrixBBox(
    const std::vector<std::shared_ptr<KalmanTracker>> &trackers,
    const std::vector<ObjectBoxInfo> &detections,
    const std::vector<int> &tracker_idxes,
    const std::vector<int> &detection_idxes) {
  assert(!tracker_idxes.empty() && !detection_idxes.empty());
  COST_MATRIX cost_m(tracker_idxes.size(), detection_idxes.size());

  int i = 0;
  for (auto &tracker_idx : tracker_idxes) {
    int j = 0;
    for (auto &detection_idx : detection_idxes) {
      float iou = MotBoxHelper::calculateIOU(
          trackers[tracker_idx]->getBoxInfo(), detections[detection_idx]);
      cost_m(i, j) = 1 - iou;
      j++;
    }
    i++;
  }

  return cost_m;
}

// COST_MATRIX CostMatrixHelper::getCostMatrixMahalanobis(
//     const KalmanFilter &KF_, const std::vector<KalmanTracker> &trackers,
//     const std::vector<ObjectBoxInfo> &detections,
//     const std::vector<int> &tracker_idxes,
//     const std::vector<int> &detection_idxes,
//     const mot_kalman_filter_config_t &kfilter_conf, float upper_bound) {
//   COST_MATRIX cost_m(tracker_idxes.size(), detection_idxes.size());

//   K_MEASUREMENT_M measurement_bboxes(detection_idxes.size(), 4);
//   int i = 0;
//   for (auto &detection_idx : detection_idxes) {
//     measurement_bboxes.row(i) =
//         MotBoxHelper::convertToXYAH(detections[detection_idx]);
//     i++;
//   }
//   i = 0;
//   for (auto &tracker_idx : tracker_idxes) {
//     const KalmanTracker &tracker = trackers[tracker_idx];
//     ROW_VECTOR maha2_d = KF_.mahalanobis(tracker.x_, tracker.P_,
//                                          measurement_bboxes, kfilter_conf);
//     int j = 0;
//     for (int j = 0; j < maha2_d.cols(); j++) {
//       cost_m(i, j) =
//           (maha2_d(0, j) > upper_bound) ? upper_bound : maha2_d(0, j);
//       j++;
//     }
//     i++;
//   }

//   return cost_m;
// }

// void CostMatrixHelper::restrictCostMatrixMahalanobis(
//     COST_MATRIX &cost_matrix, const KalmanFilter &KF_,
//     const std::vector<KalmanTracker> &trackers,
//     const std::vector<ObjectBoxInfo> &detections,
//     const std::vector<int> &tracker_idxes,
//     const std::vector<int> &detection_idxes,
//     const mot_kalman_filter_config_t &kfilter_conf, float upper_bound) {
//   // float chi2_threshold = chi2inv95[4];
//   BBOXES measurement_bboxes(detection_idxes.size(), 4);
//   int i = 0;
//   for (auto &detection_idx : detection_idxes) {
//     measurement_bboxes.row(i) =
//         MotBoxHelper::convertToXYAH(detections[detection_idx]);
//     i++;
//   }
//   i = 0;
//   for (auto &tracker_idx : tracker_idxes) {
//     const KalmanTracker &tracker = trackers[tracker_idx];
//     ROW_VECTOR maha2_d = KF_.mahalanobis(tracker.x_, tracker.P_,
//                                          measurement_bboxes, kfilter_conf);

//     for (int j = 0; j < maha2_d.cols(); j++) {
//       if (maha2_d(0, j) > kfilter_conf.chi2_threshold) {
//         cost_matrix(i, j) = upper_bound;
//       }
//       j++;
//     }
//     i++;
//   }
// }
