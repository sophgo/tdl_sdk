#ifndef TRACKER_MOT_TYPE_DEFS_HPP
#define TRACKER_MOT_TYPE_DEFS_HPP

#include <Eigen/Eigen>
#include "common/model_output_types.hpp"

typedef Eigen::Matrix<float, 1, -1> ROW_VECTOR;
typedef Eigen::Matrix<float, -1, -1> COST_MATRIX;

#define DIM_X 8
#define DIM_Z 4

// typedef Eigen::Matrix<float, DIM_X, 1> K_VECTOR;
// typedef Eigen::Matrix<float, DIM_Z, 1> K_VECTOR_Z;
// typedef Eigen::Matrix<float, DIM_X, DIM_X> K_MATRIX;
// typedef Eigen::Matrix<float, DIM_Z, DIM_Z> K_MATRIX_Z_Z;
// typedef Eigen::Matrix<float, DIM_X, DIM_Z> K_MATRIX_X_Z;
// typedef Eigen::Matrix<float, DIM_Z, DIM_X> K_MATRIX_Z_X;

// typedef Eigen::Matrix<float, DIM_Z, 1> K_MEASUREMENT_V;
// typedef Eigen::Matrix<float, -1, DIM_Z> K_MEASUREMENT_M;
// typedef Eigen::Matrix<float, DIM_X, 1> K_STATE_V;
// typedef Eigen::Matrix<float, DIM_X, DIM_X> K_COVARIANCE_M;
// typedef Eigen::Matrix<float, DIM_X, DIM_X> K_EXTRAPOLATION_M;
// typedef Eigen::Matrix<float, DIM_Z, DIM_X> K_OBSERVATION_M;

// typedef Eigen::Matrix<float, DIM_X, DIM_X> K_PROCESS_NOISE_M;
// typedef Eigen::Matrix<float, DIM_Z, DIM_Z> K_MEASUREMENT_NOISE_M;

// typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> DETECTBOX;
// typedef Eigen::Matrix<float, -1, 4, Eigen::RowMajor> DETECTBOXSS;
// typedef Eigen::Matrix<float, 1, 128, Eigen::RowMajor> FEATURE;
// typedef Eigen::Matrix<float, Eigen::Dynamic, 128, Eigen::RowMajor> FEATURESS;
// typedef std::vector<FEATURE> FEATURESS;

// Kalmanfilter
// typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_FILTER;
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> DETECTBOX;
typedef Eigen::Matrix<float, -1, 4, Eigen::RowMajor> DETECTBOXSS;

typedef Eigen::Matrix<float, 1, 8, Eigen::RowMajor> KAL_MEAN;
typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_COVA;
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> KAL_HMEAN;
typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> KAL_HCOVA;
using KAL_DATA = std::pair<KAL_MEAN, KAL_COVA>;
using KAL_HDATA = std::pair<KAL_HMEAN, KAL_HCOVA>;

typedef struct _stCorrelateInfo {
  float offset_scale_x;
  float offset_scale_y;
  float pair_size_scale_x;
  float pair_size_scale_y;
  int votes;
  int time_since_correlated;
} stCorrelateInfo;

struct MatchResult {
  std::vector<std::pair<int, int>> matched_pairs;
  std::vector<int> unmatched_bbox_idxes;
  std::vector<int> unmatched_tracker_idxes;
};

enum class TrackCostType {
  BBOX_IOU = 0,
  FEATURE,
  MAHALANOBIS,
  IOU_MIX_FEATURE
};
#endif
