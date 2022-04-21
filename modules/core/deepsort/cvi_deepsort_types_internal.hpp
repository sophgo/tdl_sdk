#ifndef _CVI_DEEPSORT_TYPES_INTERNAL_HPP_
#define _CVI_DEEPSORT_TYPES_INTERNAL_HPP_

#include <Eigen/Eigen>
#include <vector>
#include "core/deepsort/cvai_deepsort_types.h"

typedef Eigen::Matrix<float, 1, 4> BBOX;
typedef Eigen::Matrix<float, -1, 4> BBOXES;
typedef Eigen::Matrix<float, 1, -1> FEATURE;
typedef Eigen::Matrix<float, -1, -1> FEATURES;

typedef enum {
  Feature_CosineDistance = 0,
  Kalman_MahalanobisDistance,
  BBox_IoUDistance
} cost_matrix_algo_e;

#endif /* _CVI_DEEPSORT_TYPES_INTERNAL_HPP_ */
