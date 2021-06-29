#ifndef _CVIAI_SERVICE_TYPES_H_
#define _CVIAI_SERVICE_TYPES_H_

#include "core/core/cvai_core_types.h"

#include <stdint.h>

/**
 * \defgroup core_cviaiservice CVIAI Service Module
 */

/** @enum cvai_service_feature_matching_e
 *  @ingroup core_cviaiservice
 *  @brief Supported feature matching method in Service
 *
 * @var cvai_service_feature_matching_e::COS_SIMILARITY
 * Do feature matching using inner product method.
 */
typedef enum { COS_SIMILARITY } cvai_service_feature_matching_e;

/** @struct cvai_service_feature_array_t
 *  @ingroup core_cviaiservice
 *  @brief Feature array structure used in Service
 *
 * @var cvai_service_feature_array_t::ptr
 * ptr is the raw 1-D array of the feature array. Format is feature 1, feature 2...
 * @var cvai_service_feature_array_t::feature_length
 * feature length is the length of one single feature.
 * @var cvai_service_feature_array_t::data_num
 * data_num is how many features the data array has.
 * @var cvai_service_feature_array_t::type
 * type is the data type of the feature array.
 */
typedef struct {
  int8_t* ptr;
  uint32_t feature_length;
  uint32_t data_num;
  feature_type_e type;
} cvai_service_feature_array_t;

/** @enum cvai_area_detect_e
 *  @ingroup core_cviaiservice
 *  @brief The intersect state for intersect detection.
 */
typedef enum {
  UNKNOWN = 0,
  NO_INTERSECT,
  ON_LINE,
  CROSS_LINE_POS,
  CROSS_LINE_NEG,
  INSIDE_POLYGON,
  OUTSIDE_POLYGON
} cvai_area_detect_e;

/** @struct cvai_service_brush_t
 *  @ingroup core_cviaiservice
 *  @brief Brush structure for bounding box drawing
 *
 */
typedef struct {
  struct {
    float r;
    float g;
    float b;
  } color;
  uint32_t size;
} cvai_service_brush_t;

#endif  // End of _CVIAI_SERVICE_TYPES_H_