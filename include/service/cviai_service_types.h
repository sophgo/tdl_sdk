#ifndef _CVIAI_SERVICE_TYPES_H_
#define _CVIAI_SERVICE_TYPES_H_

#include "core/core/cvai_core_types.h"

#include <stdint.h>

/** @struct cvai_service_feature_array_t
 * @brief Feature array structure used in FR Service
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

#endif  // End of _CVIAI_SERVICE_TYPES_H_