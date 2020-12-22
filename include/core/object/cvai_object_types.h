#ifndef _CVI_OBJECT_TYPES_H_
#define _CVI_OBJECT_TYPES_H_
#include "core/core/cvai_core_types.h"

/** @enum cvai_obj_det_type_e
 *  @ingroup core_cviaicore
 *  @brief Gives the hint to object detection inference which type of object should be detected.
 */
typedef enum {
  CVI_DET_TYPE_ALL = 0,
  CVI_DET_TYPE_VEHICLE = (1 << 0),
  CVI_DET_TYPE_PEOPLE = (1 << 1),
  CVI_DET_TYPE_PET = (1 << 2)
} cvai_obj_det_type_e;

/** @struct cvai_object_info_t
 * @ingroup core_cviaicore
 * @brief A structure to describe a found object.
 *
 * @var cvai_object_info_t::name
 * A human readable name.
 * @var cvai_object_info_t::unique_id
 * The unique id of an object.
 * @var cvai_object_info_t::bbox
 * The bounding box of an object.
 * @var cvai_object_info_t::bpts
 * The bounding points of an object.
 * @var cvai_object_info_t::feature
 * The feature describing an object.
 * @var cvai_object_info_t::classes
 * The class label of an object.
 *
 * @see cvai_object_t
 */
typedef struct {
  char name[128];
  uint64_t unique_id;
  cvai_bbox_t bbox;
  cvai_pts_t bpts;
  cvai_feature_t feature;
  int classes;
} cvai_object_info_t;

/** @struct cvai_object_t
 *  @ingroup core_cviaicore
 *  @brief The data structure for storing object meta.
 *
 *  @var cvai_object_t::size
 *  The size of the info.
 *  @var cvai_object_t::width
 *  The current width. Affects the coordinate recovery of bbox.
 *  @var cvai_object_t::height
 *  The current height. Affects the coordinate recovery of bbox.
 *  @var cvai_object_t::info
 *  The information of each object.
 *
 *  @see cvai_object_info_t
 */
typedef struct {
  uint32_t size;
  uint32_t width;
  uint32_t height;
  meta_rescale_type_e rescale_type;
  cvai_object_info_t *info;
} cvai_object_t;

#endif