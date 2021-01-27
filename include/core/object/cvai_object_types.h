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

/** @struct cvai_pose17_meta_t
 * @ingroup core_cviaicore
 * @brief A structure to describe person pose.
 *
 * @var cvai_pose17_meta_t::x
 * Position x point.
 * @var cvai_pose17_meta_t::y
 * Position y point.
 * @var cvai_pose17_meta_t::score
 * Point score
 *
 * @see cvai_object_t
 */
typedef struct {
  float x[17];
  float y[17];
  float score[17];
} cvai_pose17_meta_t;

/** @struct cvai_vehicle_meta
 * @ingroup core_cviaicore
 * @brief A structure to describe a vehicle properity.
 * @var cvai_vehicle_meta::license_pts
 * The license plate 4 corner points.
 * @var cvai_vehicle_meta::license_bbox
 * The license bounding box.
 * @var cvai_vehicle_meta::license_char
 * The license characters
 * @see cvai_4_pts_t
 * @see cvai_bbox_t
 * @see cvai_object_info_t
 */
typedef struct {
  cvai_4_pts_t license_pts;
  cvai_bbox_t license_bbox;
  char license_char[255];
} cvai_vehicle_meta;

/** @struct cvai_pedestrian_meta
 * @ingroup core_cviaicore
 * @brief A structure to describe a pedestrian properity.
 * @var cvai_pedestrian_meta::pose_17
 * The Person 17 keypoints detected by pose estimation models
 * @var cvai_pedestrian_meta::fall
 * Whether people is fall or not
 * @see cvai_pose17_meta_t
 * @see cvai_object_info_t
 */
typedef struct {
  cvai_pose17_meta_t pose_17;
  bool fall;
} cvai_pedestrian_meta;

/** @struct cvai_object_info_t
 * @ingroup core_cviaicore
 * @brief A structure to describe a found object.
 *
 * @var cvai_object_info_t::name
 * A human readable class name.
 * @var cvai_object_info_t::unique_id
 * The unique id of an object.
 * @var cvai_object_info_t::bbox
 * The bounding box of an object.
 * @var cvai_object_info_t::bpts
 * The bounding points of an object. (Deprecated)
 * @var cvai_object_info_t::feature
 * The feature describing an object.
 * @var cvai_object_info_t::classes
 * The class label of an object.
 * @var cvai_object_info_t::vehicle_properity
 * The vehicle properity
 * @var cvai_object_info_t::pedestrian_properity
 * The pedestrian properity
 * @see cvai_object_t
 * @see cvai_pedestrian_meta
 * @see cvai_vehicle_meta
 * @see cvai_bbox_t
 * @see cvai_pts_t
 * @see cvai_feature_t
 */
typedef struct {
  char name[128];
  uint64_t unique_id;
  cvai_bbox_t bbox;
  cvai_pts_t bpts;
  cvai_feature_t feature;
  int classes;

  cvai_vehicle_meta *vehicle_properity;
  cvai_pedestrian_meta *pedestrian_properity;
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

/** @struct cvai_class_filter_t
 *  @ingroup core_cviaicore
 *  @brief Preserve class id of model output and filter out the others. This struct can be used in
 *  Semantic Segmentation.
 *  @var cvai_class_filter_t::preserved_class_ids
 *  The class IDs to be preserved
 *  @var cvai_class_filter_t::num_preserved_classes
 *  Number of classes to be preserved
 */
typedef struct {
  uint32_t *preserved_class_ids;
  uint32_t num_preserved_classes;
} cvai_class_filter_t;

#endif