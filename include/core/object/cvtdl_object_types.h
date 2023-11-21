#ifndef _CVI_OBJECT_TYPES_H_
#define _CVI_OBJECT_TYPES_H_
#include <stdbool.h>
#include "core/core/cvtdl_core_types.h"

/** @enum cvtdl_obj_class_id_e
 *  @ingroup core_cvitdlcore
 *  @brief classes id of object detection output
 *
 *  A List of class id and it's group:
 *
 * ID  | Class Name       | Group
 * --- | -------------    | -------------
 * 0   | person           | person
 * 1   | bicycle          | vehicle
 * 2   | car              | vehicle
 * 3   | motorcycle       | vehicle
 * 4   | airplane         | vehicle
 * 5   | bus              | vehicle
 * 6   | train            | vehicle
 * 7   | truck            | vehicle
 * 8   | boat             | vehicle
 * 9   | traffic light    | outdoor
 * 10  | fire hydrant     | outdoor
 * 11  | street sign      | outdoor
 * 12  | stop sign        | outdoor
 * 13  | parking meter    | outdoor
 * 14  | bench            | outdoor
 * 15  | bird             | animal
 * 16  | cat              | animal
 * 17  | dog              | animal
 * 18  | horse            | animal
 * 19  | sheep            | animal
 * 20  | cow              | animal
 * 21  | elephant         | animal
 * 22  | bear             | animal
 * 23  | zebra            | animal
 * 24  | giraffe          | animal
 * 25  | hat              | accessory
 * 26  | backpack         | accessory
 * 27  | umbrella         | accessory
 * 28  | shoe             | accessory
 * 29  | eye glasses      | accessory
 * 30  | handbag          | accessory
 * 31  | tie              | accessory
 * 32  | suitcase         | accessory
 * 33  | frisbee          | sports
 * 34  | skis             | sports
 * 35  | snowboard        | sports
 * 36  | sports ball      | sports
 * 37  | kite             | sports
 * 38  | baseball bat     | sports
 * 39  | baseball glove   | sports
 * 40  | skateboard       | sports
 * 41  | surfboard        | sports
 * 42  | tennis racket    | sports
 * 43  | bottle           | kitchen
 * 44  | plate            | kitchen
 * 45  | wine glass       | kitchen
 * 46  | cup              | kitchen
 * 47  | fork             | kitchen
 * 48  | knife            | kitchen
 * 49  | spoon            | kitchen
 * 50  | bowl             | kitchen
 * 51  | banana           | food
 * 52  | apple            | food
 * 53  | sandwich         | food
 * 54  | orange           | food
 * 55  | broccoli         | food
 * 56  | carrot           | food
 * 57  | hot dog          | food
 * 58  | pizza            | food
 * 59  | donut            | food
 * 60  | cake             | food
 * 61  | chair            | furniture
 * 62  | couch            | furniture
 * 63  | potted plant     | furniture
 * 64  | bed              | furniture
 * 65  | mirror           | furniture
 * 66  | dining table     | furniture
 * 67  | window           | furniture
 * 68  | desk             | furniture
 * 69  | toilet           | furniture
 * 70  | door             | furniture
 * 71  | tv               | electronic
 * 72  | laptop           | electronic
 * 73  | mouse            | electronic
 * 74  | remote           | electronic
 * 75  | keyboard         | electronic
 * 76  | cell phone       | electronic
 * 77  | microwave        | appliance
 * 78  | oven             | appliance
 * 79  | toaster          | appliance
 * 80  | sink             | appliance
 * 81  | refrigerator     | appliance
 * 82  | blender          | appliance
 * 83  | book             | indoor
 * 84  | clock            | indoor
 * 85  | vase             | indoor
 * 86  | scissors         | indoor
 * 87  | teddy bear       | indoor
 * 88  | hair drier       | indoor
 * 89  | toothbrush       | indoor
 * 90  | hair brush       | indoor
 */
typedef enum {
  CVI_TDL_DET_TYPE_PERSON,
  CVI_TDL_DET_TYPE_BICYCLE,
  CVI_TDL_DET_TYPE_CAR,
  CVI_TDL_DET_TYPE_MOTORBIKE,
  CVI_TDL_DET_TYPE_AEROPLANE,
  CVI_TDL_DET_TYPE_BUS,
  CVI_TDL_DET_TYPE_TRAIN,
  CVI_TDL_DET_TYPE_TRUCK,
  CVI_TDL_DET_TYPE_BOAT,
  CVI_TDL_DET_TYPE_TRAFFIC_LIGHT,
  CVI_TDL_DET_TYPE_FIRE_HYDRANT,
  CVI_TDL_DET_TYPE_STREET_SIGN,
  CVI_TDL_DET_TYPE_STOP_SIGN,
  CVI_TDL_DET_TYPE_PARKING_METER,
  CVI_TDL_DET_TYPE_BENCH,
  CVI_TDL_DET_TYPE_BIRD,
  CVI_TDL_DET_TYPE_CAT,
  CVI_TDL_DET_TYPE_DOG,
  CVI_TDL_DET_TYPE_HORSE,
  CVI_TDL_DET_TYPE_SHEEP,
  CVI_TDL_DET_TYPE_COW,
  CVI_TDL_DET_TYPE_ELEPHANT,
  CVI_TDL_DET_TYPE_BEAR,
  CVI_TDL_DET_TYPE_ZEBRA,
  CVI_TDL_DET_TYPE_GIRAFFE,
  CVI_TDL_DET_TYPE_HAT,
  CVI_TDL_DET_TYPE_BACKPACK,
  CVI_TDL_DET_TYPE_UMBRELLA,
  CVI_TDL_DET_TYPE_SHOE,
  CVI_TDL_DET_TYPE_EYE_GLASSES,
  CVI_TDL_DET_TYPE_HANDBAG,
  CVI_TDL_DET_TYPE_TIE,
  CVI_TDL_DET_TYPE_SUITCASE,
  CVI_TDL_DET_TYPE_FRISBEE,
  CVI_TDL_DET_TYPE_SKIS,
  CVI_TDL_DET_TYPE_SNOWBOARD,
  CVI_TDL_DET_TYPE_SPORTS_BALL,
  CVI_TDL_DET_TYPE_KITE,
  CVI_TDL_DET_TYPE_BASEBALL_BAT,
  CVI_TDL_DET_TYPE_BASEBALL_GLOVE,
  CVI_TDL_DET_TYPE_SKATEBOARD,
  CVI_TDL_DET_TYPE_SURFBOARD,
  CVI_TDL_DET_TYPE_TENNIS_RACKET,
  CVI_TDL_DET_TYPE_BOTTLE,
  CVI_TDL_DET_TYPE_PLATE,
  CVI_TDL_DET_TYPE_WINE_GLASS,
  CVI_TDL_DET_TYPE_CUP,
  CVI_TDL_DET_TYPE_FORK,
  CVI_TDL_DET_TYPE_KNIFE,
  CVI_TDL_DET_TYPE_SPOON,
  CVI_TDL_DET_TYPE_BOWL,
  CVI_TDL_DET_TYPE_BANANA,
  CVI_TDL_DET_TYPE_APPLE,
  CVI_TDL_DET_TYPE_SANDWICH,
  CVI_TDL_DET_TYPE_ORANGE,
  CVI_TDL_DET_TYPE_BROCCOLI,
  CVI_TDL_DET_TYPE_CARROT,
  CVI_TDL_DET_TYPE_HOT_DOG,
  CVI_TDL_DET_TYPE_PIZZA,
  CVI_TDL_DET_TYPE_DONUT,
  CVI_TDL_DET_TYPE_CAKE,
  CVI_TDL_DET_TYPE_CHAIR,
  CVI_TDL_DET_TYPE_SOFA,
  CVI_TDL_DET_TYPE_POTTED_PLANT,
  CVI_TDL_DET_TYPE_BED,
  CVI_TDL_DET_TYPE_MIRROR,
  CVI_TDL_DET_TYPE_DINING_TABLE,
  CVI_TDL_DET_TYPE_WINDOW,
  CVI_TDL_DET_TYPE_DESK,
  CVI_TDL_DET_TYPE_TOILET,
  CVI_TDL_DET_TYPE_DOOR,
  CVI_TDL_DET_TYPE_TV_MONITOR,
  CVI_TDL_DET_TYPE_LAPTOP,
  CVI_TDL_DET_TYPE_MOUSE,
  CVI_TDL_DET_TYPE_REMOTE,
  CVI_TDL_DET_TYPE_KEYBOARD,
  CVI_TDL_DET_TYPE_CELL_PHONE,
  CVI_TDL_DET_TYPE_MICROWAVE,
  CVI_TDL_DET_TYPE_OVEN,
  CVI_TDL_DET_TYPE_TOASTER,
  CVI_TDL_DET_TYPE_SINK,
  CVI_TDL_DET_TYPE_REFRIGERATOR,
  CVI_TDL_DET_TYPE_BLENDER,
  CVI_TDL_DET_TYPE_BOOK,
  CVI_TDL_DET_TYPE_CLOCK,
  CVI_TDL_DET_TYPE_VASE,
  CVI_TDL_DET_TYPE_SCISSORS,
  CVI_TDL_DET_TYPE_TEDDY_BEAR,
  CVI_TDL_DET_TYPE_HAIR_DRIER,
  CVI_TDL_DET_TYPE_TOOTHBRUSH,
  CVI_TDL_DET_TYPE_HAIR_BRUSH,
  CVI_TDL_DET_TYPE_END,
} cvtdl_obj_class_id_e;

/** @enum cvtdl_obj_det_group_type_e
 *  @ingroup core_cvitdlcore
 *  @brief group id for detection classes
 *
 * group id bit format:
 *    1 bits          15 bits            16 bits
 * -----------------------------------------------------
 * | mask bit |  start of class id  |  end of class id |
 * -----------------------------------------------------
 */
#define GROUP_ID(id_start, id_end) (1 << 31) | (id_start << 16) | id_end
typedef enum {
  CVI_TDL_DET_GROUP_ALL = GROUP_ID(CVI_TDL_DET_TYPE_PERSON, CVI_TDL_DET_TYPE_HAIR_BRUSH),
  CVI_TDL_DET_GROUP_PERSON = GROUP_ID(CVI_TDL_DET_TYPE_PERSON, CVI_TDL_DET_TYPE_PERSON),
  CVI_TDL_DET_GROUP_VEHICLE = GROUP_ID(CVI_TDL_DET_TYPE_BICYCLE, CVI_TDL_DET_TYPE_BOAT),
  CVI_TDL_DET_GROUP_OUTDOOR = GROUP_ID(CVI_TDL_DET_TYPE_TRAFFIC_LIGHT, CVI_TDL_DET_TYPE_BENCH),
  CVI_TDL_DET_GROUP_ANIMAL = GROUP_ID(CVI_TDL_DET_TYPE_BIRD, CVI_TDL_DET_TYPE_GIRAFFE),
  CVI_TDL_DET_GROUP_ACCESSORY = GROUP_ID(CVI_TDL_DET_TYPE_HAT, CVI_TDL_DET_TYPE_SUITCASE),
  CVI_TDL_DET_GROUP_SPORTS = GROUP_ID(CVI_TDL_DET_TYPE_FRISBEE, CVI_TDL_DET_TYPE_TENNIS_RACKET),
  CVI_TDL_DET_GROUP_KITCHEN = GROUP_ID(CVI_TDL_DET_TYPE_BOTTLE, CVI_TDL_DET_TYPE_BOWL),
  CVI_TDL_DET_GROUP_FOOD = GROUP_ID(CVI_TDL_DET_TYPE_BANANA, CVI_TDL_DET_TYPE_CAKE),
  CVI_TDL_DET_GROUP_FURNITURE = GROUP_ID(CVI_TDL_DET_TYPE_CHAIR, CVI_TDL_DET_TYPE_DOOR),
  CVI_TDL_DET_GROUP_ELECTRONIC = GROUP_ID(CVI_TDL_DET_TYPE_TV_MONITOR, CVI_TDL_DET_TYPE_CELL_PHONE),
  CVI_TDL_DET_GROUP_APPLIANCE = GROUP_ID(CVI_TDL_DET_TYPE_MICROWAVE, CVI_TDL_DET_TYPE_BLENDER),
  CVI_TDL_DET_GROUP_INDOOR = GROUP_ID(CVI_TDL_DET_TYPE_BOOK, CVI_TDL_DET_TYPE_HAIR_BRUSH),

  CVI_TDL_DET_GROUP_MASK_HEAD = 0x1 << 31,
  CVI_TDL_DET_GROUP_MASK_START = 0x7FFF << 16,
  CVI_TDL_DET_GROUP_MASK_END = 0xFFFF,
} cvtdl_obj_det_group_type_e;

#undef GROUP_ID

/** @struct cvtdl_pose17_meta_t
 * @ingroup core_cvitdlcore
 * @brief A structure to describe person pose.
 *
 * @var cvtdl_pose17_meta_t::x
 * Position x point.
 * @var cvtdl_pose17_meta_t::y
 * Position y point.
 * @var cvtdl_pose17_meta_t::score
 * Point score
 *
 * @see cvtdl_object_t
 */
typedef struct {
  float x[17];
  float y[17];
  float score[17];
} cvtdl_pose17_meta_t;

/** @struct cvtdl_vehicle_meta
 * @ingroup core_cvitdlcore
 * @brief A structure to describe a vehicle properity.
 * @var cvtdl_vehicle_meta::license_pts
 * The license plate 4 corner points.
 * @var cvtdl_vehicle_meta::license_bbox
 * The license bounding box.
 * @var cvtdl_vehicle_meta::license_char
 * The license characters
 * @see cvtdl_4_pts_t
 * @see cvtdl_bbox_t
 * @see cvtdl_object_info_t
 */
typedef struct {
  cvtdl_4_pts_t license_pts;
  cvtdl_bbox_t license_bbox;
  char license_char[125];
} cvtdl_vehicle_meta;

/** @struct cvtdl_pedestrian_meta
 * @ingroup core_cvitdlcore
 * @brief A structure to describe a pedestrian properity.
 * @var cvtdl_pedestrian_meta::pose_17
 * The Person 17 keypoints detected by pose estimation models
 * @var cvtdl_pedestrian_meta::fall
 * Whether people is fall or not
 * @see cvtdl_pose17_meta_t
 * @see cvtdl_object_info_t
 */
typedef struct {
  cvtdl_pose17_meta_t pose_17;
  bool fall;
} cvtdl_pedestrian_meta;

/** @struct cvtdl_object_info_t
 * @ingroup core_cvitdlcore
 * @brief A structure to describe a found object.
 *
 * @var cvtdl_object_info_t::name
 * A human readable class name.
 * @var cvtdl_object_info_t::unique_id
 * The unique id of an object.
 * @var cvtdl_object_info_t::bbox
 * The bounding box of an object.
 * @var cvtdl_object_info_t::bpts
 * The bounding points of an object. (Deprecated)
 * @var cvtdl_object_info_t::feature
 * The feature describing an object.
 * @var cvtdl_object_info_t::classes
 * The class label of an object.
 * @var cvtdl_object_info_t::vehicle_properity
 * The vehicle properity
 * @var cvtdl_object_info_t::pedestrian_properity
 * The pedestrian properity
 * @see cvtdl_object_t
 * @see cvtdl_pedestrian_meta
 * @see cvtdl_vehicle_meta
 * @see cvtdl_bbox_t
 * @see cvtdl_pts_t
 * @see cvtdl_feature_t
 */
typedef struct {
  char name[128];
  uint64_t unique_id;
  cvtdl_bbox_t bbox;
  int is_cross;
  cvtdl_feature_t feature;
  int classes;
  cvtdl_vehicle_meta *vehicle_properity;
  cvtdl_pedestrian_meta *pedestrian_properity;
  int track_state;
  // float human_angle;
  // float aspect_ratio;
  // float speed;
  // int is_moving;
  // int status;
} cvtdl_object_info_t;

/** @struct cvtdl_object_t
 *  @ingroup core_cvitdlcore
 *  @brief The data structure for storing object meta.
 *
 *  @var cvtdl_object_t::size
 *  The size of the info.
 *  @var cvtdl_object_t::width
 *  The current width. Affects the coordinate recovery of bbox.
 *  @var cvtdl_object_t::height
 *  The current height. Affects the coordinate recovery of bbox.
 *  @var cvtdl_object_t::info
 *  The information of each object.
 *
 *  @see cvtdl_object_info_t
 */
typedef struct {
  uint32_t size;
  uint32_t width;
  uint32_t height;

  // consumer counting
  uint32_t entry_num;
  uint32_t miss_num;
  meta_rescale_type_e rescale_type;
  cvtdl_object_info_t *info;
} cvtdl_object_t;

// consumer line
typedef struct {
  float A_x;
  float A_y;
  float B_x;
  float B_y;
  statistics_mode s_mode;
} cvtdl_counting_line_t;
// consumer counting buffer rectangle
typedef struct {
  float lt_x, lt_y;

  float rt_x, rt_y;

  float lb_x, lb_y;

  float rb_x, rb_y;

  float f_x, f_y;

  float a_x, a_y;
  float b_x, b_y;
  float k, b;
} randomRect;

/** @struct cvtdl_class_filter_t
 *  @ingroup core_cvitdlcore
 *  @brief Preserve class id of model output and filter out the others. This struct can be used in
 *  Semantic Segmentation.
 *  @var cvtdl_class_filter_t::preserved_class_ids
 *  The class IDs to be preserved
 *  @var cvtdl_class_filter_t::num_preserved_classes
 *  Number of classes to be preserved
 */
typedef struct {
  uint32_t *preserved_class_ids;
  uint32_t num_preserved_classes;
} cvtdl_class_filter_t;

/** @struct cvtdl_handpose21_meta_t
 * @ingroup core_cvitdlcore
 * @brief A structure to describe hand keypoint.
 *
 * @var cvtdl_handpose21_meta_t::x
 * Normalized x point.
 * @var cvtdl_handpose21_meta_t::x
 * Position x point.
 * @var cvtdl_handpose21_meta_t::y
 * Normalized y point.
 * @var cvtdl_handpose21_meta_t::y
 * Position y point.
 * @var cvtdl_handpose21_meta_t::bbox_x
 * BBox left
 * @var cvtdl_handpose21_meta_t::bbox_y
 * BBox top
 * @var cvtdl_handpose21_meta_t::bbox_w
 * BBox width
 * @var cvtdl_handpose21_meta_t::bbox_h
 * BBox height
 *
 * @see cvtdl_object_t
 */
typedef struct {
  float xn[21];
  float x[21];
  float yn[21];
  float y[21];
  float bbox_x;
  float bbox_y;
  float bbox_w;
  float bbox_h;
  int label;
  float score;
} cvtdl_handpose21_meta_t;

/** @struct cvtdl_handpose21_meta_t
 * @ingroup core_cvitdlcore
 * @brief A structure to describe hand keypoint.
 *
 * @var cvtdl_handpose21_meta_ts::info
 * The information of handpose.
 * @var cvtdl_handpose21_meta_ts::size
 * The length of info.
 * @var cvtdl_handpose21_meta_ts::width
 * The image width.
 * @var cvtdl_handpose21_meta_ts::height
 * The image height.
 *
 * @see cvtdl_object_t
 */
typedef struct {
  uint32_t size;
  uint32_t width;
  uint32_t height;
  cvtdl_handpose21_meta_t *info;
} cvtdl_handpose21_meta_ts;

/** @struct YoloPreParam
 *  @ingroup core_cvitdlcore
 *  @brief Config the yolo detection preprocess.
 *  @var YoloPreParam::factor
 *  Preprocess factor, one dimension matrix, r g b channel
 *  @var YoloPreParam::mean
 *  Preprocess mean, one dimension matrix, r g b channel
 *  @var YoloPreParam::rescale_type
 *  Preprocess config, vpss rescale type config
 *  @var YoloPreParam::keep_aspect_ratio
 *  Preprocess config  scale
 *  @var YoloPreParam:: resize_method
 *  Preprocess resize method config
 *  @var YoloPreParam::format
 *  Preprocess pixcel format config
 */
typedef struct {
  float factor[3];
  float mean[3];
  meta_rescale_type_e rescale_type;
  bool keep_aspect_ratio;
  VPSS_SCALE_COEF_E resize_method;
  PIXEL_FORMAT_E format;
} YoloPreParam;

/** @struct YoloAlgParam
 *  @ingroup core_cvitdlcore
 *  @brief Config the yolo detection algorithm parameters.
 *  @var YoloAlgParam::anchors
 *  Configure yolo model anchors
 *  @var YoloAlgParam::anchor_len
 *  Configure number of yolo model anchors
 *  @var YoloAlgParam::strides
 *  Configure yolo model strides
 *  @var YoloAlgParam::stride_len
 *  Configure number of yolo model stride
 *  @var YoloAlgParam::cls
 *  Configure the number of yolo model predict classes
 */
typedef struct {
  uint32_t *anchors;
  int anchor_len;
  uint32_t *strides;
  int stride_len;
  uint32_t cls;
} YoloAlgParam;

/** @struct VpssPreParam
 *  @ingroup core_cvitdlcore
 *  @brief Config the yolo detection preprocess.
 *  @var VpssPreParam::factor
 *  Preprocess factor, one dimension matrix, r g b channel
 *  @var VpssPreParam::mean
 *  Preprocess mean, one dimension matrix, r g b channel
 *  @var VpssPreParam::rescale_type
 *  Preprocess config, vpss rescale type config
 *  @var VpssPreParam::pad_reverse
 *  Preprocess padding config
 *  @var VpssPreParam::keep_aspect_ratio
 *  Preprocess config quantize scale
 *  @var VpssPreParam::use_crop
 *  Preprocess config, config crop
 *  @var VpssPreParam:: resize_method
 *  Preprocess resize method config
 *  @var VpssPreParam::format
 *  Preprocess pixcel format config
 */
typedef struct {
  float factor[3];
  float mean[3];
  meta_rescale_type_e rescale_type;
  bool pad_reverse;
  bool keep_aspect_ratio;
  bool use_quantize_scale;
  bool use_crop;
  VPSS_SCALE_COEF_E resize_method;
  PIXEL_FORMAT_E format;
} VpssPreParam;

/** @struct cvtdl_class_meta_t
 * @ingroup core_cvitdlcore
 * @brief A structure to class info.
 * @var cvtdl_class_meta_t::score
 * The information of top5 class score.
 * @var cvtdl_class_meta_t::cls
 * The classes of the top5 result
 */
typedef struct {
  float score[5];
  int cls[5];
} cvtdl_class_meta_t;

#endif
