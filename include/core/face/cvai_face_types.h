#ifndef _CVI_FACE_TYPES_H_
#define _CVI_FACE_TYPES_H_
#include "core/core/cvai_core_types.h"

/** @enum cvai_face_emotion_e
 *  @ingroup core_cviaicore
 *  @brief Emotion enum for attribute related AI models.
 */
typedef enum {
  EMOTION_UNKNOWN = 0,
  EMOTION_HAPPY,
  EMOTION_SURPRISE,
  EMOTION_FEAR,
  EMOTION_DISGUST,
  EMOTION_SAD,
  EMOTION_ANGER,
  EMOTION_NEUTRAL,
  EMOTION_END
} cvai_face_emotion_e;

/** @enum cvai_face_gender_e
 *  @ingroup core_cviaicore
 *  @brief Gender enum for attribute related AI models.
 */
typedef enum { GENDER_UNKNOWN = 0, GENDER_MALE, GENDER_FEMALE, GENDER_END } cvai_face_gender_e;

/** @enum cvai_face_race_e
 *  @ingroup core_cviaicore
 *  @brief Race enum for attribute related AI models.
 */
typedef enum {
  RACE_UNKNOWN = 0,
  RACE_CAUCASIAN,
  RACE_BLACK,
  RACE_ASIAN,
  RACE_END
} cvai_face_race_e;

/** @enum cvai_liveness_ir_position_e
 *  @ingroup core_cviaicore
 *  @brief Give liveness AI inference the hint the physical position of the IR camera is on the left
 * or right side of the RGB camera.
 */
typedef enum { LIVENESS_IR_LEFT = 0, LIVENESS_IR_RIGHT } cvai_liveness_ir_position_e;

/** @struct cvai_head_pose_t
 *  @ingroup core_cviaicore
 *  @brief The data structure for the head pose output.
 *
 *  @var cvai_head_pose_t::faacialUnitNormalVector
 *  The Normal vector for the face.
 *  @var cvai_head_pose_t::roll
 *  The roll angle of the head pose.
 *  @var cvai_head_pose_t::pitch
 *  The pitch angle of the head pose.
 *  @var cvai_head_pose_t::yaw
 *  The yaw angle of the head pose.
 */
typedef struct {
  float yaw;
  float pitch;
  float roll;

  // Facial normal means head direction.
  float facialUnitNormalVector[3];  // 0: x-axis, 1: y-axis, 2: z-axis
} cvai_head_pose_t;

/** @struct cvai_dms_od_info_t
 *  @ingroup core_cviaicore
 *  @brief The data structure for the dms object detection output.
 *
 *  @var cvai_dms_od_info_t::name
 *  The name for the object.
 *  @var cvai_dms_od_info_t::classes
 *  The class for the object.
 *  @var cvai_dms_od_info_t::bbox
 *  The bounding box for the object.
 */

typedef struct {
  char name[128];
  int classes;
  cvai_bbox_t bbox;
} cvai_dms_od_info_t;

/** @struct cvai_dms_od_t
 *  @ingroup core_cviaicore
 *  @brief The data structure for the dms object detection output.
 *
 *  @var cvai_dms_od_t::size
 *  The size for the objects.
 *  @var cvai_dms_od_t::width
 *  The frame width for the object detection input.
 *  @var cvai_dms_od_t::height
 *  The frame height for the object detection input.
 *  @var cvai_dms_od_t::rescale_type
 *  The rescale type for the objects.
 *  @var cvai_dms_od_t::info
 *  The info for the objects.
 */

typedef struct {
  uint32_t size;
  uint32_t width;
  uint32_t height;
  meta_rescale_type_e rescale_type;
  cvai_dms_od_info_t* info;
} cvai_dms_od_t;

/** @struct cvai_dms_t
 *  @ingroup core_cviaicore
 *  @brief The data structure for storing face meta.
 *
 *  @var cvai_dms_t::r_eye_score
 *  The right eye score.
 *  @var cvai_dms_t::l_eye_score
 *  The left eye score.
 *  @var cvai_dms_t::yawn_score
 *  The yawn score.
 *  @var cvai_dms_t::phone_score
 *  The phone score.
 *  @var cvai_dms_t::smoke_score
 *  The smoke score.
 *  @var cvai_dms_t::landmarks_106
 *  The face 106 landmarks.
 *  @var cvai_dms_t::landmarks_5
 *  The face 5 landmarks which is the same as retinaface.
 *  @var cvai_dms_t::head_pose
 *  The head pose.
 *  @var cvai_dms_t::dms_od
 *  The dms od info.
 *
 *  @see cvai_face_info_t
 */

typedef struct {
  float reye_score;
  float leye_score;
  float yawn_score;
  float phone_score;
  float smoke_score;
  cvai_pts_t landmarks_106;
  cvai_pts_t landmarks_5;
  cvai_head_pose_t head_pose;
  cvai_dms_od_t dms_od;
} cvai_dms_t;

/** @struct cvai_face_info_t
 *  @ingroup core_cviaicore
 *  @brief The data structure for storing a single face information.
 *
 *  @var cvai_face_info_t::name
 *  A human readable name.
 *  @var cvai_face_info_t::unique_id
 *  The unique id of a face.
 *  @var cvai_face_info_t::bbox
 *  The bounding box of a face. Refers to the width, height from cvai_face_t.
 *  @var cvai_face_info_t::pts
 *  The point to describe the point on the face.
 *  @var cvai_face_info_t::feature
 *  The feature to describe a face.
 *  @var cvai_face_info_t::emotion
 *  The emotion from attribute.
 *  @var cvai_face_info_t::gender
 *  The gender from attribute.
 *  @var cvai_face_info_t::race
 *  The race from attribute.
 *  @var cvai_face_info_t::age
 *  The age.
 *  @var cvai_face_info_t::liveness_score
 *  The liveness score.
 *  @var cvai_face_info_t::hardhat_score
 *  The hardhat score.
 *  @var cvai_face_info_t::mask_score
 *  The mask score.
 *  @var cvai_face_info_t::face_quality
 *  The face quality.
 *  @var cvai_face_info_t::head_pose;
 *  The head pose.
 *
 *  @see cvai_face_t
 */

typedef struct {
  char name[128];
  uint64_t unique_id;
  cvai_bbox_t bbox;
  cvai_pts_t pts;
  cvai_feature_t feature;
  cvai_face_emotion_e emotion;
  cvai_face_gender_e gender;
  cvai_face_race_e race;
  float age;
  float liveness_score;
  float hardhat_score;
  float mask_score;
  float recog_score;
  float face_quality;
  float pose_score;
  float sharpness_score;
  cvai_head_pose_t head_pose;
} cvai_face_info_t;

/** @struct cvai_face_t
 *  @ingroup core_cviaicore
 *  @brief The data structure for storing face meta.
 *
 *  @var cvai_face_t::size
 *  The size of the info.
 *  @var cvai_face_t::width
 *  The current width. Affects the coordinate recovery of bbox and pts.
 *  @var cvai_face_t::height
 *  The current height. Affects the coordinate recovery of bbox and pts.
 *  @var cvai_face_t::info
 *  The information of each face.
 *  @var cvai_face_t::dms
 *  The dms of face.
 *
 *  @see cvai_face_info_t
 */
typedef struct {
  uint32_t size;
  uint32_t width;
  uint32_t height;
  meta_rescale_type_e rescale_type;
  cvai_face_info_t* info;
  cvai_dms_t* dms;
} cvai_face_t;

#endif
