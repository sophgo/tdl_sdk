#ifndef _WRAPPER_TYPE_DEF_HPP_
#define _WRAPPER_TYPE_DEF_HPP_

#include <map>
#include "model/base_model.hpp"
#include "tdl_model_def.h"
#include "tdl_model_defs.hpp"
#include "tdl_model_factory.hpp"
#include "tdl_object_def.h"
#include "tdl_sdk.h"
#include "tdl_types.h"
typedef struct {
  std::unordered_map<cvtdl_model_e, std::shared_ptr<BaseModel>> models;
  std::shared_ptr<TDLModelFactory> model_factory;
} tdl_context_t;

typedef struct {
  std::shared_ptr<BaseImage> image;
} tdl_image_context_t;

cvtdl_object_type_e convert_object_type(TDLObjectType object_type) {
  switch (object_type) {
    case TDL_OBJECT_TYPE_PERSON:
      return TDL_OBJECT_TYPE_PERSON;
  }
}

inline cvtdl_data_type_e convert_data_type(TDLDataType data_type) {
  switch (data_type) {
    case TDLDataType::INT8:
      return TDL_TYPE_INT8;
    case TDLDataType::UINT8:
      return TDL_TYPE_UINT8;
    case TDLDataType::INT16:
      return TDL_TYPE_INT16;
    case TDLDataType::UINT16:
      return TDL_TYPE_UINT16;
    case TDLDataType::INT32:
      return TDL_TYPE_INT32;
    case TDLDataType::UINT32:
      return TDL_TYPE_UINT32;
    case TDLDataType::BF16:
      return TDL_TYPE_BF16;
    case TDLDataType::FP16:
      return TDL_TYPE_FP16;
    case TDLDataType::FP32:
      return TDL_TYPE_FP32;
    default:
      return TDL_TYPE_UNKOWN;
  }
}

ModelType convert_model_type(cvtdl_model_e model_type) {
  switch (model_type) {
    case TDL_MODEL_SCRFD_FACE:
      return ModelType::SCRFD_FACE;
    case TDL_MODEL_YOLOV8N_PERSON_VEHICLE:
      return ModelType::YOLOV8N_PERSON_VEHICLE;
    case TDL_MODEL_YOLOV8N_HEAD_HARDHAT:
      return ModelType::YOLOV8N_HEAD_HARDHAT;
    case TDL_MODEL_YOLOV8N_HAND:
      return ModelType::YOLOV8N_HAND;
    case TDL_MODEL_SEG_YOLOV8_COCO80:
      return ModelType::SEG_YOLOV8_COCO80;
    case TDL_MODEL_SEG_PERSON_FACE_VEHICLE:
      return ModelType::SEG_PERSON_FACE_VEHICLE;
    case TDL_MODEL_YOLOV10_COCO80:
      return ModelType::YOLOV10_COCO80;
    case TDL_MODEL_ATTRIBUTE_FACE:
      return ModelType::ATTRIBUTE_FACE;
    case TDL_MODEL_KEYPOINT_FACE_V2:
      return ModelType::KEYPOINT_FACE_V2;
    case TDL_MODEL_KEYPOINT_HAND:
      return ModelType::KEYPOINT_HAND;
    case TDL_MODEL_KEYPOINT_LICENSE_PLATE:
      return ModelType::KEYPOINT_LICENSE_PLATE;
    case TDL_MODEL_KEYPOINT_YOLOV8_POSE_PERSON17:
      return ModelType::YOLOV8_POSE_PERSON17;
    case TDL_MODEL_KEYPOINT_SIMICC:
      return ModelType::KEYPOINT_SIMCC;
    case TDL_MODEL_FEATURE_BMFACER34:
      return ModelType::FEATURE_BMFACER34;
    case TDL_MODEL_CLS_RGBLIVENESS:
      return ModelType::CLS_RGBLIVENESS;
    case TDL_MODEL_LANE_DETECTION_LSTR:
      return ModelType::LANE_DETECTION_LSTR;
  }
  return ModelType::INVALID;
}

inline std::shared_ptr<ModelBoxLandmarkInfo> convert_face_meta(
    cvtdl_face_t *face_meta) {
  std::shared_ptr<ModelBoxLandmarkInfo> face_info =
      std::make_shared<ModelBoxLandmarkInfo>();
  for (int i = 0; i < face_meta->size; i++) {
    ObjectBoxLandmarkInfo box_landmark_info;
    box_landmark_info.x1 = face_meta->info[i].box.x1;
    box_landmark_info.y1 = face_meta->info[i].box.y1;
    box_landmark_info.x2 = face_meta->info[i].box.x2;
    box_landmark_info.y2 = face_meta->info[i].box.y2;
    box_landmark_info.score = face_meta->info[i].score;

    for (int j = 0; j < face_meta->info[i].landmarks.size; j++) {
      box_landmark_info.landmarks_x.push_back(
          face_meta->info[i].landmarks.x[j]);
      box_landmark_info.landmarks_y.push_back(
          face_meta->info[i].landmarks.y[j]);
    }
    face_info->box_landmarks.push_back(box_landmark_info);
  }
  return face_info;
}
#endif
