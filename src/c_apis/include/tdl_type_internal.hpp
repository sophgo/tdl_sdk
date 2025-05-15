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
#include "video_decoder/video_decoder_type.hpp"
typedef struct {
  std::unordered_map<TDLModel, std::shared_ptr<BaseModel>> models;
  std::shared_ptr<TDLModelFactory> model_factory;
  std::shared_ptr<VideoDecoder> video_decoder;
} TDLContext;

typedef struct {
  std::shared_ptr<BaseImage> image;
} TDLImageContext;

inline TDLDataTypeE convertDataType(TDLDataType data_type) {
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

void TDL_convertDataTypeToCpp(TDLDataTypeE data_type,
                              size_t *type_size,
                              TDLDataType *tdl_data_type) {
  switch (data_type) {
    case TDL_TYPE_INT8:
      *type_size = sizeof(int8_t);
      *tdl_data_type = TDLDataType::INT8;
      break;
    case TDL_TYPE_UINT8:
      *type_size = sizeof(uint8_t);
      *tdl_data_type = TDLDataType::UINT8;
      break;
    case TDL_TYPE_INT16:
      *type_size = sizeof(int16_t);
      *tdl_data_type = TDLDataType::INT16;
      break;
    case TDL_TYPE_UINT16:
      *type_size = sizeof(uint16_t);
      *tdl_data_type = TDLDataType::UINT16;
      break;
    case TDL_TYPE_INT32:
      *type_size = sizeof(int32_t);
      *tdl_data_type = TDLDataType::INT32;
      break;
    case TDL_TYPE_UINT32:
      *type_size = sizeof(uint32_t);
      *tdl_data_type = TDLDataType::UINT32;
      break;
    case TDL_TYPE_BF16:
      *type_size = sizeof(uint16_t);
      *tdl_data_type = TDLDataType::BF16;
      break;
    case TDL_TYPE_FP16:
      *type_size = sizeof(uint16_t);
      *tdl_data_type = TDLDataType::FP16;
      break;
    case TDL_TYPE_FP32:
      *type_size = sizeof(float);
      *tdl_data_type = TDLDataType::FP32;
      break;
    default:
      *type_size = sizeof(uint8_t);
      *tdl_data_type = TDLDataType::UINT8;
      break;
  }
}

ModelType convertModelType(TDLModel model_type) {
  switch (model_type) {
    case TDL_MODEL_MBV2_DET_PERSON:
      return ModelType::MBV2_DET_PERSON;
    case TDL_MODEL_SCRFD_DET_FACE:
      return ModelType::SCRFD_DET_FACE;
    case TDL_MODEL_YOLOV8N_DET_PERSON_VEHICLE:
      return ModelType::YOLOV8N_DET_PERSON_VEHICLE;
    case TDL_MODEL_YOLOV8N_DET_HAND:
      return ModelType::YOLOV8N_DET_HAND;
    case TDL_MODEL_YOLOV8N_DET_LICENSE_PLATE:
      return ModelType::YOLOV8N_DET_LICENSE_PLATE;
    case TDL_MODEL_YOLOV5_DET_COCO80:
      return ModelType::YOLOV5_DET_COCO80;
    case TDL_MODEL_YOLOV6_DET_COCO80:
      return ModelType::YOLOV6_DET_COCO80;
    case TDL_MODEL_YOLOV8_DET_COCO80:
      return ModelType::YOLOV8_DET_COCO80;
    case TDL_MODEL_PPYOLOE_DET_COCO80:
    return ModelType::PPYOLOE_DET_COCO80;
    case TDL_MODEL_YOLOV8N_DET_FIRE:
      return ModelType::YOLOV8N_DET_FIRE;
    case TDL_MODEL_YOLOV8N_DET_FIRE_SMOKE:
      return ModelType::YOLOV8N_DET_FIRE_SMOKE;
    case TDL_MODEL_YOLOV8N_DET_HAND_FACE_PERSON:
      return ModelType::YOLOV8N_DET_HAND_FACE_PERSON;
    case TDL_MODEL_YOLOV8N_DET_HEAD_HARDHAT:
      return ModelType::YOLOV8N_DET_HEAD_HARDHAT;
    case TDL_MODEL_YOLOV8N_DET_HEAD_SHOULDER:
      return ModelType::YOLOV8N_DET_HEAD_SHOULDER;
    case TDL_MODEL_YOLOV8N_DET_MONITOR_PERSON:
      return ModelType::YOLOV8N_DET_MONITOR_PERSON;
    case TDL_MODEL_YOLOV8N_DET_PET_PERSON:
      return ModelType::YOLOV8N_DET_PET_PERSON;
    case TDL_MODEL_YOLOV8N_DET_TRAFFIC_LIGHT:
      return ModelType::YOLOV8N_DET_TRAFFIC_LIGHT;
    case TDL_MODEL_SEG_YOLOV8_COCO80:
      return ModelType::YOLOV8_SEG_COCO80;
    case TDL_MODEL_SEG_MOTION:
      return ModelType::TOPFORMER_SEG_MOTION;
    case TDL_MODEL_SEG_PERSON_FACE_VEHICLE:
      return ModelType::TOPFORMER_SEG_PERSON_FACE_VEHICLE;
    case TDL_MODEL_YOLOV10_DET_COCO80:
      return ModelType::YOLOV10_DET_COCO80;
    case TDL_MODEL_CLS_ATTRIBUTE_FACE:
      return ModelType::CLS_ATTRIBUTE_FACE;
    case TDL_MODEL_KEYPOINT_FACE_V2:
      return ModelType::KEYPOINT_FACE_V2;
    case TDL_MODEL_KEYPOINT_HAND:
      return ModelType::KEYPOINT_HAND;
    case TDL_MODEL_KEYPOINT_LICENSE_PLATE:
      return ModelType::KEYPOINT_LICENSE_PLATE;
    case TDL_MODEL_KEYPOINT_YOLOV8POSE_PERSON17:
      return ModelType::KEYPOINT_YOLOV8POSE_PERSON17;
    case TDL_MODEL_KEYPOINT_SIMICC:
      return ModelType::KEYPOINT_SIMCC_PERSON17;
    case TDL_MODEL_RESNET_FEATURE_BMFACE_R34:
      return ModelType::RECOGNITION_INSIGHTFACE_R34;
    case TDL_MODEL_CVIFACE:
      return ModelType::RECOGNITION_CVIFACE;
    case TDL_MODEL_CLS_RGBLIVENESS:
      return ModelType::CLS_RGBLIVENESS;
    case TDL_MODEL_CLS_HAND_GESTURE:
      return ModelType::CLS_HAND_GESTURE;
    case TDL_MODEL_CLS_KEYPOINT_HAND_GESTURE:
      return ModelType::CLS_KEYPOINT_HAND_GESTURE;
    case TDL_MODEL_LSTR_DET_LANE:
      return ModelType::LSTR_DET_LANE;
    case TDL_MODEL_CLS_BABAY_CRY:
      return ModelType::CLS_SOUND_BABAY_CRY;
    case TDL_MODEL_CLS_SOUND_COMMAND:
      return ModelType::CLS_SOUND_COMMAND;
    case TDL_MODEL_RECOGNITION_LICENSE_PLATE:
      return ModelType::RECOGNITION_LICENSE_PLATE;
  }
  return ModelType::INVALID;
}

inline std::shared_ptr<ModelBoxLandmarkInfo> convertFaceMeta(
    TDLFace *face_meta) {
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

inline std::shared_ptr<ModelBoxInfo> convertObjMeta(TDLObject *object_meta) {
  std::shared_ptr<ModelBoxInfo> obj_info = std::make_shared<ModelBoxInfo>();
  for (int i = 0; i < object_meta->size; i++) {
    ObjectBoxInfo box_landmark_info;
    box_landmark_info.x1 = object_meta->info[i].box.x1;
    box_landmark_info.y1 = object_meta->info[i].box.y1;
    box_landmark_info.x2 = object_meta->info[i].box.x2;
    box_landmark_info.y2 = object_meta->info[i].box.y2;
    box_landmark_info.score = object_meta->info[i].score;

    obj_info->bboxes.push_back(box_landmark_info);
  }
  return obj_info;
}
#endif
