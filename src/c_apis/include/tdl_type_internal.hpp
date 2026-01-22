#ifndef TDL_TYPE_INTERNAL_HPP
#define TDL_TYPE_INTERNAL_HPP

#include <map>
#include "app/app_task.hpp"
#include "cv/intrusion_detect/intrusion_detect.hpp"
#include "cv/motion_detect/motion_detect.hpp"
#include "encoder/image_encoder/image_encoder.hpp"
#include "model/base_model.hpp"
#include "tdl_model_def.h"
#include "tdl_model_defs.hpp"
#include "tdl_model_factory.hpp"
#include "tdl_object_def.h"
#include "tdl_sdk.h"
#include "tdl_types.h"
#include "tracker/tracker_types.hpp"
#include "video_decoder/video_decoder_type.hpp"

typedef struct {
  std::unordered_map<TDLModel, std::shared_ptr<BaseModel>> models;
  std::shared_ptr<TDLModelFactory> model_factory;
  std::shared_ptr<VideoDecoder> video_decoder;
  std::shared_ptr<MotionDetection> md;
  std::shared_ptr<AppTask> app_task;
  std::shared_ptr<ImageEncoder> encoder;
  std::shared_ptr<IntrusionDetection> intrusion_detect;
  std::shared_ptr<Tracker> tracker;
  std::shared_ptr<ModelASRInfo> asr_meta;
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
inline TDLDataType convertDataTypeE(TDLDataTypeE data_type) {
  switch (data_type) {
    case TDL_TYPE_INT8:
      return TDLDataType::INT8;
    case TDL_TYPE_UINT8:
      return TDLDataType::UINT8;
    case TDL_TYPE_INT16:
      return TDLDataType::INT16;
    case TDL_TYPE_UINT16:
      return TDLDataType::UINT16;
    case TDL_TYPE_INT32:
      return TDLDataType::INT32;
    case TDL_TYPE_UINT32:
      return TDLDataType::UINT32;
    case TDL_TYPE_BF16:
      return TDLDataType::BF16;
    case TDL_TYPE_FP16:
      return TDLDataType::FP16;
    case TDL_TYPE_FP32:
      return TDLDataType::FP32;
    default:
      return TDLDataType::UNKOWN;
  }
}

inline ImageFormatE convertImageFormat(ImageFormat format) {
  switch (format) {
    case ImageFormat::GRAY:
      return IMAGE_GRAY;
    case ImageFormat::RGB_PLANAR:
      return IMAGE_RGB_PLANAR;
    case ImageFormat::RGB_PACKED:
      return IMAGE_RGB_PACKED;
    case ImageFormat::BGR_PLANAR:
      return IMAGE_BGR_PLANAR;
    case ImageFormat::BGR_PACKED:
      return IMAGE_BGR_PACKED;
    case ImageFormat::YUV420SP_UV:
      return IMAGE_YUV420SP_UV;
    case ImageFormat::YUV420SP_VU:
      return IMAGE_YUV420SP_VU;
    case ImageFormat::YUV420P_UV:
      return IMAGE_YUV420P_UV;
    case ImageFormat::YUV420P_VU:
      return IMAGE_YUV420P_VU;
    case ImageFormat::YUV422P_UV:
      return IMAGE_YUV422P_UV;
    case ImageFormat::YUV422P_VU:
      return IMAGE_YUV422P_VU;
    case ImageFormat::YUV422SP_UV:
      return IMAGE_YUV422SP_UV;
    case ImageFormat::YUV422SP_VU:
      return IMAGE_YUV422SP_VU;
    case ImageFormat::UNKOWN:
    default:
      return IMAGE_UNKOWN;
  }
}

inline ImageFormat convertImageFormatE(ImageFormatE format) {
  switch (format) {
    case IMAGE_GRAY:
      return ImageFormat::GRAY;
    case IMAGE_RGB_PLANAR:
      return ImageFormat::RGB_PLANAR;
    case IMAGE_RGB_PACKED:
      return ImageFormat::RGB_PACKED;
    case IMAGE_BGR_PLANAR:
      return ImageFormat::BGR_PLANAR;
    case IMAGE_BGR_PACKED:
      return ImageFormat::BGR_PACKED;
    case IMAGE_YUV420SP_UV:
      return ImageFormat::YUV420SP_UV;
    case IMAGE_YUV420SP_VU:
      return ImageFormat::YUV420SP_VU;
    case IMAGE_YUV420P_UV:
      return ImageFormat::YUV420P_UV;
    case IMAGE_YUV420P_VU:
      return ImageFormat::YUV420P_VU;
    case IMAGE_YUV422P_UV:
      return ImageFormat::YUV422P_UV;
    case IMAGE_YUV422P_VU:
      return ImageFormat::YUV422P_VU;
    case IMAGE_YUV422SP_UV:
      return ImageFormat::YUV422SP_UV;
    case IMAGE_YUV422SP_VU:
      return ImageFormat::YUV422SP_VU;
    case IMAGE_UNKOWN:
    default:
      return ImageFormat::UNKOWN;
  }
}
inline ModelType convertModelType(TDLModel m) {
  switch (m) {
    // 对 MODEL_TYPE_LIST 中的每一项都展开一条 case
#define X(name, comment) \
  case TDL_MODEL_##name: \
    return ModelType::name;
    MODEL_TYPE_LIST
#undef X

    default:
      return ModelType::INVALID;
  }
}
inline std::shared_ptr<ModelBoxLandmarkInfo> convertFaceMeta(
    TDLFace *face_meta) {
  std::shared_ptr<ModelBoxLandmarkInfo> face_info =
      std::make_shared<ModelBoxLandmarkInfo>();
  for (size_t i = 0; i < face_meta->size; i++) {
    ObjectBoxLandmarkInfo box_landmark_info;
    box_landmark_info.x1 = face_meta->info[i].box.x1;
    box_landmark_info.y1 = face_meta->info[i].box.y1;
    box_landmark_info.x2 = face_meta->info[i].box.x2;
    box_landmark_info.y2 = face_meta->info[i].box.y2;
    box_landmark_info.score = face_meta->info[i].score;

    for (size_t j = 0; j < face_meta->info[i].landmarks.size; j++) {
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
  for (size_t i = 0; i < object_meta->size; i++) {
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
