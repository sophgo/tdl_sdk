#include "tdl_model_factory.hpp"

#include "audio_classification/audio_classification.hpp"
#include "face_attribute/face_attribute_cls.hpp"
#include "face_detection/scrfd.hpp"
#include "face_landmark/face_landmark_det2.hpp"
#include "feature_extract/clip_image.hpp"
#include "feature_extract/clip_text.hpp"
#include "feature_extract/feature_extraction.hpp"
#include "image_classification/rgb_image_classification.hpp"
#include "keypoints_detection/hand_keypoint.hpp"
#include "keypoints_detection/license_plate_keypoint.hpp"
#include "keypoints_detection/lstr_lane.hpp"
#include "keypoints_detection/simcc_pose.hpp"
#include "keypoints_detection/yolov8_pose.hpp"
#include "image_classification/hand_keypopint_classification.hpp"
#include "license_plate_recognition/license_plate_recognition.hpp"
#include "object_detection/mobiledet.hpp"
#include "object_detection/yolov10.hpp"
#include "object_detection/yolov6.hpp"
#include "object_detection/yolov8.hpp"
#include "segmentation/topformer_seg.hpp"
#include "segmentation/yolov8_seg.hpp"
#include "utils/tdl_log.hpp"

TDLModelFactory::TDLModelFactory(const std::string model_dir)
    : model_dir_(model_dir + "/") {
  std::string str_ext = ".cvimodel";
#if defined(__BM168X__) || defined(__CV186X__)
  str_ext = ".bmodel";
#endif
  setModelPath(ModelType::SCRFD_DET_FACE,
               model_dir_ + "scrfd_500m_bnkps_432_768" + str_ext);
  setModelPath(ModelType::YOLOV8N_DET_PERSON_VEHICLE,
               model_dir_ + "yolov8n_384_640_person_vehicle" + str_ext);
  setModelPath(ModelType::KEYPOINT_FACE_V2,
               model_dir_ + "pipnet_mbv1_at_50ep_v8" + str_ext);
  setModelPath(ModelType::FEATURE_BMFACER34,
               model_dir_ + "bmface_r34" + str_ext);
  setModelPath(ModelType::YOLOV8N_DET_HEAD_HARDHAT,
               model_dir_ + "hardhat_detection" + str_ext);
  setModelPath(ModelType::CLS_ATTRIBUTE_FACE,
               model_dir_ + "face_attribute_cls" + str_ext);
  setModelPath(ModelType::CLS_RGBLIVENESS,
               model_dir_ + "face_anti_spoof_classification" + str_ext);
  setModelPath(ModelType::MBV2_DET_PERSON,
               model_dir_ + "mobiledetv2-pedestrian-d0-448_cv186x" + str_ext);
}

std::shared_ptr<BaseModel> TDLModelFactory::getModel(const ModelType model_type,
                                                     const int device_id) {
  if (model_path_map_.find(model_type) == model_path_map_.end()) {
    LOGE("model path not found for model type: %d", model_type);
    return nullptr;
  }
  std::string model_path = model_path_map_[model_type];

  return getModel(model_type, model_path, device_id);
}

std::shared_ptr<BaseModel> TDLModelFactory::getModel(
    const ModelType model_type, const std::string &model_path,
    const int device_id) {
  std::shared_ptr<BaseModel> model = nullptr;
  (void)device_id;
  // 先创建模型实例
  std::map<int, TDLObjectType> model_type_mapping;
  if (model_type == ModelType::SCRFD_DET_FACE) {
    model = std::make_shared<SCRFD>();
  } else if (model_type == ModelType::YOLOV8N_DET_PERSON_VEHICLE) {
    model = std::make_shared<YoloV8Detection>(std::make_pair(64, 7));
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_CAR;
    model_type_mapping[1] = TDLObjectType::OBJECT_TYPE_BUS;
    model_type_mapping[2] = TDLObjectType::OBJECT_TYPE_TRUCK;
    model_type_mapping[3] = TDLObjectType::OBJECT_RIDER_WITH_MOTORCYCLE;
    model_type_mapping[4] = TDLObjectType::OBJECT_TYPE_PERSON;
    model_type_mapping[5] = TDLObjectType::OBJECT_TYPE_BICYCLE;
    model_type_mapping[6] = TDLObjectType::OBJECT_TYPE_MOTORBIKE;
  } else if (model_type == ModelType::YOLOV8N_DET_LICENSE_PLATE) {
    model = std::make_shared<YoloV8Detection>(std::make_pair(64, 1));
  } else if (model_type == ModelType::YOLOV8N_DET_HAND) {
    model = std::make_shared<YoloV8Detection>(std::make_pair(64, 1));
  } else if (model_type == ModelType::YOLOV8N_DET_PET_PERSON) {
    model = std::make_shared<YoloV8Detection>(std::make_pair(64, 3));
  } else if (model_type == ModelType::YOLOV8N_DET_HAND_FACE_PERSON) {
    model = std::make_shared<YoloV8Detection>(std::make_pair(64, 3));
  } else if (model_type == ModelType::YOLOV8N_DET_FIRE_SMOKE) {
    model = std::make_shared<YoloV8Detection>(std::make_pair(64, 2));
  } else if (model_type == ModelType::YOLOV8N_DET_FIRE) {
    model = std::make_shared<YoloV8Detection>(std::make_pair(64, 1));
  } else if (model_type == ModelType::YOLOV8N_DET_HEAD_SHOULDER) {
    model = std::make_shared<YoloV8Detection>(std::make_pair(64, 1));
  } else if (model_type == ModelType::YOLOV8N_DET_TRAFFIC_LIGHT) {
    model = std::make_shared<YoloV8Detection>(std::make_pair(64, 5));
  } else if (model_type == ModelType::YOLOV8N_DET_HEAD_HARDHAT) {
    model = std::make_shared<YoloV8Detection>(std::make_pair(64, 2));
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_HEAD;
    model_type_mapping[1] = TDLObjectType::OBJECT_TYPE_HARD_HAT;
  } else if (model_type == ModelType::YOLOV8N_DET_MONITOR_PERSON) {
    model = std::make_shared<YoloV10Detection>(std::make_pair(64, 1));
  } else if (model_type == ModelType::YOLOV10_DET_COCO80) {
    model = std::make_shared<YoloV10Detection>(std::make_pair(64, 80));
  } else if (model_type == ModelType::YOLOV6_DET_COCO80) {
    model = std::make_shared<YoloV6Detection>(std::make_pair(4, 80));
  } else if (model_type == ModelType::MBV2_DET_PERSON) {
    model = std::make_shared<MobileDetV2Detection>(
        MobileDetV2Detection::Category::pedestrian, 0.5);

    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_PERSON;

  } else if (model_type == ModelType::KEYPOINT_FACE_V2) {
    model = std::make_shared<FaceLandmarkerDet2>();
  } else if (model_type == ModelType::CLS_ATTRIBUTE_FACE) {
    model = std::make_shared<FaceAttribute_CLS>();
  } else if (model_type == ModelType::FEATURE_BMFACER34) {
    model = std::make_shared<FeatureExtraction>();
  } else if (model_type == ModelType::CLS_RGBLIVENESS) {
    model = std::make_shared<RgbImageClassification>();
  } else if (model_type == ModelType::CLS_HAND_GESTURE) {
    model = std::make_shared<RgbImageClassification>();
  } else if (model_type == ModelType::KEYPOINT_SIMCC_PERSON17) {
    model = std::make_shared<SimccPose>();
  } else if (model_type == ModelType::KEYPOINT_HAND) {
    model = std::make_shared<HandKeypoint>();
  } else if (model_type == ModelType::KEYPOINT_LICENSE_PLATE) {
    model = std::make_shared<LicensePlateKeypoint>();
  } else if (model_type == ModelType::CLS_KEYPOINT_HAND_GESTURE) {
    model = std::make_shared<HandKeypointClassification>();
  } else if (model_type == ModelType::RECOGNITION_LICENSE_PLATE) {
    model = std::make_shared<LicensePlateRecognition>();
  } else if (model_type == ModelType::YOLOV8_SEG_COCO80) {
    model = std::make_shared<YoloV8Segmentation>(std::make_tuple(64, 32, 80));
  } else if (model_type == ModelType::KEYPOINT_YOLOV8POSE_PERSON17) {
    model = std::make_shared<YoloV8Pose>(std::make_tuple(64, 17, 1));
  } else if (model_type == ModelType::CLIP_FEATURE_IMG) {
    model = std::make_shared<Clip_Image>();
  } else if (model_type == ModelType::CLIP_FEATURE_TEXT) {
    model = std::make_shared<Clip_Text>();
  } else if (model_type == ModelType::TOPFORMER_SEG_PERSON_FACE_VEHICLE) {
    model = std::make_shared<TopformerSeg>(16);  // Downsampling ratio
  } else if (model_type == ModelType::LSTR_DET_LANE) {
    model = std::make_shared<LstrLane>();
  } else if (model_type == ModelType::CLS_SOUND_BABAY_CRY) {
    model = std::make_shared<AudioClassification>();
  } else if (model_type == ModelType::CLS_SOUND_COMMAND) {
    model = std::make_shared<AudioClassification>(std::make_pair(128, 1));
  } else {
    LOGE("model type not supported: %d", model_type);
    return nullptr;
  }
  LOGI("to open model: %s", model_path.c_str());
  // 然后初始化模型
  if (model) {
    int ret = model->modelOpen(model_path);
    if (ret != 0) {
      return nullptr;
    }
    model->setTypeMapping(model_type_mapping);
  }
  return model;
}

void TDLModelFactory::setModelPath(const ModelType model_type,
                                   const std::string &model_path) {
  model_path_map_[model_type] = model_path;
}

void TDLModelFactory::setModelPathMap(
    const std::map<ModelType, std::string> &model_path_map) {
  model_path_map_ = model_path_map;
}
