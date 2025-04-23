#include "tdl_model_factory.hpp"

#include "audio_classification/audio_classification.hpp"
#include "face_attribute/face_attribute_cls.hpp"
#include "face_detection/scrfd.hpp"
#include "face_landmark/face_landmark_det2.hpp"
#include "feature_extract/clip_image.hpp"
#include "feature_extract/clip_text.hpp"
#include "feature_extract/feature_extraction.hpp"
#include "image_classification/hand_keypopint_classification.hpp"
#include "image_classification/rgb_image_classification.hpp"
#include "keypoints_detection/hand_keypoint.hpp"
#include "keypoints_detection/license_plate_keypoint.hpp"
#include "keypoints_detection/lstr_lane.hpp"
#include "keypoints_detection/simcc_pose.hpp"
#include "keypoints_detection/yolov8_pose.hpp"
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
#if defined(__BM168X__) || defined(__CV186X__) || defined(__CV184X__)
  str_ext = ".bmodel";
#endif
  setModelPath(ModelType::SCRFD_DET_FACE,
               model_dir_ + "scrfd_500m_bnkps_432_768" + str_ext);
  setModelPath(ModelType::YOLOV8N_DET_PERSON_VEHICLE,
               model_dir_ + "yolov8n_384_640_person_vehicle" + str_ext);
  setModelPath(ModelType::KEYPOINT_FACE_V2,
               model_dir_ + "pipnet_mbv1_at_50ep_v8" + str_ext);
  setModelPath(ModelType::RESNET_FEATURE_BMFACE_R34,
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

std::shared_ptr<BaseModel> TDLModelFactory::getModel(
    const ModelType model_type,
    const std::map<std::string, std::string> &config, const int device_id) {
  if (model_path_map_.find(model_type) == model_path_map_.end()) {
    LOGE("model path not found for model type: %d", model_type);
    return nullptr;
  }
  std::string model_path = model_path_map_[model_type];

  return getModel(model_type, model_path, config, device_id);
}

std::shared_ptr<BaseModel> TDLModelFactory::getModel(
    const ModelType model_type, const std::string &model_path,
    const std::map<std::string, std::string> &config, const int device_id) {
  std::shared_ptr<BaseModel> model = nullptr;
  (void)device_id;
  std::map<int, TDLObjectType> model_type_mapping;

  // 按模型类别组织代码
  // 1. 目标检测模型（YOLO和MobileNet系列）
  if (isObjectDetectionModel(model_type)) {
    model = createObjectDetectionModel(model_type, config);
  }
  // 2. 人脸检测模型
  else if (isFaceDetectionModel(model_type)) {
    model = createFaceDetectionModel(model_type);
  }
  // 3. 车道线检测模型
  else if (isLaneDetectionModel(model_type)) {
    model = createLaneDetectionModel(model_type);
  }
  // 4. 关键点检测模型
  else if (isKeypointDetectionModel(model_type)) {
    model = createKeypointDetectionModel(model_type);
  }
  // 5. 分类模型
  else if (isClassificationModel(model_type)) {
    model = createClassificationModel(model_type);
  }
  // 6. 分割模型
  else if (isSegmentationModel(model_type)) {
    model = createSegmentationModel(model_type, config);
  }
  // 7. 特征提取模型
  else if (isFeatureExtractionModel(model_type)) {
    model = createFeatureExtractionModel(model_type, config);
  }
  // 8. 其他模型
  else if (isOCRModel(model_type)) {
    model = createOCRModel(model_type);
  } else {
    LOGE("model type not supported: %d", model_type);
    return nullptr;
  }

  LOGI("to open model: %s", model_path.c_str());
  // 初始化模型
  if (model) {
    int ret = model->modelOpen(model_path);
    if (ret != 0) {
      return nullptr;
    }
  }
  return model;
}

// 判断函数实现
bool TDLModelFactory::isObjectDetectionModel(const ModelType model_type) {
  return (model_type == ModelType::YOLOV8N_DET_PERSON_VEHICLE ||
          model_type == ModelType::YOLOV8N_DET_LICENSE_PLATE ||
          model_type == ModelType::YOLOV8N_DET_HAND ||
          model_type == ModelType::YOLOV8N_DET_PET_PERSON ||
          model_type == ModelType::YOLOV8N_DET_HAND_FACE_PERSON ||
          model_type == ModelType::YOLOV8N_DET_FIRE_SMOKE ||
          model_type == ModelType::YOLOV8N_DET_FIRE ||
          model_type == ModelType::YOLOV8N_DET_HEAD_SHOULDER ||
          model_type == ModelType::YOLOV8N_DET_TRAFFIC_LIGHT ||
          model_type == ModelType::YOLOV8N_DET_HEAD_HARDHAT ||
          model_type == ModelType::YOLOV8N_DET_MONITOR_PERSON ||
          model_type == ModelType::YOLOV10_DET_COCO80 ||
          model_type == ModelType::YOLOV6_DET_COCO80 ||
          model_type == ModelType::YOLOV8 || model_type == ModelType::YOLOV10 ||
          model_type == ModelType::YOLOV3 || model_type == ModelType::YOLOV5 ||
          model_type == ModelType::YOLOV6 ||
          model_type == ModelType::MBV2_DET_PERSON);
}

bool TDLModelFactory::isFaceDetectionModel(const ModelType model_type) {
  return (model_type == ModelType::SCRFD_DET_FACE);
}

bool TDLModelFactory::isLaneDetectionModel(const ModelType model_type) {
  return (model_type == ModelType::LSTR_DET_LANE);
}

bool TDLModelFactory::isKeypointDetectionModel(const ModelType model_type) {
  return (model_type == ModelType::KEYPOINT_SIMCC_PERSON17 ||
          model_type == ModelType::KEYPOINT_HAND ||
          model_type == ModelType::KEYPOINT_LICENSE_PLATE ||
          model_type == ModelType::KEYPOINT_YOLOV8POSE_PERSON17 ||
          model_type == ModelType::KEYPOINT_FACE_V2);
}

bool TDLModelFactory::isClassificationModel(const ModelType model_type) {
  return (model_type == ModelType::CLS_HAND_GESTURE ||
          model_type == ModelType::CLS_KEYPOINT_HAND_GESTURE ||
          model_type == ModelType::CLS_SOUND_BABAY_CRY ||
          model_type == ModelType::CLS_SOUND_COMMAND ||
          model_type == ModelType::CLS_ATTRIBUTE_FACE ||
          model_type == ModelType::CLS_RGBLIVENESS ||
          model_type == ModelType::CLS_IMG);
}

bool TDLModelFactory::isSegmentationModel(const ModelType model_type) {
  return (model_type == ModelType::YOLOV8_SEG_COCO80 ||
          model_type == ModelType::YOLOV8_SEG ||
          model_type == ModelType::TOPFORMER_SEG_PERSON_FACE_VEHICLE);
}

bool TDLModelFactory::isFeatureExtractionModel(const ModelType model_type) {
  return (model_type == ModelType::CLIP_FEATURE_IMG ||
          model_type == ModelType::CLIP_FEATURE_TEXT ||
          model_type == ModelType::RESNET_FEATURE_BMFACE_R34 ||
          model_type == ModelType::RESNET_FEATURE_BMFACE_R50 ||
          model_type == ModelType::FEATURE_IMG);
}

bool TDLModelFactory::isOCRModel(const ModelType model_type) {
  return (model_type == ModelType::RECOGNITION_LICENSE_PLATE);
}

// 创建函数实现
std::shared_ptr<BaseModel> TDLModelFactory::createObjectDetectionModel(
    const ModelType model_type,

    const std::map<std::string, std::string> &config) {
  std::shared_ptr<BaseModel> model = nullptr;
  int num_classes = 0;
  std::map<int, TDLObjectType> model_type_mapping;
  int model_category =
      0;  // 0:yolov8,1:yolov10,2:yolov6,3:yolov3,4:yolov5,5:yolov6,6:mbv2
  if (model_type == ModelType::YOLOV8N_DET_PERSON_VEHICLE) {
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_CAR;
    model_type_mapping[1] = TDLObjectType::OBJECT_TYPE_BUS;
    model_type_mapping[2] = TDLObjectType::OBJECT_TYPE_TRUCK;
    model_type_mapping[3] = TDLObjectType::OBJECT_RIDER_WITH_MOTORCYCLE;
    model_type_mapping[4] = TDLObjectType::OBJECT_TYPE_PERSON;
    model_type_mapping[5] = TDLObjectType::OBJECT_TYPE_BICYCLE;
    model_type_mapping[6] = TDLObjectType::OBJECT_TYPE_MOTORBIKE;
    num_classes = 7;
  } else if (model_type == ModelType::YOLOV8N_DET_LICENSE_PLATE) {
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_LICENSE_PLATE;
    num_classes = 1;
  } else if (model_type == ModelType::YOLOV8N_DET_HAND) {
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_HAND;
    num_classes = 1;
  } else if (model_type == ModelType::YOLOV8N_DET_PET_PERSON) {
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_CAT;
    model_type_mapping[1] = TDLObjectType::OBJECT_TYPE_DOG;
    model_type_mapping[2] = TDLObjectType::OBJECT_TYPE_PERSON;
    num_classes = 3;
  } else if (model_type == ModelType::YOLOV8N_DET_HAND_FACE_PERSON) {
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_HAND;
    model_type_mapping[1] = TDLObjectType::OBJECT_TYPE_FACE;
    model_type_mapping[2] = TDLObjectType::OBJECT_TYPE_PERSON;
    num_classes = 3;
  } else if (model_type == ModelType::YOLOV8N_DET_FIRE_SMOKE) {
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_FIRE;
    model_type_mapping[1] = TDLObjectType::OBJECT_TYPE_SMOKE;
    num_classes = 2;
  } else if (model_type == ModelType::YOLOV8N_DET_FIRE) {
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_FIRE;
    num_classes = 1;
  } else if (model_type == ModelType::YOLOV8N_DET_HEAD_SHOULDER) {
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_HEAD_SHOULDER;

  } else if (model_type == ModelType::YOLOV8N_DET_TRAFFIC_LIGHT) {
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_TRAFFIC_LIGHT_RED;
    model_type_mapping[1] = TDLObjectType::OBJECT_TYPE_TRAFFIC_LIGHT_YELLOW;
    model_type_mapping[2] = TDLObjectType::OBJECT_TYPE_TRAFFIC_LIGHT_GREEN;
    model_type_mapping[3] = TDLObjectType::OBJECT_TYPE_TRAFFIC_LIGHT_OFF;
    model_type_mapping[4] = TDLObjectType::OBJECT_TYPE_TRAFFIC_LIGHT_WAIT_ON;
    num_classes = 5;
  } else if (model_type == ModelType::YOLOV8N_DET_HEAD_HARDHAT) {
    num_classes = 2;
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_HEAD;
    model_type_mapping[1] = TDLObjectType::OBJECT_TYPE_HARD_HAT;
  } else if (model_type == ModelType::YOLOV8N_DET_MONITOR_PERSON) {
    num_classes = 1;
  } else if (model_type == ModelType::YOLOV10_DET_COCO80) {
    model_category = 1;
    num_classes = 80;
  } else if (model_type == ModelType::YOLOV6_DET_COCO80) {
    model_category = 2;
    num_classes = 80;
  } else if (model_type == ModelType::MBV2_DET_PERSON) {
    model_category = 6;
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_PERSON;
  } else if (model_type == ModelType::YOLOV8) {
    if (config.find("num_cls") != config.end()) {
      num_classes = std::stoi(config.at("num_cls"));
    } else {
      printf(
          "num_cls not found in config for custom yolov8 model,would parse "
          "automatically");
    }
  } else if (model_type == ModelType::YOLOV10) {
    model_category = 1;
    if (config.find("num_cls") != config.end()) {
      num_classes = std::stoi(config.at("num_cls"));
    } else {
      printf(
          "num_cls not found in config for custom yolov8 model,would parse "
          "automatically");
    }
  } else if (model_type == ModelType::YOLOV6) {
    model_category = 2;
    if (config.find("num_cls") != config.end()) {
      num_classes = std::stoi(config.at("num_cls"));
    } else {
      printf(
          "num_cls not found in config for custom yolov6 model,would parse "
          "automatically");
    }
  } else {
    LOGE("model type not supported: %d", model_type);
    return nullptr;
  }

  if (model_category == 0) {
    model = std::make_shared<YoloV8Detection>(std::make_pair(64, num_classes));
  } else if (model_category == 1) {
    model = std::make_shared<YoloV10Detection>(std::make_pair(64, num_classes));
  } else if (model_category == 2) {
    model = std::make_shared<YoloV6Detection>(std::make_pair(64, num_classes));
  } else if (model_category == 6) {
    model = std::make_shared<MobileDetV2Detection>(MobileDetV2Detection::Category::pedestrian);
  } else {
    LOGE("model type not supported: %d", model_type);
    return nullptr;
  }

  model->setTypeMapping(model_type_mapping);

  return model;
}

std::shared_ptr<BaseModel> TDLModelFactory::createFaceDetectionModel(
    const ModelType model_type) {
  std::shared_ptr<BaseModel> model = nullptr;

  if (model_type == ModelType::SCRFD_DET_FACE) {
    model = std::make_shared<SCRFD>();
  }

  return model;
}

std::shared_ptr<BaseModel> TDLModelFactory::createLaneDetectionModel(
    const ModelType model_type) {
  std::shared_ptr<BaseModel> model = nullptr;

  if (model_type == ModelType::LSTR_DET_LANE) {
    model = std::make_shared<LstrLane>();
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

std::shared_ptr<BaseModel> TDLModelFactory::createSegmentationModel(
    const ModelType model_type,
    const std::map<std::string, std::string> &config) {
  std::shared_ptr<BaseModel> model = nullptr;

  if (model_type == ModelType::YOLOV8_SEG_COCO80) {
    model = std::make_shared<YoloV8Segmentation>(std::make_tuple(64, 32, 80));
  } else if (model_type == ModelType::YOLOV8_SEG) {
    int num_cls = 0;
    if (config.find("num_cls") != config.end()) {
      num_cls = std::stoi(config.at("num_cls"));
    } else {
      printf(
          "num_cls not found in config for custom yolov8 seg model,would parse "
          "automatically");
    }
    model =
        std::make_shared<YoloV8Segmentation>(std::make_tuple(64, 32, num_cls));
  } else if (model_type == ModelType::TOPFORMER_SEG_PERSON_FACE_VEHICLE) {
    model = std::make_shared<TopformerSeg>(16);  // Downsampling ratio
  }

  return model;
}

std::shared_ptr<BaseModel> TDLModelFactory::createFeatureExtractionModel(
    const ModelType model_type,
    const std::map<std::string, std::string> &config) {
  std::shared_ptr<BaseModel> model = nullptr;

  if (model_type == ModelType::CLIP_FEATURE_IMG) {
    model = std::make_shared<Clip_Image>();
  } else if (model_type == ModelType::CLIP_FEATURE_TEXT) {
    model = std::make_shared<Clip_Text>();
  } else if (model_type == ModelType::RESNET_FEATURE_BMFACE_R34 ||
             model_type == ModelType::RESNET_FEATURE_BMFACE_R50 ||
             model_type == ModelType::FEATURE_IMG) {
    model = std::make_shared<FeatureExtraction>();
  }

  return model;
}

std::shared_ptr<BaseModel> TDLModelFactory::createKeypointDetectionModel(
    const ModelType model_type) {
  std::shared_ptr<BaseModel> model = nullptr;

  if (model_type == ModelType::KEYPOINT_SIMCC_PERSON17) {
    model = std::make_shared<SimccPose>();
  } else if (model_type == ModelType::KEYPOINT_HAND) {
    model = std::make_shared<HandKeypoint>();
  } else if (model_type == ModelType::KEYPOINT_LICENSE_PLATE) {
    model = std::make_shared<LicensePlateKeypoint>();
  } else if (model_type == ModelType::KEYPOINT_YOLOV8POSE_PERSON17) {
    model = std::make_shared<YoloV8Pose>(std::make_tuple(64, 17, 1));
  } else if (model_type == ModelType::KEYPOINT_FACE_V2) {
    model = std::make_shared<FaceLandmarkerDet2>();
  }

  return model;
}

std::shared_ptr<BaseModel> TDLModelFactory::createClassificationModel(
    const ModelType model_type) {
  std::shared_ptr<BaseModel> model = nullptr;

  if (model_type == ModelType::CLS_HAND_GESTURE ||
      model_type == ModelType::CLS_RGBLIVENESS ||
      model_type == ModelType::CLS_IMG) {
    model = std::make_shared<RgbImageClassification>();
  } else if (model_type == ModelType::CLS_KEYPOINT_HAND_GESTURE) {
    model = std::make_shared<HandKeypointClassification>();
  } else if (model_type == ModelType::CLS_SOUND_BABAY_CRY) {
    model = std::make_shared<AudioClassification>();
  } else if (model_type == ModelType::CLS_SOUND_COMMAND) {
    model = std::make_shared<AudioClassification>(std::make_pair(128, 1));
  } else if (model_type == ModelType::CLS_ATTRIBUTE_FACE) {
    model = std::make_shared<FaceAttribute_CLS>();
  }

  return model;
}

std::shared_ptr<BaseModel> TDLModelFactory::createOCRModel(
    const ModelType model_type) {
  std::shared_ptr<BaseModel> model = nullptr;

  if (model_type == ModelType::RECOGNITION_LICENSE_PLATE) {
    model = std::make_shared<LicensePlateRecognition>();
  }

  return model;
}
