#include "tdl_model_factory.hpp"

#include <fstream>
#include "audio_classification/audio_classification.hpp"
#include "face_attribute/face_attribute_cls.hpp"
#include "face_detection/scrfd.hpp"
#include "face_landmark/face_landmark_det2.hpp"
#include "feature_extract/clip_image.hpp"
#include "feature_extract/clip_text.hpp"
#include "feature_extract/feature_extraction.hpp"
#include "image_classification/hand_keypopint_classification.hpp"
#include "image_classification/isp_image_classification.hpp"
#include "image_classification/rgb_image_classification.hpp"
#include "keypoints_detection/hand_keypoint.hpp"
#include "keypoints_detection/license_plate_keypoint.hpp"
#include "keypoints_detection/lstr_lane.hpp"
#include "keypoints_detection/simcc_pose.hpp"
#include "keypoints_detection/yolov8_pose.hpp"
#include "license_plate_recognition/license_plate_recognition.hpp"
#include "object_detection/mobiledet.hpp"
#include "object_detection/ppyoloe.hpp"
#include "object_detection/yolov10.hpp"
#include "object_detection/yolov5.hpp"
#include "object_detection/yolov6.hpp"
#include "object_detection/yolov7.hpp"
#include "object_detection/yolov8.hpp"
#include "object_detection/yolox.hpp"
#include "object_tracking/feartrack.hpp"
#include "segmentation/topformer_seg.hpp"
#include "segmentation/yolov8_seg.hpp"
#include "utils/common_utils.hpp"
#include "utils/tdl_log.hpp"
TDLModelFactory::TDLModelFactory() {
  coco_types_ = {"person",        "bicycle",      "car",
                 "motorcycle",    "airplane",     "bus",
                 "train",         "truck",        "boat",
                 "traffic light", "fire hydrant", "stop sign",
                 "parking meter", "bench",        "bird",
                 "cat",           "dog",          "horse",
                 "sheep",         "cow",          "elephant",
                 "bear",          "zebra",        "giraffe",
                 "backpack",      "umbrella",     "handbag",
                 "tie",           "suitcase",     "frisbee",
                 "skis",          "snowboard",    "sports ball",
                 "kite",          "baseball bat", "baseball glove",
                 "skateboard",    "surfboard",    "tennis racket",
                 "bottle",        "wine glass",   "cup",
                 "fork",          "knife",        "spoon",
                 "bowl",          "banana",       "apple",
                 "sandwich",      "orange",       "broccoli",
                 "carrot",        "hot dog",      "pizza",
                 "donut",         "cake",         "chair",
                 "couch",         "potted plant", "bed",
                 "dining table",  "toilet",       "tv",
                 "laptop",        "mouse",        "remote",
                 "keyboard",      "cell phone",   "microwave",
                 "oven",          "toaster",      "sink",
                 "refrigerator",  "book",         "clock",
                 "vase",          "scissors",     "teddy bear",
                 "hair drier",    "toothbrush"};
  loadModelConfig();
}

TDLModelFactory::~TDLModelFactory() {}

int32_t TDLModelFactory::loadModelConfig(const std::string &model_config_file) {
  std::string config_file = model_config_file;
  std::string parent_dir;
  if (config_file.empty()) {
    std::string so_dir = CommonUtils::getLibraryDir();
    std::string exe_dir = CommonUtils::getExecutableDir();
    if (so_dir != exe_dir) {
      parent_dir = CommonUtils::getParentDir(so_dir);
    } else {
      parent_dir =
          CommonUtils::getParentDir(CommonUtils::getParentDir(exe_dir));
    }
    config_file = parent_dir + "/configs/model/model_factory.json";
    LOGIP("input model config file is empty, load model config from %s",
          config_file.c_str());
  }
  std::ifstream inf(config_file);
  nlohmann::json json_config;
  model_config_map_.clear();
  if (!inf.is_open()) {
    LOGE("model config file not found: %s", config_file.c_str());
    return -1;
  }

  try {
    inf >> json_config;
  } catch (const nlohmann::json::parse_error &e) {
    LOGE("model config file %s parse error: %s", config_file.c_str(), e.what());
    return -1;
  }

  const auto &model_list = json_config.at("model_list");
  for (auto it = model_list.begin(); it != model_list.end(); ++it) {
    const std::string model_name = it.key();
    const nlohmann::json &info_json = it.value();
    model_config_map_[model_name] = info_json;
  }
  LOGIP("load model config from %s done,model size:%ld", config_file.c_str(),
        model_config_map_.size());
  return 0;
}
TDLModelFactory &TDLModelFactory::getInstance() {
  static TDLModelFactory instance;
  return instance;
}
std::shared_ptr<BaseModel> TDLModelFactory::getModel(const ModelType model_type,
                                                     const int device_id) {
  if (model_type == ModelType::INVALID) {
    LOGE("model type not found for model type: %d",
         static_cast<int>(model_type));
    return nullptr;
  }
  std::string model_name = modelTypeToString(model_type);
  if (model_config_map_.find(model_name) == model_config_map_.end()) {
    LOGE("model path not found for model type: %s,model size:%ld",
         model_name.c_str(), model_config_map_.size());
    return nullptr;
  }

  std::string model_path = getModelPath(model_type);
  if (model_path.empty()) {
    LOGE("model path not found for model type: %s",
         modelTypeToString(model_type).c_str());
    return nullptr;
  }
  ModelConfig model_config = parseModelConfig(model_config_map_[model_name]);
  return getModel(model_type, model_path, model_config, device_id);
}
std::shared_ptr<BaseModel> TDLModelFactory::getModel(
    const std::string &model_type, const int device_id) {
  ModelType model_type_enum = modelTypeFromString(model_type);
  return getModel(model_type_enum, device_id);
}
std::shared_ptr<BaseModel> TDLModelFactory::getModel(
    const ModelType model_type, const std::string &model_path,
    const int device_id) {
  ModelConfig model_config;
  std::string model_name = modelTypeToString(model_type);
  if (model_config_map_.find(model_name) != model_config_map_.end()) {
    model_config = parseModelConfig(model_config_map_[model_name]);
  }
  return getModel(model_type, model_path, model_config, device_id);
}

std::shared_ptr<BaseModel> TDLModelFactory::getModel(
    const std::string &model_type, const std::string &model_path,
    const int device_id) {
  ModelType model_type_enum = modelTypeFromString(model_type);
  return getModel(model_type_enum, model_path, device_id);
}

std::shared_ptr<BaseModel> TDLModelFactory::getModel(
    const ModelType model_type, const std::string &model_path,
    const ModelConfig &model_config, const int device_id) {
  if (model_type == ModelType::INVALID) {
    LOGE("model type not found for model type: %d",
         static_cast<int>(model_type));
    return nullptr;
  }
  std::string model_name = modelTypeToString(model_type);
  std::shared_ptr<BaseModel> model = getModelInstance(model_type);
  if (model == nullptr) {
    LOGE("model not found for model type: %d", static_cast<int>(model_type));
    return nullptr;
  }
  NetParam net_param_default = model->getNetParam();
  net_param_default.device_id = device_id;
  // merge net_param_default into  model_config
  ModelConfig model_config_merged = model_config;
  if (model_config_merged.rgb_order.empty()) {
    model_config_merged.rgb_order = net_param_default.model_config.rgb_order;
  }
  if (model_config_merged.mean.empty()) {
    model_config_merged.mean = net_param_default.model_config.mean;
  }
  if (model_config_merged.std.empty()) {
    model_config_merged.std = net_param_default.model_config.std;
  }
  net_param_default.model_config = model_config_merged;
  model->setNetParam(net_param_default);
  LOGI("model_path: %s", model_path.c_str());
  int ret = model->modelOpen(model_path);
  if (ret != 0) {
    return nullptr;
  }
  return model;
}
std::shared_ptr<BaseModel> TDLModelFactory::getModel(
    const std::string &model_type, const std::string &model_path,
    const ModelConfig &model_config, const int device_id) {
  ModelType model_type_enum = modelTypeFromString(model_type);
  if (model_type_enum == ModelType::INVALID) {
    LOGE("model type not found for model type: %s", model_type.c_str());
    return nullptr;
  }
  return getModel(model_type_enum, model_path, model_config, device_id);
}
std::shared_ptr<BaseModel> TDLModelFactory::getModel(
    const std::string &model_type, const std::string &model_path,
    const std::string &model_config_json, const int device_id) {
  ModelConfig model_config;
  try {
    nlohmann::json json_config = nlohmann::json::parse(model_config_json);
    model_config = parseModelConfig(json_config);
  } catch (const std::exception &e) {
    LOGE("Failed to parse model config: %s", e.what());
    return nullptr;
  }
  return getModel(model_type, model_path, model_config, device_id);
}
ModelConfig TDLModelFactory::getModelConfig(const ModelType model_type) {
  std::string model_type_str = modelTypeToString(model_type);
  if (model_config_map_.find(model_type_str) == model_config_map_.end()) {
    LOGE("model config not found for model type: %s", model_type_str.c_str());
    return ModelConfig();
  }
  auto json_config = model_config_map_[model_type_str];
  ModelConfig model_config = parseModelConfig(json_config);
  return model_config;
}
std::shared_ptr<BaseModel> TDLModelFactory::getModelInstance(
    const ModelType model_type) {
  std::shared_ptr<BaseModel> model = nullptr;
  LOGIP("getModelInstance model_type:%s",
        modelTypeToString(model_type).c_str());
  // 1. 目标检测模型（YOLO和MobileNet系列）
  if (isObjectDetectionModel(model_type)) {
    model = createObjectDetectionModel(model_type);
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
    model = createSegmentationModel(model_type);
  }
  // 7. 特征提取模型
  else if (isFeatureExtractionModel(model_type)) {
    model = createFeatureExtractionModel(model_type);
  }
  // 8. 目标跟踪模型
  else if (isObjectTrackingModel(model_type)) {
    model = createObjectTrackingModel(model_type);
  }
  // 9. 其他模型
  else if (isOCRModel(model_type)) {
    model = createOCRModel(model_type);
  } else {
    LOGE("model type %s not supported", modelTypeToString(model_type).c_str());
    return nullptr;
  }
  return model;
}

// 判断函数实现
bool TDLModelFactory::isObjectDetectionModel(const ModelType model_type) {
  return (model_type == ModelType::YOLOV8N_DET_PERSON_VEHICLE ||
          model_type == ModelType::YOLOV8N_DET_LICENSE_PLATE ||
          model_type == ModelType::YOLOV8N_DET_HAND ||
          model_type == ModelType::YOLOV8N_DET_PET_PERSON ||
          model_type == ModelType::YOLOV8N_DET_BICYCLE_MOTOR_EBICYCLE ||
          model_type == ModelType::YOLOV8N_DET_HAND_FACE_PERSON ||
          model_type == ModelType::YOLOV8N_DET_FACE_HEAD_PERSON_PET ||
          model_type == ModelType::YOLOV8N_DET_HEAD_PERSON ||
          model_type == ModelType::YOLOV8N_DET_FIRE_SMOKE ||
          model_type == ModelType::YOLOV8N_DET_FIRE ||
          model_type == ModelType::YOLOV8N_DET_HEAD_SHOULDER ||
          model_type == ModelType::YOLOV8N_DET_TRAFFIC_LIGHT ||
          model_type == ModelType::YOLOV8N_DET_HEAD_HARDHAT ||
          model_type == ModelType::YOLOV8N_DET_MONITOR_PERSON ||
          model_type == ModelType::YOLOV8_DET_COCO80 ||
          model_type == ModelType::YOLOV10_DET_COCO80 ||
          model_type == ModelType::YOLOV7_DET_COCO80 ||
          model_type == ModelType::YOLOV6_DET_COCO80 ||
          model_type == ModelType::YOLOV5_DET_COCO80 ||
          model_type == ModelType::PPYOLOE_DET_COCO80 ||
          model_type == ModelType::YOLOX_DET_COCO80 ||
          model_type == ModelType::YOLOV8 || model_type == ModelType::YOLOV10 ||
          model_type == ModelType::YOLOV6 || model_type == ModelType::PPYOLOE ||
          model_type == ModelType::YOLOV5 || model_type == ModelType::YOLOX ||
          model_type == ModelType::YOLOV7 ||
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
          model_type == ModelType::CLS_SOUND_COMMAND_NIHAOSHIYUN ||
          model_type == ModelType::CLS_SOUND_COMMAND_NIHAOSUANNENG ||
          model_type == ModelType::CLS_SOUND_COMMAND_XIAOAIXIAOAI ||
          model_type == ModelType::CLS_ATTRIBUTE_GENDER_AGE_GLASS ||
          model_type == ModelType::CLS_ATTRIBUTE_GENDER_AGE_GLASS_MASK ||
          model_type == ModelType::CLS_ATTRIBUTE_GENDER_AGE_GLASS_EMOTION ||
          model_type == ModelType::CLS_RGBLIVENESS ||
          model_type == ModelType::CLS_YOLOV8 ||
          model_type == ModelType::CLS_ISP_SCENE ||
          model_type == ModelType::CLS_IMG);
}

bool TDLModelFactory::isSegmentationModel(const ModelType model_type) {
  return (model_type == ModelType::YOLOV8_SEG_COCO80 ||
          model_type == ModelType::YOLOV8_SEG ||
          model_type == ModelType::TOPFORMER_SEG_PERSON_FACE_VEHICLE);
}

bool TDLModelFactory::isFeatureExtractionModel(const ModelType model_type) {
  return (model_type == ModelType::FEATURE_CLIP_IMG ||
          model_type == ModelType::FEATURE_CLIP_TEXT ||
          model_type == ModelType::FEATURE_BMFACE_R34 ||
          model_type == ModelType::FEATURE_BMFACE_R50 ||
          model_type == ModelType::FEATURE_CVIFACE ||
          model_type == ModelType::FEATURE_IMG);
}

bool TDLModelFactory::isOCRModel(const ModelType model_type) {
  return (model_type == ModelType::RECOGNITION_LICENSE_PLATE);
}

bool TDLModelFactory::isObjectTrackingModel(const ModelType model_type) {
  return (model_type == ModelType::TRACKING_FEARTRACK);
}

// 创建函数实现
std::shared_ptr<BaseModel> TDLModelFactory::createObjectDetectionModel(
    const ModelType model_type) {
  std::shared_ptr<BaseModel> model = nullptr;
  int num_classes = 0;
  std::map<int, TDLObjectType> model_type_mapping;
  int model_category =
      0;  // 0:yolov8,1:yolov10,2:yolov6,3:yolov3,4:yolov5,5:yolov6,6:mbv2,7:ppyoloe,8:yolox,9:yolov7
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
  } else if (model_type == ModelType::YOLOV8N_DET_BICYCLE_MOTOR_EBICYCLE) {
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_BICYCLE;
    model_type_mapping[1] = TDLObjectType::OBJECT_TYPE_MOTORBIKE;
    model_type_mapping[2] = TDLObjectType::OBJECT_TYPE_EBICYCLE;
    num_classes = 3;
  } else if (model_type == ModelType::YOLOV8N_DET_HAND_FACE_PERSON) {
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_HAND;
    model_type_mapping[1] = TDLObjectType::OBJECT_TYPE_FACE;
    model_type_mapping[2] = TDLObjectType::OBJECT_TYPE_PERSON;
    num_classes = 3;
  } else if (model_type == ModelType::YOLOV8N_DET_FACE_HEAD_PERSON_PET) {
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_FACE;
    model_type_mapping[1] = TDLObjectType::OBJECT_TYPE_HEAD;
    model_type_mapping[2] = TDLObjectType::OBJECT_TYPE_PERSON;
    model_type_mapping[3] = TDLObjectType::OBJECT_TYPE_PET;
    num_classes = 4;
  } else if (model_type == ModelType::YOLOV8N_DET_HEAD_PERSON) {
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_HEAD;
    model_type_mapping[1] = TDLObjectType::OBJECT_TYPE_PERSON;
    num_classes = 2;
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
  } else if (model_type == ModelType::YOLOV8_DET_COCO80) {
    model_category = 0;  // YOLOV8
    num_classes = 80;
  } else if (model_type == ModelType::YOLOV10_DET_COCO80) {
    model_category = 1;  // YOLOV10
    num_classes = 80;
  } else if (model_type == ModelType::YOLOV6_DET_COCO80) {
    model_category = 2;  // YOLOV6
    num_classes = 80;
  } else if (model_type == ModelType::YOLOV5_DET_COCO80) {
    model_category = 4;  // YOLOV5
    num_classes = 80;
  } else if (model_type == ModelType::PPYOLOE_DET_COCO80) {
    model_category = 7;  // PPYOLOE
    num_classes = 80;
  } else if (model_type == ModelType::YOLOX_DET_COCO80) {
    model_category = 8;  // YOLOX
    num_classes = 80;
  } else if (model_type == ModelType::YOLOV7_DET_COCO80) {
    model_category = 9;  // YOLOX
    num_classes = 80;
  } else if (model_type == ModelType::MBV2_DET_PERSON) {
    model_category = 6;  // MobileDetV2
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_PERSON;
  } else if (model_type == ModelType::YOLOV8) {
    model_category = 0;  // YOLOV8
  } else if (model_type == ModelType::YOLOV10) {
    model_category = 1;  // YOLOV10
  } else if (model_type == ModelType::YOLOV6) {
    model_category = 2;  // YOLOV6
  } else if (model_type == ModelType::PPYOLOE) {
    model_category = 7;  // PPYOLOE
  } else if (model_type == ModelType::YOLOX) {
    model_category = 8;  // YOLOX
  } else if (model_type == ModelType::YOLOV7) {
    model_category = 9;  // YOLOX
  } else {
    LOGE("model type not supported: %d", static_cast<int>(model_type));
    return nullptr;
  }

  if (model_category == 0) {
    model = std::make_shared<YoloV8Detection>(std::make_pair(64, num_classes));
  } else if (model_category == 1) {
    model = std::make_shared<YoloV10Detection>(std::make_pair(64, num_classes));
  } else if (model_category == 2) {
    model = std::make_shared<YoloV6Detection>(std::make_pair(4, num_classes));
  } else if (model_category == 4) {
    model = std::make_shared<YoloV5Detection>(std::make_pair(4, num_classes));
  } else if (model_category == 6) {
    model = std::make_shared<MobileDetV2Detection>(
        MobileDetV2Detection::Category::pedestrian);
  } else if (model_category == 7) {
    model = std::make_shared<PPYoloEDetection>(std::make_pair(4, num_classes));
  } else if (model_category == 8) {
    model = std::make_shared<YoloXDetection>();
  } else if (model_category == 9) {
    model = std::make_shared<YoloV7Detection>(std::make_pair(4, num_classes));
  } else {
    LOGE("createObjectDetectionModel failed,model type not supported: %d",
         static_cast<int>(model_type));
    return nullptr;
  }
  LOGIP("createObjectDetectionModel success,model type:%d,category:%d",
        static_cast<int>(model_type), model_category);
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
void TDLModelFactory::setModelDir(const std::string &model_dir) {
  model_dir_ = model_dir;
  LOGIP("setModelDir success,model_dir:%s", model_dir.c_str());
}

std::shared_ptr<BaseModel> TDLModelFactory::createSegmentationModel(
    const ModelType model_type) {
  std::shared_ptr<BaseModel> model = nullptr;

  if (model_type == ModelType::YOLOV8_SEG_COCO80) {
    model = std::make_shared<YoloV8Segmentation>(std::make_tuple(64, 32, 80));
  } else if (model_type == ModelType::YOLOV8_SEG) {
    int num_cls = 0;
    model =
        std::make_shared<YoloV8Segmentation>(std::make_tuple(64, 32, num_cls));
  } else if (model_type == ModelType::TOPFORMER_SEG_PERSON_FACE_VEHICLE) {
    model = std::make_shared<TopformerSeg>(16);  // Downsampling ratio
  }

  return model;
}

std::shared_ptr<BaseModel> TDLModelFactory::createFeatureExtractionModel(
    const ModelType model_type) {
  std::shared_ptr<BaseModel> model = nullptr;

  if (model_type == ModelType::FEATURE_CLIP_IMG) {
    model = std::make_shared<Clip_Image>();
  } else if (model_type == ModelType::FEATURE_CLIP_TEXT) {
    model = std::make_shared<Clip_Text>();
  } else if (model_type == ModelType::FEATURE_BMFACE_R34 ||
             model_type == ModelType::FEATURE_BMFACE_R50 ||
             model_type == ModelType::FEATURE_CVIFACE ||
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
      model_type == ModelType::CLS_YOLOV8 || model_type == ModelType::CLS_IMG) {
    model = std::make_shared<RgbImageClassification>();
  } else if (model_type == ModelType::CLS_KEYPOINT_HAND_GESTURE) {
    model = std::make_shared<HandKeypointClassification>();
  } else if (model_type == ModelType::CLS_SOUND_BABAY_CRY) {
    model = std::make_shared<AudioClassification>();
  } else if (model_type == ModelType::CLS_SOUND_COMMAND ||
             model_type == ModelType::CLS_SOUND_COMMAND_NIHAOSHIYUN ||
             model_type == ModelType::CLS_SOUND_COMMAND_NIHAOSUANNENG ||
             model_type == ModelType::CLS_SOUND_COMMAND_XIAOAIXIAOAI) {
    model = std::make_shared<AudioClassification>();
  } else if (model_type == ModelType::CLS_ATTRIBUTE_GENDER_AGE_GLASS) {
    model = std::make_shared<FaceAttribute_CLS>(
        FaceAttributeModel::GENDER_AGE_GLASS);
  } else if (model_type == ModelType::CLS_ATTRIBUTE_GENDER_AGE_GLASS_MASK) {
    model = std::make_shared<FaceAttribute_CLS>(
        FaceAttributeModel::GENDER_AGE_GLASS_MASK);
  } else if (model_type == ModelType::CLS_ATTRIBUTE_GENDER_AGE_GLASS_EMOTION) {
    model = std::make_shared<FaceAttribute_CLS>(
        FaceAttributeModel::GENDER_AGE_GLASS_EMOTION);
  } else if (model_type == ModelType::CLS_ISP_SCENE) {
    model = std::make_shared<IspImageClassification>();
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

std::shared_ptr<BaseModel> TDLModelFactory::createObjectTrackingModel(
    const ModelType model_type) {
  std::shared_ptr<BaseModel> model = nullptr;

  if (model_type == ModelType::TRACKING_FEARTRACK) {
    model = std::make_shared<FearTrack>();
  }

  return model;
}

std::string TDLModelFactory::getModelPath(const ModelType model_type) {
  if (model_dir_ == "") {
    LOGE("model_dir not set");
    return "";
  }
  std::string model_type_str = modelTypeToString(model_type);
  if (model_config_map_.find(model_type_str) == model_config_map_.end()) {
    LOGE("model config not found for model type: %s", model_type_str.c_str());
    return "";
  }
  nlohmann::json model_config = model_config_map_[model_type_str];
  std::string model_file_name = model_config.at("file_name").get<std::string>();
  std::string model_path;
  std::string platform;
  std::string model_extension;
  getPlatformAndModelExtension(platform, model_extension);
  if (model_file_name.find(model_extension) != std::string::npos) {
    if ('/' == model_file_name[0]) {
      model_path = model_file_name;
    } else {
      model_path = model_dir_ + std::string("/") + platform + std::string("/") +
                   model_file_name;
    }
  } else {
    model_path = model_dir_ + std::string("/") + platform + std::string("/") +
                 model_file_name + std::string("_") + platform +
                 model_extension;
  }

  return model_path;
}

ModelConfig TDLModelFactory::parseModelConfig(
    const nlohmann::json &json_config) {
  ModelConfig model_config;

  for (auto it = json_config.begin(); it != json_config.end(); ++it) {
    const std::string key = it.key();
    const nlohmann::json &val = it.value();

    if (key == "_comment") {
      model_config.comment = val.get<std::string>();
      continue;
    } else if (key == "is_coco_types") {
      bool is_coco_types = val.get<bool>();
      if (is_coco_types) {
        model_config.types = coco_types_;
      }
    } else if (key == "types") {
      model_config.types = val.get<std::vector<std::string>>();
    } else if (key == "rgb_order") {
      model_config.rgb_order = val.get<std::string>();
    } else if (key == "mean") {
      model_config.mean = val.get<std::vector<float>>();
    } else if (key == "std") {
      model_config.std = val.get<std::vector<float>>();
    } else if (val.is_number_integer()) {
      model_config.custom_config_i[key] = val.get<int>();
      LOGI("model config parse int,key:%s,value:%d", key.c_str(),
           val.get<int>());
    } else if (val.is_number_float()) {
      model_config.custom_config_f[key] = val.get<float>();
      LOGI("model config parse float,key:%s,value:%f", key.c_str(),
           val.get<float>());
    } else if (val.is_string()) {
      model_config.custom_config_str[key] = val.get<std::string>();
      LOGI("model config parse string,key:%s,value:%s", key.c_str(),
           val.get<std::string>().c_str());
    } else {
      // 修复 dump() 的临时对象悬空问题：先把它存到局部变量里
      std::string dumped = val.dump();
      LOGW("model config %s : %s not supported", key.c_str(), dumped.c_str());
    }
  }

  return model_config;
}

std::vector<std::string> TDLModelFactory::getModelList() {
  std::vector<std::string> model_list;
  for (auto &item : model_config_map_) {
    model_list.push_back(item.first);
  }
  return model_list;
}

void TDLModelFactory::getPlatformAndModelExtension(
    std::string &platform, std::string &model_extension) {
#if defined(__CV180X__)
  platform = "cv180x";
  model_extension = ".cvimodel";
#elif defined(__CV181X__)
  platform = "cv181x";
  model_extension = ".cvimodel";
#elif defined(__CV182X__)
  platform = "cv182x";
  model_extension = ".cvimodel";
#elif defined(__CV184X__)
  platform = "cv184x";
  model_extension = ".bmodel";
#elif defined(__CV186X__)
  platform = "cv186x";
  model_extension = ".bmodel";
#elif defined(__BM1684__)
  platform = "bm1684";
  model_extension = ".bmodel";
#elif defined(__BM1688__)
  platform = "bm1688";
  model_extension = ".bmodel";
#elif defined(__BM1684X__)
  platform = "bm1684x";
  model_extension = ".bmodel";
#elif defined(__CMODEL_CV181X__)
  platform = "cv181x";
  model_extension = ".cvimodel";
#elif defined(__CMODEL_CV184X__)
  platform = "cv184x";
  model_extension = ".bmodel";
#else
  LOGE("platform not supported");
  assert(false);
#endif
}
