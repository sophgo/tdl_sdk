#include "tdl_model_factory.hpp"

#include "face_attribute/face_attribute_cls.hpp"
#include "face_detection/scrfd.hpp"
#include "face_landmark/face_landmark_det2.hpp"
#include "feature_extract/clip_image.hpp"
#include "feature_extract/clip_text.hpp"
#include "feature_extract/feature_extraction.hpp"
#include "image_classification/rgb_image_classification.hpp"
#include "keypoints_detection/simcc_pose.hpp"
#include "keypoints_detection/yolov8_pose.hpp"
#include "keypoints_detection/hand_keypoint.hpp"
#include "object_detection/mobiledet.hpp"
#include "object_detection/yolov10.hpp"
#include "object_detection/yolov6.hpp"
#include "object_detection/yolov8.hpp"
#include "segmentation/yolov8_seg.hpp"
#include "utils/tdl_log.hpp"

TDLModelFactory::TDLModelFactory(const std::string model_dir)
    : model_dir_(model_dir + "/") {
  std::string str_ext = ".cvimodel";
#if defined(__BM168X__) || defined(__CV186X__)
  str_ext = ".bmodel";
#endif
  setModelPath(ModelType::SCRFD_FACE,
               model_dir_ + "scrfd_500m_bnkps_432_768" + str_ext);
  setModelPath(ModelType::YOLOV8N_PERSON_VEHICLE,
               model_dir_ + "yolov8n_384_640_person_vehicle" + str_ext);
  setModelPath(ModelType::KEYPOINT_FACE_V2,
               model_dir_ + "pipnet_mbv1_at_50ep_v8" + str_ext);
  setModelPath(ModelType::FEATURE_BMFACER34,
               model_dir_ + "bmface_r34" + str_ext);
  setModelPath(ModelType::YOLOV8N_HEAD_HARDHAT,
               model_dir_ + "hardhat_detection" + str_ext);
  setModelPath(ModelType::ATTRIBUTE_FACE,
               model_dir_ + "face_attribute_cls" + str_ext);
  setModelPath(ModelType::CLS_RGBLIVENESS,
               model_dir_ + "face_anti_spoof_classification" + str_ext);
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
  if (model_type == ModelType::SCRFD_FACE) {
    model = std::make_shared<SCRFD>();
  } else if (model_type == ModelType::YOLOV8N_PERSON_VEHICLE) {
    model = std::make_shared<YoloV8Detection>(std::make_pair(64, 7));
  } else if (model_type == ModelType::YOLOV8N_HAND) {
    model = std::make_shared<YoloV8Detection>(std::make_pair(64, 1));
  } else if (model_type == ModelType::YOLOV8N_HEAD_HARDHAT) {
    model = std::make_shared<YoloV8Detection>(std::make_pair(64, 2));
  } else if (model_type == ModelType::YOLOV10_COCO80) {
    model = std::make_shared<YoloV10Detection>(std::make_pair(64, 80));
  } else if (model_type == ModelType::YOLOV6_COCO80) {
    model = std::make_shared<YoloV6Detection>(std::make_pair(4, 80));
  } else if (model_type == ModelType::MBV2_PERSON) {
    model = std::make_shared<MobileDetV2Detection>(
        MobileDetV2Detection::Category::pedestrian, 0.5);
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_PERSON;

  } else if (model_type == ModelType::KEYPOINT_FACE_V2) {
    model = std::make_shared<FaceLandmarkerDet2>();
  } else if (model_type == ModelType::ATTRIBUTE_FACE) {
    model = std::make_shared<FaceAttribute_CLS>();
  } else if (model_type == ModelType::FEATURE_BMFACER34) {
    model = std::make_shared<FeatureExtraction>();
  } else if (model_type == ModelType::CLS_RGBLIVENESS) {
    model = std::make_shared<RgbImageClassification>();
  } else if (model_type == ModelType::KEYPOINT_SIMCC) {
    model = std::make_shared<SimccPose>();
  } else if (model_type == ModelType::KEYPOINT_HAND) {
    model = std::make_shared<HandKeypoint>();    
  } else if (model_type == ModelType::YOLOV8_SEG_COCO80) {
    model = std::make_shared<YoloV8Segmentation>(std::make_tuple(64, 32, 80));
  } else if (model_type == ModelType::YOLOV8_POSE_PERSON17) {
    model = std::make_shared<YoloV8Pose>(std::make_tuple(64, 17, 1));
  } else if (model_type == ModelType::IMG_FEATURE_CLIP) {
    model = std::make_shared<Clip_Image>();
  } else if (model_type == ModelType::TEXT_FEATURE_CLIP) {
    model = std::make_shared<Clip_Text>();
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
