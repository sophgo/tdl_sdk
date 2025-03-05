#include "tdl_model_factory.hpp"

#include "face_attribute/face_attribute_cls.hpp"
#include "face_detection/scrfd.hpp"
#include "face_landmark/face_landmark_det2.hpp"
#include "feature_extract/feature_extraction.hpp"
#include "image_classification/rgb_image_classification.hpp"
#include "object_detection/mobiledet.hpp"
#include "object_detection/yolov10.hpp"
#include "object_detection/yolov6.hpp"
#include "object_detection/yolov8.hpp"
#include "utils/tdl_log.hpp"
TDLModelFactory::TDLModelFactory(const std::string model_dir)
    : model_dir_(model_dir + "/") {
  std::string str_ext = ".cvimodel";
#if defined(__BM168X__) || defined(__CV186X__)
  str_ext = ".bmodel";
#endif
  setModelPath(TDL_MODEL_TYPE_FACE_DETECTION_SCRFD,
               model_dir_ + "scrfd_500m_bnkps_432_768" + str_ext);
  setModelPath(TDL_MODEL_TYPE_OBJECT_DETECTION_YOLOV8_PERSON_VEHICLE,
               model_dir_ + "yolov8n_384_640_person_vehicle" + str_ext);
  setModelPath(TDL_MODEL_TYPE_FACE_LANDMARKER_LANDMARKERDETV2,
               model_dir_ + "pipnet_mbv1_at_50ep_v8" + str_ext);
  setModelPath(TDL_MODEL_TYPE_FACE_FEATURE_BMFACER34,
               model_dir_ + "bmface_r34" + str_ext);
  setModelPath(TDL_MODEL_TYPE_OBJECT_DETECTION_YOLOV8_HARDHAT,
               model_dir_ + "hardhat_detection" + str_ext);
  setModelPath(TDL_MODEL_TYPE_FACE_ATTRIBUTE_CLS,
               model_dir_ + "face_attribute_cls" + str_ext);
  setModelPath(TDL_MODEL_TYPE_FACE_ANTI_SPOOF_CLASSIFICATION,
               model_dir_ + "face_anti_spoof_classification" + str_ext);

  output_datas_type_str_[TDL_MODEL_TYPE_FACE_FEATURE_BMFACER34] = "feature";
  output_datas_type_str_[TDL_MODEL_TYPE_FACE_LANDMARKER_LANDMARKERDETV2] =
      "landmark";
  output_datas_type_str_
      [TDL_MODEL_TYPE_OBJECT_DETECTION_YOLOV8_PERSON_VEHICLE] = "objdet";
  output_datas_type_str_
      [TDL_MODEL_TYPE_OBJECT_DETECTION_MOBILEDETV2_PEDESTRIAN] = "objdet";
  output_datas_type_str_[TDL_MODEL_TYPE_OBJECT_DETECTION_YOLOV8_HARDHAT] =
      "objdet";
  output_datas_type_str_[TDL_MODEL_TYPE_OBJECT_DETECTION_YOLOV10] = "objdet";
  output_datas_type_str_[TDL_MODEL_TYPE_OBJECT_DETECTION_YOLOV6] = "objdet";
  output_datas_type_str_[TDL_MODEL_TYPE_FACE_DETECTION_SCRFD] = "face_det";
  output_datas_type_str_[TDL_MODEL_TYPE_FACE_ATTRIBUTE_CLS] = "face_det";
  output_datas_type_str_[TDL_MODEL_TYPE_FACE_ANTI_SPOOF_CLASSIFICATION] = "cls";
}

std::shared_ptr<BaseModel> TDLModelFactory::getModel(
    const TDL_MODEL_TYPE model_type, const int device_id) {
  if (model_path_map_.find(model_type) == model_path_map_.end()) {
    LOGE("model path not found for model type: %d", model_type);
    return nullptr;
  }
  std::string model_path = model_path_map_[model_type];

  return getModel(model_type, model_path, device_id);
}

std::shared_ptr<BaseModel> TDLModelFactory::getModel(
    const TDL_MODEL_TYPE model_type, const std::string &model_path,
    const int device_id) {
  std::shared_ptr<BaseModel> model = nullptr;
  (void)device_id;
  // 先创建模型实例
  std::map<int, TDLObjectType> model_type_mapping;
  if (model_type == TDL_MODEL_TYPE_FACE_DETECTION_SCRFD) {
    model = std::make_shared<SCRFD>();
  } else if (model_type ==
             TDL_MODEL_TYPE_OBJECT_DETECTION_YOLOV8_PERSON_VEHICLE) {
    model = std::make_shared<YoloV8Detection>(std::make_pair(64, 7));
  } else if (model_type == TDL_MODEL_TYPE_OBJECT_DETECTION_YOLOV8_HARDHAT) {
    model = std::make_shared<YoloV8Detection>(std::make_pair(64, 2));
  } else if (model_type == TDL_MODEL_TYPE_OBJECT_DETECTION_YOLOV10) {
    model = std::make_shared<YoloV10Detection>(std::make_pair(64, 80));
  } else if (model_type == TDL_MODEL_TYPE_OBJECT_DETECTION_YOLOV6) {
    model = std::make_shared<YoloV6Detection>(std::make_pair(4, 80));
  } else if (model_type ==
             TDL_MODEL_TYPE_OBJECT_DETECTION_MOBILEDETV2_PEDESTRIAN) {
    model = std::make_shared<MobileDetV2Detection>(
        MobileDetV2Detection::Category::pedestrian, 0.5);
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_PERSON;

  } else if (model_type == TDL_MODEL_TYPE_FACE_LANDMARKER_LANDMARKERDETV2) {
    model = std::make_shared<FaceLandmarkerDet2>();
  } else if (model_type == TDL_MODEL_TYPE_FACE_ATTRIBUTE_CLS) {
    model = std::make_shared<FaceAttribute_CLS>();
  } else if (model_type == TDL_MODEL_TYPE_FACE_FEATURE_BMFACER34) {
    model = std::make_shared<FeatureExtraction>();
  } else if (model_type == TDL_MODEL_TYPE_FACE_ANTI_SPOOF_CLASSIFICATION) {
    model = std::make_shared<RgbImageClassification>();
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

void TDLModelFactory::setModelPath(const TDL_MODEL_TYPE model_type,
                                   const std::string &model_path) {
  model_path_map_[model_type] = model_path;
}

void TDLModelFactory::setModelPathMap(
    const std::map<TDL_MODEL_TYPE, std::string> &model_path_map) {
  model_path_map_ = model_path_map;
}
