#include "models/tdl_model_factory.hpp"

#include "core/cvi_tdl_types_mem.h"
#include "cvi_tdl_log.hpp"
#include "face_detection/scrfd.hpp"
#include "face_landmark/face_landmark_det2.hpp"
#include "face_attribute/face_attribute_cls.hpp"
#include "feature_extract/feature_extraction.hpp"
#include "object_detection/mobiledet.hpp"
#include "object_detection/yolov10.hpp"
#include "object_detection/yolov6.hpp"
#include "object_detection/yolov8.hpp"
#include "image_classification/rgb_image_classification.hpp"
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

  // 然后初始化模型
  if (model) {
    int ret = model->modelOpen(model_path);
    if (ret != 0) {
      return nullptr;
    }
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

int32_t TDLModelFactory::releaseOutput(const TDL_MODEL_TYPE model_type,
                                       std::vector<void *> &output_datas) {
  if (output_datas_type_str_.find(model_type) == output_datas_type_str_.end()) {
    LOGE("model path not found for model type: %d", model_type);
    assert(false);
    return -1;
  }
  std::string output_datas_type = output_datas_type_str_[model_type];
  if (output_datas_type == "feature") {
    for (size_t i = 0; i < output_datas.size(); i++) {
      cvtdl_feature_t *feature = (cvtdl_feature_t *)output_datas[i];
      CVI_TDL_FreeCpp(feature);
      free(feature);
    }
  } else if (output_datas_type == "face_det") {
    for (size_t i = 0; i < output_datas.size(); i++) {
      cvtdl_face_t *face_info = (cvtdl_face_t *)output_datas[i];
      CVI_TDL_FreeCpp(face_info);
      free(face_info);
    }
  } else if (output_datas_type == "objdet") {
    for (size_t i = 0; i < output_datas.size(); i++) {
      cvtdl_object_t *obj_info = (cvtdl_object_t *)output_datas[i];
      CVI_TDL_FreeCpp(obj_info);
      free(obj_info);
    }
  } else if (output_datas_type == "landmark") {
    for (size_t i = 0; i < output_datas.size(); i++) {
      cvtdl_face_info_t *face_info = (cvtdl_face_info_t *)output_datas[i];
      CVI_TDL_FreeCpp(face_info);
      free(face_info);
    }
  } else if (output_datas_type == "cls") {
    for (size_t i = 0; i < output_datas.size(); i++) {
      cvtdl_class_meta_t *cls_meta = (cvtdl_class_meta_t *)output_datas[i];
      CVI_TDL_FreeCpp(cls_meta);
      free(cls_meta);
    }
  } else {
    LOGE("output datas type not supported: %s", output_datas_type.c_str());
    assert(false);
    return -1;
  }
  return 0;
}