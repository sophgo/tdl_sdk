#include "models/tdl_model_factory.hpp"

#include "face_detection/scrfd.hpp"
#include "face_landmark/face_landmark_det2.hpp"
#include "feature_extract/feature_extraction.hpp"
#include "object_detection/mobiledet.hpp"
#include "object_detection/yolov8.hpp"

std::shared_ptr<BaseModel> TDLModelFactory::createModel(
    const TDL_MODEL_TYPE model_type, const std::string &model_path) {
  std::shared_ptr<BaseModel> model = nullptr;

  // 先创建模型实例
  if (model_type == TDL_MODEL_TYPE_FACE_DETECTION_SCRFD) {
    model = std::make_shared<SCRFD>();
  } else if (model_type ==
             TDL_MODEL_TYPE_OBJECT_DETECTION_YOLOV8_PERSON_VEHICLE) {
    model = std::make_shared<YoloV8Detection>(std::make_pair(64, 7));
  } else if (model_type ==
             TDL_MODEL_TYPE_OBJECT_DETECTION_MOBILEDETV2_PEDESTRIAN) {
    model = std::make_shared<MobileDetV2Detection>(
        MobileDetV2Detection::Category::pedestrian, 0.5);
  } else if (model_type == TDL_MODEL_TYPE_FACE_LANDMARKER_LANDMARKERDETV2) {
    model = std::make_shared<FaceLandmarkerDet2>();
  } else if (model_type == TDL_MODEL_TYPE_FACE_FEATURE_BMFACER34) {
    model = std::make_shared<FeatureExtraction>();
  } else {
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