#ifndef TDL_MODEL_FACTORY_H
#define TDL_MODEL_FACTORY_H

#include "model/base_model.hpp"
#include "tdl_model_defs.hpp"
class TDLModelFactory {
 public:
  TDLModelFactory(const std::string model_dir = "");
  std::shared_ptr<BaseModel> getModel(
      const ModelType model_type,
      const std::map<std::string, std::string> &config = {},
      const int device_id = 0);
  std::shared_ptr<BaseModel> getModel(
      const ModelType model_type, const std::string &model_path,
      const std::map<std::string, std::string> &config = {},
      const int device_id = 0);
  void setModelPath(const ModelType model_type, const std::string &model_path);
  void setModelPathMap(const std::map<ModelType, std::string> &model_path_map);

 private:
  // 辅助函数：判断模型类型
  static bool isObjectDetectionModel(
      const ModelType model_type);  // YOLO和MobileNet系列
  static bool isFaceDetectionModel(const ModelType model_type);
  static bool isLaneDetectionModel(const ModelType model_type);
  static bool isKeypointDetectionModel(const ModelType model_type);
  static bool isClassificationModel(const ModelType model_type);
  static bool isSegmentationModel(const ModelType model_type);
  static bool isFeatureExtractionModel(const ModelType model_type);
  static bool isOCRModel(const ModelType model_type);

  // 辅助函数：创建各类模型
  static std::shared_ptr<BaseModel> createObjectDetectionModel(
      const ModelType model_type,
      const std::map<std::string, std::string> &config);
  static std::shared_ptr<BaseModel> createFaceDetectionModel(
      const ModelType model_type);
  static std::shared_ptr<BaseModel> createLaneDetectionModel(
      const ModelType model_type);
  static std::shared_ptr<BaseModel> createKeypointDetectionModel(
      const ModelType model_type);
  static std::shared_ptr<BaseModel> createClassificationModel(
      const ModelType model_type);
  static std::shared_ptr<BaseModel> createSegmentationModel(
      const ModelType model_type,
      const std::map<std::string, std::string> &config);
  static std::shared_ptr<BaseModel> createFeatureExtractionModel(
      const ModelType model_type,
      const std::map<std::string, std::string> &config);
  static std::shared_ptr<BaseModel> createOCRModel(const ModelType model_type);

  std::string model_dir_;
  std::map<ModelType, std::string> model_path_map_;
};
#endif
