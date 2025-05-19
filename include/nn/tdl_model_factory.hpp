#ifndef TDL_MODEL_FACTORY_H
#define TDL_MODEL_FACTORY_H

#include <json.hpp>
#include "model/base_model.hpp"
#include "tdl_model_defs.hpp"
class TDLModelFactory {
 public:
  /*
   * get model instance,would use information loaded from model_config_file
   * @param model_type
   * @param device_id
   * @return model instance
   */
  std::shared_ptr<BaseModel> getModel(const ModelType model_type,
                                      const int device_id = 0);

  std::shared_ptr<BaseModel> getModel(const std::string &model_type,
                                      const int device_id = 0);
  /*
   * get model instance,use the default model config
   * @param model_type
   * @param model_path, absolute path of the model
   * @param device_id
   * @return model instance
   */
  std::shared_ptr<BaseModel> getModel(const ModelType model_type,
                                      const std::string &model_path,
                                      const int device_id = 0);
  std::shared_ptr<BaseModel> getModel(const std::string &model_type,
                                      const std::string &model_path,
                                      const int device_id = 0);
  /*
   * get model instance,use the information from the parameters
   * @param model_type
   * @param model_path, absolute path of the model
   * @param model_config, model config including mean, std, img_format, etc.
   * @param device_id
   * @return model instance
   */
  std::shared_ptr<BaseModel> getModel(const ModelType model_type,
                                      const std::string &model_path,
                                      const ModelConfig &model_config,
                                      const int device_id = 0);

  std::shared_ptr<BaseModel> getModel(const std::string &model_type,
                                      const std::string &model_path,
                                      const ModelConfig &model_config,
                                      const int device_id = 0);

  std::shared_ptr<BaseModel> getModel(const std::string &model_type,
                                      const std::string &model_path,
                                      const std::string &model_config_json,
                                      const int device_id = 0);
  ModelConfig getModelConfig(const ModelType model_type);
  /*
   * load model config from model_config_file
   * @param model_config_file, if empty, would load from
   * parent_dir_of(tdl.so)/configs/model/model_factory.json
   * @return 0 if success, -1 if failed
   */
  int32_t loadModelConfig(const std::string &model_config_file = "");
  void setModelDir(const std::string &model_dir);
  std::string getModelPath(const ModelType model_type);
  ModelConfig parseModelConfig(const nlohmann::json &json_config);
  std::vector<std::string> getModelList();
  static TDLModelFactory &getInstance();

 private:
  TDLModelFactory();
  ~TDLModelFactory();
  std::shared_ptr<BaseModel> getModelInstance(const ModelType model_type);

  void getPlatformAndModelExtension(std::string &platform,
                                    std::string &model_extension);
  // 辅助函数：判断模型类型
  bool isObjectDetectionModel(
      const ModelType model_type);  // YOLO和MobileNet系列
  bool isFaceDetectionModel(const ModelType model_type);
  bool isLaneDetectionModel(const ModelType model_type);
  bool isKeypointDetectionModel(const ModelType model_type);
  bool isClassificationModel(const ModelType model_type);
  bool isSegmentationModel(const ModelType model_type);
  bool isFeatureExtractionModel(const ModelType model_type);
  bool isOCRModel(const ModelType model_type);

  // 辅助函数：创建各类模型
  std::shared_ptr<BaseModel> createObjectDetectionModel(
      const ModelType model_type);
  std::shared_ptr<BaseModel> createFaceDetectionModel(
      const ModelType model_type);
  std::shared_ptr<BaseModel> createLaneDetectionModel(
      const ModelType model_type);
  std::shared_ptr<BaseModel> createKeypointDetectionModel(
      const ModelType model_type);
  std::shared_ptr<BaseModel> createClassificationModel(
      const ModelType model_type);
  std::shared_ptr<BaseModel> createSegmentationModel(
      const ModelType model_type);
  std::shared_ptr<BaseModel> createFeatureExtractionModel(
      const ModelType model_type);
  std::shared_ptr<BaseModel> createOCRModel(const ModelType model_type);

  std::string model_dir_;
  //   std::map<ModelType, std::string> model_path_map_;
  std::map<InferencePlatform, std::string> model_extension_map_;
  std::map<std::string, nlohmann::json> model_config_map_;
  InferencePlatform platform_;
  std::vector<std::string> coco_types_;
};
#endif
