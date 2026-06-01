#pragma once
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include "components/media_analysis/media_analysis_task.hpp"

class BaseImage;
class BaseModel;
class ModelOutputInfo;
class ModelFeatureInfo;

class FaceRegistrationTask : public MediaAnalysisTask {
 public:
  FaceRegistrationTask(const std::string& model_dir,
                       const std::string& data_path);
  virtual ~FaceRegistrationTask() = default;

  std::string get_event_type() const override { return "register_face"; }
  json handle_event(const json& request,
                    const std::string& description) override;

  json registerFace(const std::string& image_b64, const std::string& name,
                    bool force = false);

  void loadRegisteredNames();

 private:
  bool initModels();
  std::string getRegisteredNamesJson();

  struct RegistrationError {
    std::string error_code;
    std::string error_message;
  };

  RegistrationError validateFace(const std::shared_ptr<BaseImage>& image,
                                 std::shared_ptr<ModelOutputInfo>& out_fd);

  bool checkDuplicate(const std::shared_ptr<ModelFeatureInfo>& feature,
                      float threshold = 0.7f);

  int assignRegisteredId();
  bool saveFeature(const std::shared_ptr<ModelFeatureInfo>& feature,
                   int registered_id);
  bool appendRegisteredInfo(int registered_id, const std::string& name);

  std::string model_dir_;
  std::string data_path_;

  std::shared_ptr<BaseModel> model_fd_;  // face detection
  std::shared_ptr<BaseModel> model_fl_;  // face landmark
  std::shared_ptr<BaseModel> model_fe_;  // feature extraction

  std::map<std::string, int> name_to_id_map_;
  std::mutex mutex_;
  bool models_initialized_ = false;
};