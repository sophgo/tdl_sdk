#ifndef TDL_MODEL_FACTORY_H
#define TDL_MODEL_FACTORY_H

#include "model/base_model.hpp"
#include "tdl_model_defs.hpp"
class TDLModelFactory {
 public:
  TDLModelFactory(const std::string model_dir = "");
  std::shared_ptr<BaseModel> getModel(const ModelType model_type,
                                      const int device_id = 0);
  std::shared_ptr<BaseModel> getModel(const ModelType model_type,
                                      const std::string &model_path,
                                      const int device_id = 0);
  void setModelPath(const ModelType model_type, const std::string &model_path);
  void setModelPathMap(const std::map<ModelType, std::string> &model_path_map);

 private:
  std::string model_dir_;
  std::map<ModelType, std::string> model_path_map_;
};
#endif
