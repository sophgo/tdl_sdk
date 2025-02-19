#ifndef TDL_MODEL_FACTORY_H
#define TDL_MODEL_FACTORY_H

#include "framework/model/base_model.hpp"
#include "models/tdl_model_defs.hpp"
class TDLModelFactory {
 public:
  TDLModelFactory(const std::string model_dir = "");
  std::shared_ptr<BaseModel> getModel(const TDL_MODEL_TYPE model_type,
                                      const int device_id = 0);
  std::shared_ptr<BaseModel> getModel(const TDL_MODEL_TYPE model_type,
                                      const std::string &model_path,
                                      const int device_id = 0);
  void setModelPath(const TDL_MODEL_TYPE model_type,
                    const std::string &model_path);
  void setModelPathMap(
      const std::map<TDL_MODEL_TYPE, std::string> &model_path_map);

  int32_t releaseOutput(const TDL_MODEL_TYPE model_type,
                        std::vector<void *> &output_datas);

 private:
  std::string model_dir_;
  std::map<TDL_MODEL_TYPE, std::string> model_path_map_;

  std::map<TDL_MODEL_TYPE, std::string> output_datas_type_str_;
};
#endif
