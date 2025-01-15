#ifndef TDL_MODEL_FACTORY_H
#define TDL_MODEL_FACTORY_H

#include "framework/model/base_model.hpp"
#include "models/tdl_model_defs.hpp"
class TDLModelFactory {
 public:
  static std::shared_ptr<BaseModel> createModel(const TDL_MODEL_TYPE model_type,
                                                const std::string &model_path);
};
#endif
