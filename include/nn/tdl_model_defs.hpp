#ifndef TDL_MODEL_DEFS_H
#define TDL_MODEL_DEFS_H

#include "tdl_model_list.h"

enum class ModelType {
  INVALID = 0,
#define X(name, comment) name,
  MODEL_TYPE_LIST
#undef X
};

inline std::string modelTypeToString(ModelType c) {
  switch (c) {
#define X(name, comment) \
  case ModelType::name:  \
    return #name;
    MODEL_TYPE_LIST
#undef X
  }
  return "";
}

inline ModelType modelTypeFromString(const std::string& str) {
#define X(name, comment)    \
  if (str == #name) {       \
    return ModelType::name; \
  }
  MODEL_TYPE_LIST
#undef X
  return ModelType::INVALID;
}

static constexpr ModelType kAllModelTypes[] = {
#define X(name, comment) ModelType::name,
    MODEL_TYPE_LIST
#undef X
};

#endif
