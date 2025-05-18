#ifndef TDL_MODEL_DEF_H
#define TDL_MODEL_DEF_H
#include "tdl_model_list.h"
#ifdef __cplusplus
extern "C" {
#endif

// clang-format off
typedef enum {
  TDL_MODEL_INVALID = 0,

// generate from MODEL_TYPE_LIST by adding TDL_MODEL_ prefix
#define X(name, comment) TDL_MODEL_##name,
  MODEL_TYPE_LIST
#undef X

  TDL_MODEL_MAX
} TDLModel;
// clang-format on
#ifdef __cplusplus
}
#endif

#endif
