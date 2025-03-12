#ifndef _WRAPPER_TYPE_DEF_HPP_
#define _WRAPPER_TYPE_DEF_HPP_

#include <map>
#include "model/base_model.hpp"
#include "tdl_model_def.h"
#include "tdl_model_defs.hpp"
#include "tdl_model_factory.hpp"
#include "tdl_object_def.h"
#include "tdl_sdk.h"
#include "tdl_types.h"
typedef struct {
  std::unordered_map<cvtdl_model_e, std::shared_ptr<BaseModel>> models;
  std::shared_ptr<TDLModelFactory> model_factory;
} tdl_context_t;

typedef struct {
  std::shared_ptr<BaseImage> image;
} tdl_image_context_t;

cvtdl_object_type_e convertObjectType(TDLObjectType object_type) {
  switch (object_type) {
    case TDL_OBJECT_TYPE_PERSON:
      return TDL_OBJECT_TYPE_PERSON;
  }
}

TDL_MODEL_TYPE convertModelType(cvtdl_model_e model_type) {
  switch(model_type) {
    case TDL_MODEL_SCRFDFACE:
      return TDL_MODEL_TYPE_FACE_DETECTION_SCRFD;
  }
  return TDL_MODEL_TYPE_INVALID;
}
#endif
