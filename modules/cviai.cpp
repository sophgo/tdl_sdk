#include <unistd.h>
#include <memory>
#include <string>
#include <vector>

#include "cviai.h"

#include "yolov3/yolov3.hpp"

using namespace std;
using namespace cviai;

typedef struct {
  Core *instance = nullptr;
  std::string model_path = "";
} cviai_model_t;

typedef struct {
  cviai_model_t yolov3;
} cviai_context_t;

int CVI_AI_InitHandle(cviai_handle_t *handle) {
  cviai_context_t *context = new cviai_context_t;
  *handle = context;
  return CVI_RC_SUCCESS;
}

int CVI_AI_SetModelPath(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                        const char *filepath) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  switch (config) {
    case CVI_AI_SUPPORTED_MODEL_YOLOV3:
      ctx->yolov3.model_path = filepath;
      break;
    default:
      printf("Unsupported model.\n");
      return CVI_RC_FAILURE;
      break;
  }
  return CVI_RC_SUCCESS;
}

int CVI_AI_ObjDetect(cviai_handle_t handle, VIDEO_FRAME_INFO_S *stObjDetFrame, cvi_object_t *obj,
                     cvi_obj_det_type_t det_type) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  if (ctx->yolov3.instance == nullptr) {
    if (ctx->yolov3.model_path.empty()) {
      printf("Model path for Yolov3 is empty.\n");
      return CVI_RC_FAILURE;
    }
    ctx->yolov3.instance = new Yolov3();
    if (ctx->yolov3.instance->modelOpen(ctx->yolov3.model_path.c_str()) != CVI_RC_SUCCESS) {
      printf("Open model failed (%s).\n", ctx->yolov3.model_path.c_str());
      return CVI_RC_FAILURE;
    }
  }

  Yolov3 *yolov3 = dynamic_cast<Yolov3*>(ctx->yolov3.instance);
  if (yolov3 == nullptr) {
    printf("No instance found for Yolov3.\n");
    return CVI_RC_FAILURE;
  }
  return yolov3->inference(stObjDetFrame, obj, det_type);
}
