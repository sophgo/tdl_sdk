#include "evaluation/cviai_evaluation.h"

#include "coco/coco.hpp"

typedef struct {
  cviai::evaluation::CocoEval *coco_eval = nullptr;
} cviai_eval_context_t;

CVI_S32 CVI_AI_Eval_CreateHandle(cviai_eval_handle_t *handle) {
  cviai_eval_context_t *ctx = new cviai_eval_context_t;
  *handle = ctx;
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_DestroyHandle(cviai_eval_handle_t handle) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  delete ctx->coco_eval;
  delete ctx;
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_CocoInit(cviai_eval_handle_t handle, const char *path_prefix,
                             const char *json_path, uint32_t *image_num) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->coco_eval == nullptr) {
    ctx->coco_eval = new cviai::evaluation::CocoEval(path_prefix, json_path);
  } else {
    ctx->coco_eval->getEvalData(path_prefix, json_path);
  }
  *image_num = ctx->coco_eval->getTotalImage();
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_CocoGetImageIdPair(cviai_eval_handle_t handle, const uint32_t index,
                                       char **filepath, int *id) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->coco_eval == nullptr) {
    return CVI_FAILURE;
  }
  std::string filestr;
  ctx->coco_eval->getImageIdPair(index, &filestr, id);
  auto stringlength = strlen(filestr.c_str()) + 1;
  *filepath = (char *)malloc(stringlength);
  strncpy(*filepath, filestr.c_str(), stringlength);
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_CocoInsertObject(cviai_eval_handle_t handle, const int id, cvai_object_t *obj) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->coco_eval == nullptr) {
    return CVI_FAILURE;
  }
  ctx->coco_eval->insertObjectData(id, obj);
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_CocoSave2Json(cviai_eval_handle_t handle) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->coco_eval == nullptr) {
    return CVI_FAILURE;
  }
  ctx->coco_eval->saveJsonObject2File();
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_CocoClearInput(cviai_eval_handle_t handle) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->coco_eval == nullptr) {
    return CVI_FAILURE;
  }
  ctx->coco_eval->resetReadJsonObject();
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_CocoClearObject(cviai_eval_handle_t handle) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->coco_eval == nullptr) {
    return CVI_FAILURE;
  }
  ctx->coco_eval->resetWriteJsonObject();
  return CVI_SUCCESS;
}