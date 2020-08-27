#include "evaluation/cviai_evaluation.h"

#include "coco/coco.hpp"
#include "lfw/lfw.hpp"
#include "wider_face/wider_face.hpp"

typedef struct {
  cviai::evaluation::CocoEval *coco_eval = nullptr;
  cviai::evaluation::lfwEval *lfw_eval = nullptr;
  cviai::evaluation::widerFaceEval *widerface_eval = nullptr;
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

CVI_S32 CVI_AI_Eval_CocoInit(cviai_eval_handle_t handle, const char *pathPrefix,
                             const char *jsonPath, uint32_t *imageNum) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->coco_eval == nullptr) {
    ctx->coco_eval = new cviai::evaluation::CocoEval(pathPrefix, jsonPath);
  } else {
    ctx->coco_eval->getEvalData(pathPrefix, jsonPath);
  }
  *imageNum = ctx->coco_eval->getTotalImage();
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

CVI_S32 CVI_AI_Eval_CocoSave2Json(cviai_eval_handle_t handle, const char *filepath) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->coco_eval == nullptr) {
    return CVI_FAILURE;
  }
  ctx->coco_eval->saveJsonObject2File(filepath);
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

CVI_S32 CVI_AI_Eval_LfwInit(cviai_eval_handle_t handle, const char *filepath, uint32_t *imageNum) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->lfw_eval == nullptr) {
    ctx->lfw_eval = new cviai::evaluation::lfwEval(filepath);
  } else {
    ctx->lfw_eval->getEvalData(filepath);
  }
  *imageNum = ctx->lfw_eval->getTotalImage();
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_LfwGetImageLabelPair(cviai_eval_handle_t handle, const uint32_t index,
                                         char **filepath, char **filepath2, int *label) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->lfw_eval == nullptr) {
    return CVI_FAILURE;
  }
  std::string filestr, filestr2;
  ctx->lfw_eval->getImageLabelPair(index, &filestr, &filestr2, label);
  auto stringlength = strlen(filestr.c_str()) + 1;
  *filepath = (char *)malloc(stringlength);
  strncpy(*filepath, filestr.c_str(), stringlength);
  stringlength = strlen(filestr2.c_str()) + 1;
  *filepath2 = (char *)malloc(stringlength);
  strncpy(*filepath2, filestr2.c_str(), stringlength);
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_LfwInsertFace(cviai_eval_handle_t handle, const int index, const int label,
                                  const cvai_face_t *face1, const cvai_face_t *face2) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->lfw_eval == nullptr) {
    return CVI_FAILURE;
  }
  ctx->lfw_eval->insertFaceData(index, label, face1, face2);
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_LfwSave2File(cviai_eval_handle_t handle, const char *filepath) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->lfw_eval == nullptr) {
    return CVI_FAILURE;
  }
  ctx->lfw_eval->saveEval2File(filepath);
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_LfwClearInput(cviai_eval_handle_t handle) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->lfw_eval == nullptr) {
    return CVI_FAILURE;
  }
  ctx->lfw_eval->resetData();
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_LfwClearEvalData(cviai_eval_handle_t handle) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->lfw_eval == nullptr) {
    return CVI_FAILURE;
  }
  ctx->lfw_eval->resetEvalData();
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_WiderFaceInit(cviai_eval_handle_t handle, const char *datasetDir,
                                  const char *resultDir, uint32_t *imageNum) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->widerface_eval == nullptr) {
    ctx->widerface_eval = new cviai::evaluation::widerFaceEval(datasetDir, resultDir);
  } else {
    ctx->widerface_eval->getEvalData(datasetDir, resultDir);
  }
  *imageNum = ctx->widerface_eval->getTotalImage();
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_WiderFaceGetImagePath(cviai_eval_handle_t handle, const uint32_t index,
                                          char **filepath) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->widerface_eval == nullptr) {
    return CVI_FAILURE;
  }
  std::string filestr;
  ctx->widerface_eval->getImageFilePath(index, &filestr);
  auto stringlength = strlen(filestr.c_str()) + 1;
  *filepath = (char *)malloc(stringlength);
  strncpy(*filepath, filestr.c_str(), stringlength);
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_WiderFaceResultSave2File(cviai_eval_handle_t handle, const int index,
                                             const VIDEO_FRAME_INFO_S *frame,
                                             const cvai_face_t *face) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->widerface_eval == nullptr) {
    return CVI_FAILURE;
  }
  ctx->widerface_eval->saveFaceData(index, frame, face);
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_WiderFaceClearInput(cviai_eval_handle_t handle) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->widerface_eval == nullptr) {
    return CVI_FAILURE;
  }
  ctx->widerface_eval->resetData();
  return CVI_SUCCESS;
}