#include "evaluation/cviai_evaluation.h"

#include "cityscapes/cityscapes.hpp"
#include "coco/coco.hpp"
#include "coco_utils.hpp"
#include "core/core/cvai_errno.h"
#include "cvi_lpdr/cvi_lpdr.hpp"
#include "lfw/lfw.hpp"
#include "market1501/market1501.hpp"
#include "wflw/wflw.hpp"
#include "wider_face/wider_face.hpp"

typedef struct {
  cviai::evaluation::cityscapesEval *cityscapes_eval = nullptr;
  cviai::evaluation::CocoEval *coco_eval = nullptr;
  cviai::evaluation::market1501Eval *market1501_eval = nullptr;
  cviai::evaluation::lfwEval *lfw_eval = nullptr;
  cviai::evaluation::widerFaceEval *widerface_eval = nullptr;
  cviai::evaluation::wflwEval *wflw_eval = nullptr;
  cviai::evaluation::LPDREval *lpdr_eval = nullptr;
} cviai_eval_context_t;

CVI_S32 CVI_AI_Eval_CreateHandle(cviai_eval_handle_t *handle) {
  cviai_eval_context_t *ctx = new cviai_eval_context_t;
  *handle = ctx;
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_DestroyHandle(cviai_eval_handle_t handle) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  delete ctx->coco_eval;
  delete ctx->lpdr_eval;
  delete ctx;
  return CVIAI_SUCCESS;
}

/****************************************************************
 * Cityscapes evaluation functions
 **/
CVI_S32 CVI_AI_Eval_CityscapesInit(cviai_eval_handle_t handle, const char *image_dir,
                                   const char *output_dir) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->cityscapes_eval == nullptr) {
    ctx->cityscapes_eval = new cviai::evaluation::cityscapesEval(image_dir, output_dir);
  }

  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_CityscapesGetImage(cviai_eval_handle_t handle, const uint32_t index,
                                       char **fileName) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->cityscapes_eval == nullptr) {
    return CVIAI_FAILURE;
  }
  std::string filestr;
  ctx->cityscapes_eval->getImage(index, filestr);
  auto stringlength = strlen(filestr.c_str()) + 1;
  *fileName = (char *)malloc(stringlength);
  strncpy(*fileName, filestr.c_str(), stringlength);

  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_CityscapesGetImageNum(cviai_eval_handle_t handle, uint32_t *num) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->cityscapes_eval == nullptr) {
    return CVIAI_FAILURE;
  }
  ctx->cityscapes_eval->getImageNum(num);

  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_CityscapesWriteResult(cviai_eval_handle_t handle,
                                          VIDEO_FRAME_INFO_S *label_frame, const int index) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->cityscapes_eval == nullptr) {
    return CVIAI_FAILURE;
  }
  ctx->cityscapes_eval->writeResult(label_frame, index);
  return CVIAI_SUCCESS;
}

/****************************************************************
 * Coco evaluation functions
 **/
CVI_S32 CVI_AI_Eval_CocoInit(cviai_eval_handle_t handle, const char *pathPrefix,
                             const char *jsonPath, uint32_t *imageNum) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->coco_eval == nullptr) {
    ctx->coco_eval = new cviai::evaluation::CocoEval(pathPrefix, jsonPath);
  } else {
    ctx->coco_eval->getEvalData(pathPrefix, jsonPath);
  }
  *imageNum = ctx->coco_eval->getTotalImage();
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_CocoGetImageIdPair(cviai_eval_handle_t handle, const uint32_t index,
                                       char **filepath, int *id) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->coco_eval == nullptr) {
    return CVIAI_FAILURE;
  }
  std::string filestr;
  ctx->coco_eval->getImageIdPair(index, &filestr, id);
  auto stringlength = strlen(filestr.c_str()) + 1;
  *filepath = (char *)malloc(stringlength);
  strncpy(*filepath, filestr.c_str(), stringlength);
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_CocoInsertObject(cviai_eval_handle_t handle, const int id, cvai_object_t *obj) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->coco_eval == nullptr) {
    return CVIAI_FAILURE;
  }
  ctx->coco_eval->insertObjectData(id, obj);
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_CocoStartEval(cviai_eval_handle_t handle, const char *filepath) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->coco_eval == nullptr) {
    return CVIAI_FAILURE;
  }
  ctx->coco_eval->start_eval(filepath);
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_CocoEndEval(cviai_eval_handle_t handle) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->coco_eval == nullptr) {
    return CVIAI_FAILURE;
  }
  ctx->coco_eval->end_eval();
  return CVIAI_SUCCESS;
}

/****************************************************************
 * LFW evaluation functions
 **/
CVI_S32 CVI_AI_Eval_LfwInit(cviai_eval_handle_t handle, const char *filepath, bool label_pos_first,
                            uint32_t *imageNum) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->lfw_eval == nullptr) {
    ctx->lfw_eval = new cviai::evaluation::lfwEval();
  }

  if (ctx->lfw_eval->getEvalData(filepath, label_pos_first) != CVIAI_SUCCESS) {
    return CVIAI_FAILURE;
  }

  *imageNum = ctx->lfw_eval->getTotalImage();
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_LfwGetImageLabelPair(cviai_eval_handle_t handle, const uint32_t index,
                                         char **filepath, char **filepath2, int *label) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->lfw_eval == nullptr) {
    return CVIAI_FAILURE;
  }
  std::string filestr, filestr2;
  ctx->lfw_eval->getImageLabelPair(index, &filestr, &filestr2, label);
  auto stringlength = strlen(filestr.c_str()) + 1;
  *filepath = (char *)malloc(stringlength);
  strncpy(*filepath, filestr.c_str(), stringlength);
  stringlength = strlen(filestr2.c_str()) + 1;
  *filepath2 = (char *)malloc(stringlength);
  strncpy(*filepath2, filestr2.c_str(), stringlength);
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_LfwInsertFace(cviai_eval_handle_t handle, const int index, const int label,
                                  const cvai_face_t *face1, const cvai_face_t *face2) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->lfw_eval == nullptr) {
    return CVIAI_FAILURE;
  }
  ctx->lfw_eval->insertFaceData(index, label, face1, face2);
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_LfwInsertLabelScore(cviai_eval_handle_t handle, const int index,
                                        const int label, const float score) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->lfw_eval == nullptr) {
    return CVIAI_FAILURE;
  }
  ctx->lfw_eval->insertLabelScore(index, label, score);
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_LfwSave2File(cviai_eval_handle_t handle, const char *filepath) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->lfw_eval == nullptr) {
    return CVIAI_FAILURE;
  }
  ctx->lfw_eval->saveEval2File(filepath);
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_LfwClearInput(cviai_eval_handle_t handle) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->lfw_eval == nullptr) {
    return CVIAI_FAILURE;
  }
  ctx->lfw_eval->resetData();
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_LfwClearEvalData(cviai_eval_handle_t handle) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->lfw_eval == nullptr) {
    return CVIAI_FAILURE;
  }
  ctx->lfw_eval->resetEvalData();
  return CVIAI_SUCCESS;
}

/****************************************************************
 * Wider Face evaluation functions
 **/
CVI_S32 CVI_AI_Eval_WiderFaceInit(cviai_eval_handle_t handle, const char *datasetDir,
                                  const char *resultDir, uint32_t *imageNum) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->widerface_eval == nullptr) {
    ctx->widerface_eval = new cviai::evaluation::widerFaceEval(datasetDir, resultDir);
  } else {
    ctx->widerface_eval->getEvalData(datasetDir, resultDir);
  }
  *imageNum = ctx->widerface_eval->getTotalImage();
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_WiderFaceGetImagePath(cviai_eval_handle_t handle, const uint32_t index,
                                          char **filepath) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->widerface_eval == nullptr) {
    return CVIAI_FAILURE;
  }
  std::string filestr;
  ctx->widerface_eval->getImageFilePath(index, &filestr);
  auto stringlength = strlen(filestr.c_str()) + 1;
  *filepath = (char *)malloc(stringlength);
  strncpy(*filepath, filestr.c_str(), stringlength);
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_WiderFaceResultSave2File(cviai_eval_handle_t handle, const int index,
                                             const VIDEO_FRAME_INFO_S *frame,
                                             const cvai_face_t *face) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->widerface_eval == nullptr) {
    return CVIAI_FAILURE;
  }
  ctx->widerface_eval->saveFaceData(index, frame, face);
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_WiderFaceClearInput(cviai_eval_handle_t handle) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->widerface_eval == nullptr) {
    return CVIAI_FAILURE;
  }
  ctx->widerface_eval->resetData();
  return CVIAI_SUCCESS;
}

/****************************************************************
 * Market1501 evaluation functions
 **/
CVI_S32 CVI_AI_Eval_Market1501Init(cviai_eval_handle_t handle, const char *filepath) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->market1501_eval == nullptr) {
    ctx->market1501_eval = new cviai::evaluation::market1501Eval(filepath);
  } else {
    ctx->market1501_eval->getEvalData(filepath);
  }

  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_Market1501GetImageNum(cviai_eval_handle_t handle, bool is_query,
                                          uint32_t *num) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->market1501_eval == nullptr) {
    return CVIAI_FAILURE;
  }

  *num = ctx->market1501_eval->getImageNum(is_query);
  printf("query_dir: %d\n", *num);

  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_Market1501GetPathIdPair(cviai_eval_handle_t handle, const uint32_t index,
                                            bool is_query, char **filepath, int *cam_id, int *pid) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->market1501_eval == nullptr) {
    return CVIAI_FAILURE;
  }

  std::string filestr;
  ctx->market1501_eval->getPathIdPair(index, is_query, &filestr, cam_id, pid);
  auto stringlength = strlen(filestr.c_str()) + 1;
  *filepath = (char *)malloc(stringlength);
  strncpy(*filepath, filestr.c_str(), stringlength);

  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_Market1501InsertFeature(cviai_eval_handle_t handle, const int index,
                                            bool is_query, const cvai_feature_t *feature) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->market1501_eval == nullptr) {
    return CVIAI_FAILURE;
  }
  ctx->market1501_eval->insertFeature(index, is_query, feature);
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_Market1501EvalCMC(cviai_eval_handle_t handle) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->market1501_eval == nullptr) {
    return CVIAI_FAILURE;
  }
  ctx->market1501_eval->evalCMC();
  return CVIAI_SUCCESS;
}

/****************************************************************
 * WLFW evaluation functions
 **/
CVI_S32 CVI_AI_Eval_WflwInit(cviai_eval_handle_t handle, const char *filepath, uint32_t *imageNum) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->wflw_eval == nullptr) {
    ctx->wflw_eval = new cviai::evaluation::wflwEval(filepath);
  } else {
    ctx->wflw_eval->getEvalData(filepath);
  }
  *imageNum = ctx->wflw_eval->getTotalImage();
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_WflwGetImage(cviai_eval_handle_t handle, const uint32_t index,
                                 char **fileName) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->wflw_eval == nullptr) {
    return CVIAI_FAILURE;
  }
  std::string filestr;
  ctx->wflw_eval->getImage(index, &filestr);
  auto stringlength = strlen(filestr.c_str()) + 1;
  *fileName = (char *)malloc(stringlength);
  strncpy(*fileName, filestr.c_str(), stringlength);

  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_WflwInsertPoints(cviai_eval_handle_t handle, const int index,
                                     const cvai_pts_t points, const int width, const int height) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->wflw_eval == nullptr) {
    return CVIAI_FAILURE;
  }
  ctx->wflw_eval->insertPoints(index, points, width, height);
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_WflwDistance(cviai_eval_handle_t handle) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->wflw_eval == nullptr) {
    return CVIAI_FAILURE;
  }
  ctx->wflw_eval->distance();
  return CVIAI_SUCCESS;
}

/****************************************************************
 * LPDR evaluation functions
 **/
CVI_S32 CVI_AI_Eval_LPDRInit(cviai_eval_handle_t handle, const char *pathPrefix,
                             const char *jsonPath, uint32_t *imageNum) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->lpdr_eval == nullptr) {
    ctx->lpdr_eval = new cviai::evaluation::LPDREval(pathPrefix, jsonPath);
  } else {
    ctx->lpdr_eval->getEvalData(pathPrefix, jsonPath);
    return CVIAI_FAILURE;
  }
  *imageNum = ctx->lpdr_eval->getTotalImage();
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_Eval_LPDRGetImageIdPair(cviai_eval_handle_t handle, const uint32_t index,
                                       char **filepath, int *id) {
  cviai_eval_context_t *ctx = static_cast<cviai_eval_context_t *>(handle);
  if (ctx->lpdr_eval == nullptr) {
    return CVIAI_FAILURE;
  }
  std::string filestr;
  ctx->lpdr_eval->getImageIdPair(index, &filestr, id);
  auto stringlength = strlen(filestr.c_str()) + 1;
  *filepath = (char *)malloc(stringlength);
  strncpy(*filepath, filestr.c_str(), stringlength);
  return CVIAI_SUCCESS;
}