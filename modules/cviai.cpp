#include <unistd.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "cviai.h"

#include "face_attribute/face_attribute.hpp"
#include "retina_face/retina_face.hpp"
#include "yolov3/yolov3.hpp"

using namespace std;
using namespace cviai;

typedef struct {
  Core *instance = nullptr;
  std::string model_path = "";
} cviai_model_t;

typedef struct {
  std::unordered_map<CVI_AI_SUPPORTED_MODEL_E, cviai_model_t> model_cont;
} cviai_context_t;

int CVI_AI_CreateHandle(cviai_handle_t *handle) {
  cviai_context_t *ctx = new cviai_context_t;
  *handle = ctx;
  return CVI_RC_SUCCESS;
}

int CVI_AI_DestroyHandle(cviai_handle_t handle) {
  cviai_context_t *ctx = new cviai_context_t;
  CVI_AI_CloseAllModel(handle);
  delete ctx;
  return CVI_RC_SUCCESS;
}

int CVI_AI_SetModelPath(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                        const char *filepath) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  ctx->model_cont[config].model_path = filepath;
  return CVI_RC_SUCCESS;
}

int CVI_AI_CloseAllModel(cviai_handle_t handle) {
  cviai_context_t *ctx = new cviai_context_t;
  for (auto &m_inst : ctx->model_cont) {
    m_inst.second.instance->modelClose();
    delete m_inst.second.instance;
    m_inst.second.instance = nullptr;
  }
  ctx->model_cont.clear();
  return CVI_RC_SUCCESS;
}

int CVI_AI_CloseModel(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config) {
  cviai_context_t *ctx = new cviai_context_t;
  cviai_model_t &m_t = ctx->model_cont[config];
  if (m_t.instance == nullptr) {
    return CVI_RC_FAILURE;
  }
  m_t.instance->modelClose();
  delete m_t.instance;
  m_t.instance = nullptr;
  return CVI_RC_SUCCESS;
}

int CVI_AI_FaceAttribute(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvi_face_t *faces) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  cviai_model_t &m_t = ctx->model_cont[CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE];
  if (m_t.instance == nullptr) {
    if (m_t.model_path.empty()) {
      printf("Model path for FaceAttribute is empty.\n");
      return CVI_RC_FAILURE;
    }
    m_t.instance = new FaceAttribute();
    if (m_t.instance->modelOpen(m_t.model_path.c_str()) != CVI_RC_SUCCESS) {
      printf("Open model failed (%s).\n", m_t.model_path.c_str());
      return CVI_RC_FAILURE;
    }
  }

  FaceAttribute *face_attr = dynamic_cast<FaceAttribute *>(m_t.instance);
  if (face_attr == nullptr) {
    printf("No instance found for RetinaFace.\n");
    return CVI_RC_FAILURE;
  }

  return face_attr->inference(frame, faces);
}

int CVI_AI_Yolov3(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvi_object_t *obj,
                  cvi_obj_det_type_t det_type) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  cviai_model_t &m_t = ctx->model_cont[CVI_AI_SUPPORTED_MODEL_YOLOV3];
  if (m_t.instance == nullptr) {
    if (m_t.model_path.empty()) {
      printf("Model path for Yolov3 is empty.\n");
      return CVI_RC_FAILURE;
    }
    m_t.instance = new Yolov3();
    if (m_t.instance->modelOpen(m_t.model_path.c_str()) != CVI_RC_SUCCESS) {
      printf("Open model failed (%s).\n", m_t.model_path.c_str());
      return CVI_RC_FAILURE;
    }
  }

  Yolov3 *yolov3 = dynamic_cast<Yolov3 *>(m_t.instance);
  if (yolov3 == nullptr) {
    printf("No instance found for Yolov3.\n");
    return CVI_RC_FAILURE;
  }
  return yolov3->inference(frame, obj, det_type);
}

int CVI_AI_RetinaFace(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvi_face_t *faces,
                      int *face_count) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  cviai_model_t &m_t = ctx->model_cont[CVI_AI_SUPPORTED_MODEL_RETINAFACE];
  if (m_t.instance == nullptr) {
    if (m_t.model_path.empty()) {
      printf("Model path for RetinaFace is empty.\n");
      return CVI_RC_FAILURE;
    }
    m_t.instance = new RetinaFace();
    if (m_t.instance->modelOpen(m_t.model_path.c_str()) != CVI_RC_SUCCESS) {
      printf("Open model failed (%s).\n", m_t.model_path.c_str());
      return CVI_RC_FAILURE;
    }
  }

  RetinaFace *retina_face = dynamic_cast<RetinaFace *>(m_t.instance);
  if (retina_face == nullptr) {
    printf("No instance found for RetinaFace.\n");
    return CVI_RC_FAILURE;
  }

  return retina_face->inference(frame, faces, face_count);
}
