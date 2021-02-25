#include "version.hpp"

#include "core/cviai_core.h"
#include "cviai_core_internal.hpp"
#include "cviai_log.hpp"
#include "cviai_trace.hpp"

#include "alphapose/alphapose.hpp"
#include "cviai_experimental.h"
#include "cviai_perfetto.h"
#include "deepsort/cvi_deepsort.hpp"
#include "es_classification/es_classification.hpp"
#include "eye_classification/eye_classification.hpp"
#include "face_attribute/face_attribute.hpp"
#include "face_landmarker/face_landmarker.hpp"
#include "face_quality/face_quality.hpp"
#include "fall_detection/fall_detection.hpp"
#include "license_plate_detection/license_plate_detection.hpp"
#include "license_plate_recognition/license_plate_recognition.hpp"
#include "liveness/liveness.hpp"
#include "mask_classification/mask_classification.hpp"
#include "mask_face_recognition/mask_face_recognition.hpp"
#include "object_detection/mobiledetv2/mobiledetv2.hpp"
#include "object_detection/yolov3/yolov3.hpp"
#include "osnet/osnet.hpp"
#include "retina_face/retina_face.hpp"
#include "segmentation/deeplabv3.hpp"
#include "thermal_face_detection/thermal_face.hpp"
#include "yawn_classification/yawn_classification.hpp"

#include <stdarg.h>
#include <stdio.h>
#include <unistd.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;
using namespace cviai;

void CVI_AI_PerfettoInit() { prefettoInit(); }

void CVI_AI_TraceBegin(const char *name) {
#ifdef SYSTRACE_FALLBACK
  TRACE_EVENT_BEGIN("cviai_api", name);
#else
  TRACE_EVENT_BEGIN("cviai_api", perfetto::StaticString{name});
#endif
}

void CVI_AI_TraceEnd() { TRACE_EVENT_END("cviai_api"); }

//*************************************************
// Experimental features
void CVI_AI_EnableGDC(cviai_handle_t handle, bool use_gdc) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  ctx->use_gdc_wrap = use_gdc;
  LOGI("Experimental feature GDC hardware %s.\n", use_gdc ? "enabled" : "disabled");
}
//*************************************************

inline void __attribute__((always_inline)) removeCtx(cviai_context_t *ctx) {
  delete ctx->td_model;
  delete ctx->ds_tracker;
  CVI_IVE_DestroyHandle(ctx->ive_handle);
  for (auto it : ctx->vec_vpss_engine) {
    delete it;
  }
  delete ctx;
}

CVI_S32 CVI_AI_CreateHandle(cviai_handle_t *handle) { return CVI_AI_CreateHandle2(handle, -1); }

CVI_S32 CVI_AI_CreateHandle2(cviai_handle_t *handle, const VPSS_GRP vpssGroupId) {
  cviai_context_t *ctx = new cviai_context_t;
  ctx->ive_handle = CVI_IVE_CreateHandle();
  if (ctx->ive_handle == NULL) {
    LOGC("IVE handle init failed.\n");
    return CVI_FAILURE;
  }
  ctx->vec_vpss_engine.push_back(new VpssEngine());
  if (ctx->vec_vpss_engine[0]->init(vpssGroupId) != CVI_SUCCESS) {
    LOGC("cviai_handle_t create failed.");
    removeCtx(ctx);
    return CVI_FAILURE;
  }
  const char timestamp[] = __DATE__ " " __TIME__;
  LOGI("cviai_handle_t created, version %s-%s", CVIAI_TAG, timestamp);
  *handle = ctx;
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_DestroyHandle(cviai_handle_t handle) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  CVI_AI_CloseAllModel(handle);
  removeCtx(ctx);
  LOGI("cviai_handle_t destroyed.");
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_SetModelPath(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                            const char *filepath) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  cviai_model_t &m_t = ctx->model_cont[config];
  if (m_t.instance != nullptr) {
    if (m_t.instance->isInitialized()) {
      LOGE("Inference already init. Please call CVI_AI_CloseModel to reset.\n");
      return CVI_FAILURE;
    }
  }
  m_t.model_path = filepath;
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_GetModelPath(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                            char **filepath) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  return GetModelName(ctx->model_cont[config], filepath);
}

CVI_S32 CVI_AI_SetSkipVpssPreprocess(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                     bool skip) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  auto &m_t = ctx->model_cont[config];
  m_t.skip_vpss_preprocess = skip;
  if (m_t.instance != nullptr) {
    m_t.instance->skipVpssPreprocess(m_t.skip_vpss_preprocess);
  }
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_GetSkipVpssPreprocess(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                     bool *skip) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  *skip = ctx->model_cont[config].skip_vpss_preprocess;
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_SetModelThreshold(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                 float threshold) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  auto &m_t = ctx->model_cont[config];
  m_t.model_threshold = threshold;
  if (m_t.instance != nullptr) {
    m_t.instance->setModelThreshold(m_t.model_threshold);
  }
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_GetModelThreshold(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                 float *threshold) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  *threshold = ctx->model_cont[config].model_threshold;
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_SetVpssThread(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                             const uint32_t thread) {
  return CVI_AI_SetVpssThread2(handle, config, thread, -1);
}

CVI_S32 CVI_AI_SetVpssThread2(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                              const uint32_t thread, const VPSS_GRP vpssGroupId) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  return setVPSSThread(ctx->model_cont[config], ctx->vec_vpss_engine, thread, vpssGroupId);
}

CVI_S32 CVI_AI_GetVpssThread(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                             uint32_t *thread) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  *thread = ctx->model_cont[config].vpss_thread;
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_GetVpssGrpIds(cviai_handle_t handle, VPSS_GRP **groups, uint32_t *num) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  VPSS_GRP *ids = (VPSS_GRP *)malloc(ctx->vec_vpss_engine.size() * sizeof(VPSS_GRP));
  for (size_t i = 0; i < ctx->vec_vpss_engine.size(); i++) {
    ids[i] = ctx->vec_vpss_engine[i]->getGrpId();
  }
  *groups = ids;
  *num = ctx->vec_vpss_engine.size();
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_SetVpssTimeout(cviai_handle_t handle, uint32_t timeout) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  ctx->vpss_timeout_value = timeout;
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_CloseAllModel(cviai_handle_t handle) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  for (auto &m_inst : ctx->model_cont) {
    if (m_inst.second.instance != nullptr) {
      m_inst.second.instance->modelClose();
      delete m_inst.second.instance;
      m_inst.second.instance = nullptr;
      if (m_inst.second.selected_classes) {
        delete m_inst.second.selected_classes;
        m_inst.second.selected_classes = nullptr;
      }
    }
  }
  for (auto &m_inst : ctx->custom_cont) {
    if (m_inst.instance != nullptr) {
      m_inst.instance->modelClose();
      delete m_inst.instance;
      m_inst.instance = nullptr;
    }
  }
  ctx->model_cont.clear();
  ctx->custom_cont.clear();
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_CloseModel(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  cviai_model_t &m_t = ctx->model_cont[config];
  if (m_t.instance == nullptr) {
    return CVI_FAILURE;
  }
  if (m_t.selected_classes) {
    delete m_t.selected_classes;
    m_t.selected_classes = nullptr;
  }

  m_t.instance->modelClose();
  delete m_t.instance;
  m_t.instance = nullptr;
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_SelectDetectClass(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                 uint32_t num_selection, ...) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  auto &m_t = ctx->model_cont[config];

  va_list args;
  va_start(args, num_selection);

  if (m_t.selected_classes) delete m_t.selected_classes;
  m_t.selected_classes = new std::vector<uint32_t>();
  for (uint32_t i = 0; i < num_selection; i++) {
    uint32_t selected_class = va_arg(args, uint32_t);

    if (selected_class & CVI_AI_DET_GROUP_MASK_HEAD) {
      uint32_t group_start = (selected_class & CVI_AI_DET_GROUP_MASK_START) >> 16;
      uint32_t group_end = (selected_class & CVI_AI_DET_GROUP_MASK_END);
      for (uint32_t i = group_start; i <= group_end; i++) {
        m_t.selected_classes->push_back(i);
      }
    } else {
      if (selected_class >= CVI_AI_DET_TYPE_END) {
        LOGE("Invalid class id: %d\n", selected_class);
        return CVI_FAILURE;
      }
      m_t.selected_classes->push_back(selected_class);
    }
  }

  if (m_t.instance != nullptr) {
    // TODO: only support MobileDetV2 for now
    if (MobileDetV2 *mdetv2 = dynamic_cast<MobileDetV2 *>(m_t.instance)) {
      mdetv2->select_classes(*m_t.selected_classes);
    }
  }
  return CVI_FAILURE;
}

template <class C, typename V, typename... Arguments>
inline C *__attribute__((always_inline))
getInferenceInstance(const V index, cviai_context_t *ctx, Arguments &&... arg) {
  cviai_model_t &m_t = ctx->model_cont[index];
  if (m_t.instance == nullptr) {
    if (m_t.model_path.empty()) {
      LOGE("Model path for %s is empty.\n", CVI_AI_GetModelName(index));
      return nullptr;
    }
    m_t.instance = new C(arg...);
    if (m_t.instance->modelOpen(m_t.model_path.c_str()) != CVI_SUCCESS) {
      LOGE("Open model failed (%s).\n", m_t.model_path.c_str());
      return nullptr;
    }
    m_t.instance->setVpssEngine(ctx->vec_vpss_engine[m_t.vpss_thread]);
    m_t.instance->skipVpssPreprocess(m_t.skip_vpss_preprocess);
    if (m_t.model_threshold == -1) {
      m_t.model_threshold = m_t.instance->getModelThreshold();
    } else {
      m_t.instance->setModelThreshold(m_t.model_threshold);
    }

    if (m_t.selected_classes) {
      // TODO: only support MobileDetV2 for now
      if (MobileDetV2 *mdetv2 = dynamic_cast<MobileDetV2 *>(m_t.instance)) {
        mdetv2->select_classes(*m_t.selected_classes);
      }
    }
  }
  m_t.instance->setVpssTimeout(ctx->vpss_timeout_value);
  C *class_inst = dynamic_cast<C *>(m_t.instance);
  return class_inst;
}

CVI_S32 CVI_AI_GetVpssChnConfig(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                const CVI_U32 frameWidth, const CVI_U32 frameHeight,
                                const CVI_U32 idx, cvai_vpssconfig_t *chnConfig) {
  CVI_S32 ret = CVI_SUCCESS;
  cviai::Core *instance = nullptr;
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  switch (config) {
    case CVI_AI_SUPPORTED_MODEL_RETINAFACE: {
      instance = getInferenceInstance<RetinaFace>(config, ctx);
    } break;
    case CVI_AI_SUPPORTED_MODEL_THERMALFACE: {
      instance = getInferenceInstance<ThermalFace>(config, ctx);
    } break;
    case CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_LITE: {
      instance = getInferenceInstance<MobileDetV2>(config, ctx, MobileDetV2::Model::lite);
    } break;
    case CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE_D0: {
      instance = getInferenceInstance<MobileDetV2>(config, ctx, MobileDetV2::Model::vehicle_d0);
    } break;
    case CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D0: {
      instance = getInferenceInstance<MobileDetV2>(config, ctx, MobileDetV2::Model::d0);
    } break;
    case CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D1: {
      instance = getInferenceInstance<MobileDetV2>(config, ctx, MobileDetV2::Model::d1);
    } break;
    case CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D2: {
      instance = getInferenceInstance<MobileDetV2>(config, ctx, MobileDetV2::Model::d2);
    } break;
    case CVI_AI_SUPPORTED_MODEL_YOLOV3: {
      instance = getInferenceInstance<Yolov3>(config, ctx);
    } break;
    default: {
      LOGE("Currently model %u does not support exporting channel attribute.\n", config);
    } break;
  }
  if (instance == nullptr) {
    LOGE("Instance is null.\n");
    return CVI_FAILURE;
  }
  ret = instance->getChnConfig(frameWidth, frameHeight, idx, chnConfig);
  return ret;
}

// Face detection

CVI_S32 CVI_AI_RetinaFace(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                          cvai_face_t *faces) {
  TRACE_EVENT("cviai_core", "CVI_AI_RetinaFace");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  RetinaFace *retina_face =
      getInferenceInstance<RetinaFace>(CVI_AI_SUPPORTED_MODEL_RETINAFACE, ctx);
  if (retina_face == nullptr) {
    LOGE("No instance found for RetinaFace.\n");
    return CVI_FAILURE;
  }
  return retina_face->inference(frame, faces);
}

CVI_S32 CVI_AI_ThermalFace(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                           cvai_face_t *faces) {
  TRACE_EVENT("cviai_core", "CVI_AI_ThermalFace");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  ThermalFace *thermal_face =
      getInferenceInstance<ThermalFace>(CVI_AI_SUPPORTED_MODEL_THERMALFACE, ctx);
  if (thermal_face == nullptr) {
    LOGE("No instance found for ThermalFace.\n");
    return CVI_FAILURE;
  }
  return thermal_face->inference(frame, faces);
}

// Face recognition

inline int __attribute__((always_inline))
CVI_AI_FaceAttributeBase(const CVI_AI_SUPPORTED_MODEL_E index, const cviai_handle_t handle,
                         VIDEO_FRAME_INFO_S *frame, cvai_face_t *faces, int face_idx,
                         bool set_attribute) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  FaceAttribute *face_attr = getInferenceInstance<FaceAttribute>(index, ctx, ctx->use_gdc_wrap);
  if (face_attr == nullptr) {
    LOGE("No instance found for FaceAttribute.\n");
    return CVI_FAILURE;
  }
  face_attr->setWithAttribute(set_attribute);
  return face_attr->inference(frame, faces, face_idx);
}

CVI_S32 CVI_AI_FaceAttribute(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                             cvai_face_t *faces) {
  TRACE_EVENT("cviai_core", "CVI_AI_FaceAttribute");
  return CVI_AI_FaceAttributeBase(CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE, handle, frame, faces, -1,
                                  true);
}

CVI_S32 CVI_AI_FaceAttributeOne(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                cvai_face_t *faces, int face_idx) {
  TRACE_EVENT("cviai_core", "CVI_AI_FaceAttributeOne");
  return CVI_AI_FaceAttributeBase(CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE, handle, frame, faces,
                                  face_idx, true);
}

CVI_S32 CVI_AI_FaceRecognition(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                               cvai_face_t *faces) {
  TRACE_EVENT("cviai_core", "CVI_AI_FaceRecognition");
  return CVI_AI_FaceAttributeBase(CVI_AI_SUPPORTED_MODEL_FACERECOGNITION, handle, frame, faces, -1,
                                  false);
}

CVI_S32 CVI_AI_FaceRecognitionOne(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                  cvai_face_t *faces, int face_idx) {
  TRACE_EVENT("cviai_core", "CVI_AI_FaceRecognitionOne");
  return CVI_AI_FaceAttributeBase(CVI_AI_SUPPORTED_MODEL_FACERECOGNITION, handle, frame, faces,
                                  face_idx, false);
}

CVI_S32 CVI_AI_MaskFaceRecognition(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                   cvai_face_t *faces) {
  TRACE_EVENT("cviai_core", "CVI_AI_MaskFaceRecognition");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  MaskFaceRecognition *mask_face_rec =
      getInferenceInstance<MaskFaceRecognition>(CVI_AI_SUPPORTED_MODEL_MASKFACERECOGNITION, ctx);
  if (mask_face_rec == nullptr) {
    LOGE("No instance found for MaskFaceRecognition.\n");
    return CVI_FAILURE;
  }

  return mask_face_rec->inference(frame, faces);
}

// Face classification

CVI_S32 CVI_AI_FaceQuality(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                           cvai_face_t *face) {
  TRACE_EVENT("cviai_core", "CVI_AI_FaceQuality");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  FaceQuality *face_quality =
      getInferenceInstance<FaceQuality>(CVI_AI_SUPPORTED_MODEL_FACEQUALITY, ctx);
  if (face_quality == nullptr) {
    LOGE("No instance found for FaceQuality.\n");
    return CVI_FAILURE;
  }
  return face_quality->inference(frame, face);
}

CVI_S32 CVI_AI_Liveness(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *rgbFrame,
                        VIDEO_FRAME_INFO_S *irFrame, cvai_face_t *rgb_face, cvai_face_t *ir_face) {
  TRACE_EVENT("cviai_core", "CVI_AI_Liveness");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  Liveness *liveness = getInferenceInstance<Liveness>(CVI_AI_SUPPORTED_MODEL_LIVENESS, ctx);
  if (liveness == nullptr) {
    LOGE("No instance found for Liveness.\n");
    return CVI_FAILURE;
  }
  return liveness->inference(rgbFrame, irFrame, rgb_face, ir_face);
}

CVI_S32 CVI_AI_MaskClassification(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                  cvai_face_t *face) {
  TRACE_EVENT("cviai_core", "CVI_AI_MaskClassification");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  MaskClassification *mask_classification =
      getInferenceInstance<MaskClassification>(CVI_AI_SUPPORTED_MODEL_MASKCLASSIFICATION, ctx);
  if (mask_classification == nullptr) {
    LOGE("No instance found for MaskClassification.\n");
    return CVI_FAILURE;
  }
  return mask_classification->inference(frame, face);
}

// Object detection

inline int __attribute__((always_inline))
MobileDetV2Base(const CVI_AI_SUPPORTED_MODEL_E index, const MobileDetV2::Model model_type,
                cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj,
                cvai_obj_det_type_e det_type) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  MobileDetV2 *detector = getInferenceInstance<MobileDetV2>(index, ctx, model_type);
  if (detector == nullptr) {
    LOGE("No instance found for detector.\n");
    return CVI_RC_FAILURE;
  }
  return detector->inference(frame, obj, det_type);
}

CVI_S32 CVI_AI_MobileDetV2_Vehicle_D0(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                      cvai_object_t *obj) {
  TRACE_EVENT("cviai_core", "CVI_AI_MobileDetV2_Vehicle_D0");
  return MobileDetV2Base(CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE_D0,
                         MobileDetV2::Model::vehicle_d0, handle, frame, obj, CVI_DET_TYPE_ALL);
}

CVI_S32 CVI_AI_MobileDetV2_Lite(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                cvai_object_t *obj, cvai_obj_det_type_e det_type) {
  TRACE_EVENT("cviai_core", "CVI_AI_MobileDetV2_Lite");
  return MobileDetV2Base(CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_LITE, MobileDetV2::Model::lite, handle,
                         frame, obj, det_type);
}

CVI_S32 CVI_AI_MobileDetV2_D0(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj,
                              cvai_obj_det_type_e det_type) {
  TRACE_EVENT("cviai_core", "CVI_AI_MobileDetV2_D0");
  return MobileDetV2Base(CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D0, MobileDetV2::Model::d0, handle,
                         frame, obj, det_type);
}

CVI_S32 CVI_AI_MobileDetV2_D1(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj,
                              cvai_obj_det_type_e det_type) {
  TRACE_EVENT("cviai_core", "CVI_AI_MobileDetV2_D1");
  return MobileDetV2Base(CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D1, MobileDetV2::Model::d1, handle,
                         frame, obj, det_type);
}

CVI_S32 CVI_AI_MobileDetV2_D2(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj,
                              cvai_obj_det_type_e det_type) {
  TRACE_EVENT("cviai_core", "CVI_AI_MobileDetV2_D2");
  return MobileDetV2Base(CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D2, MobileDetV2::Model::d2, handle,
                         frame, obj, det_type);
}

CVI_S32 CVI_AI_Yolov3(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj,
                      cvai_obj_det_type_e det_type) {
  TRACE_EVENT("cviai_core", "CVI_AI_Yolov3");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  Yolov3 *yolov3 = getInferenceInstance<Yolov3>(CVI_AI_SUPPORTED_MODEL_YOLOV3, ctx);
  if (yolov3 == nullptr) {
    LOGE("No instance found for Yolov3.\n");
    return CVI_FAILURE;
  }
  return yolov3->inference(frame, obj, det_type);
}

// Object recognition

inline int __attribute__((always_inline))
CVI_AI_OSNetBase(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj,
                 int obj_idx) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  OSNet *osnet = getInferenceInstance<OSNet>(CVI_AI_SUPPORTED_MODEL_OSNET, ctx);
  if (osnet == nullptr) {
    LOGE("No instance found for OSNet.\n");
    return CVI_FAILURE;
  }
  return osnet->inference(frame, obj, obj_idx);
}

CVI_S32 CVI_AI_OSNet(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj) {
  TRACE_EVENT("cviai_core", "CVI_AI_OSNet");
  return CVI_AI_OSNetBase(handle, frame, obj, -1);
}

CVI_S32 CVI_AI_OSNetOne(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj,
                        int obj_idx) {
  TRACE_EVENT("cviai_core", "CVI_AI_OSNetOne");
  return CVI_AI_OSNetBase(handle, frame, obj, obj_idx);
}

// Audio AI Inference

CVI_S32 CVI_AI_ESClassification(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                int *index) {
  TRACE_EVENT("cviai_core", "CVI_AI_ESClassification");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  ESClassification *es_classification =
      getInferenceInstance<ESClassification>(CVI_AI_SUPPORTED_MODEL_ESCLASSIFICATION, ctx);
  if (es_classification == nullptr) {
    LOGE("No instance found for ESClassification.\n");
    return CVI_FAILURE;
  }
  return es_classification->inference(frame, index);
}

// Segmentation

CVI_S32 CVI_AI_DeeplabV3(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                         VIDEO_FRAME_INFO_S *out_frame, cvai_class_filter_t *filter) {
  TRACE_EVENT("cviai_core", "CVI_AI_DeeplabV3");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  Deeplabv3 *deeplab = getInferenceInstance<Deeplabv3>(CVI_AI_SUPPORTED_MODEL_DEEPLABV3, ctx);
  if (deeplab == nullptr) {
    LOGE("No instance found for Deeplabv3.\n");
    return CVI_FAILURE;
  }
  return deeplab->inference(frame, out_frame, filter);
}

// Tracker

CVI_S32 CVI_AI_Deepsort_Init(const cviai_handle_t handle) {
  TRACE_EVENT("cviai_core", "CVI_AI_Deepsort_Init");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  Deepsort *ds_tracker = ctx->ds_tracker;
  if (ds_tracker == nullptr) {
    printf("Init Deepsort Tracker.\n");
    ctx->ds_tracker = new Deepsort();
  }
  return 0;
}

CVI_S32 CVI_AI_Deepsort_GetDefaultConfig(cvai_deepsort_config_t *ds_conf) {
  TRACE_EVENT("cviai_core", "CVI_AI_Deepsort_GetDefaultConfig");
  cvai_deepsort_config_t default_conf = Deepsort::get_DefaultConfig();
  memcpy(ds_conf, &default_conf, sizeof(cvai_deepsort_config_t));

  return 0;
}

CVI_S32 CVI_AI_Deepsort_SetConfig(const cviai_handle_t handle, cvai_deepsort_config_t *ds_conf) {
  TRACE_EVENT("cviai_core", "CVI_AI_Deepsort_SetConf");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  Deepsort *ds_tracker = ctx->ds_tracker;
  if (ds_tracker == nullptr) {
    LOGE("Please initialize deepsort first.\n");
    return CVI_FAILURE;
  }
  ds_tracker->setConfig(*ds_conf);

  return 0;
}

CVI_S32 CVI_AI_Deepsort(const cviai_handle_t handle, cvai_object_t *obj, cvai_tracker_t *tracker_t,
                        bool use_reid) {
  TRACE_EVENT("cviai_core", "CVI_AI_Deepsort_Track");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  Deepsort *ds_tracker = ctx->ds_tracker;
  if (ds_tracker == nullptr) {
    LOGE("Please initialize deepsort first.\n");
    return CVI_FAILURE;
  }
  ctx->ds_tracker->track(obj, tracker_t, use_reid);
  return 0;
}

CVI_S32 CVI_AI_Deepsort_DebugInfo_1(const cviai_handle_t handle, char *debug_info) {
  TRACE_EVENT("cviai_core", "CVI_AI_Deepsort_DebugInfo_1");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  Deepsort *ds_tracker = ctx->ds_tracker;
  if (ds_tracker == nullptr) {
    LOGE("Please initialize deepsort first.\n");
    return CVI_FAILURE;
  }
  std::string str_info;
  ctx->ds_tracker->get_TrackersInfo_UnmatchedLastTime(str_info);
  strncpy(debug_info, str_info.c_str(), 8192);

  return 0;
}

// License Plate Detection & Recognition
CVI_S32 CVI_AI_LicensePlateRecognition_TW(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                          cvai_object_t *license_plate_meta) {
  TRACE_EVENT("cviai_core", "CVI_AI_LicensePlateRecognition");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  LicensePlateRecognition *lpr_model =
      getInferenceInstance<LicensePlateRecognition>(CVI_AI_SUPPORTED_MODEL_LPRNET_TW, ctx, "tw");
  if (lpr_model == nullptr) {
    LOGE("No instance found for LicensePlateRecognition.\n");
    return CVI_FAILURE;
  }

  return lpr_model->inference(frame, license_plate_meta);
}

CVI_S32 CVI_AI_LicensePlateRecognition_CN(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                          cvai_object_t *license_plate_meta) {
  TRACE_EVENT("cviai_core", "CVI_AI_LicensePlateRecognition");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  LicensePlateRecognition *lpr_model =
      getInferenceInstance<LicensePlateRecognition>(CVI_AI_SUPPORTED_MODEL_LPRNET_CN, ctx, "cn");
  if (lpr_model == nullptr) {
    LOGE("No instance found for LicensePlateRecognition.\n");
    return CVI_FAILURE;
  }

  return lpr_model->inference(frame, license_plate_meta);
}

CVI_S32 CVI_AI_LicensePlateDetection(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                     cvai_object_t *vehicle_meta) {
  TRACE_EVENT("cviai_core", "CVI_AI_LicensePlateDetection");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  LicensePlateDetection *lpd_model =
      getInferenceInstance<LicensePlateDetection>(CVI_AI_SUPPORTED_MODEL_WPODNET, ctx);
  if (lpd_model == nullptr) {
    LOGE("No instance found for LicensePlateRecognition.\n");
    return CVI_FAILURE;
  }

  return lpd_model->inference(frame, vehicle_meta);
}

// Pose Detection

CVI_S32 CVI_AI_AlphaPose(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                         cvai_object_t *objects) {
  TRACE_EVENT("cviai_core", "CVI_AI_AlphaPose");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  AlphaPose *model = getInferenceInstance<AlphaPose>(CVI_AI_SUPPORTED_MODEL_ALPHAPOSE, ctx);
  if (model == nullptr) {
    LOGE("No instance found for AlphaPose.\n");
    return CVI_FAILURE;
  }

  return model->inference(frame, objects);
}

// Fall Detection

CVI_S32 CVI_AI_Fall(const cviai_handle_t handle, cvai_object_t *objects) {
  TRACE_EVENT("cviai_core", "CVI_AI_Fall");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  FallMD *fall_model = ctx->fall_model;
  if (fall_model == nullptr) {
    LOGI("Init Fall Detection Model.\n");
    ctx->fall_model = new FallMD();
    ctx->fall_model->detect(objects);
    return 0;
  }
  return ctx->fall_model->detect(objects);
}

// Others

CVI_S32 CVI_AI_TamperDetection(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                               float *moving_score) {
  TRACE_EVENT("cviai_core", "CVI_AI_TamperDetection");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  TamperDetectorMD *td_model = ctx->td_model;
  if (td_model == nullptr) {
    LOGI("Init Tamper Detection Model.\n");
    ctx->td_model = new TamperDetectorMD(ctx->ive_handle, frame, (float)0.05, (int)10);
    ctx->td_model->print_info();

    *moving_score = -1.0;
    return 0;
  }
  return ctx->td_model->detect(frame, moving_score);
}

// Eye Classification

CVI_S32 CVI_AI_EyeClassification(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                 cvai_face_t *face) {
  TRACE_EVENT("cviai_core", "CVI_AI_EyeClassifcation");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  EyeClassification *eye_classification =
      getInferenceInstance<EyeClassification>(CVI_AI_SUPPORTED_MODEL_EYECLASSIFICATION, ctx);
  if (eye_classification == nullptr) {
    LOGE("No instance found for EyeClassifcation.\n");
    return CVI_FAILURE;
  }
  return eye_classification->inference(frame, face);
}

// Yawn Classification

CVI_S32 CVI_AI_YawnClassification(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                  cvai_face_t *face) {
  TRACE_EVENT("cviai_core", "CVI_AI_YawnClassifcation");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  YawnClassification *yawn_classification =
      getInferenceInstance<YawnClassification>(CVI_AI_SUPPORTED_MODEL_YAWNCLASSIFICATION, ctx);
  if (yawn_classification == nullptr) {
    LOGE("No instance found for YawnClassifcation.\n");
    return CVI_FAILURE;
  }
  return yawn_classification->inference(frame, face);
}

// Face Landmarker

CVI_S32 CVI_AI_FaceLandmarker(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                              cvai_face_t *face) {
  TRACE_EVENT("cviai_core", "CVI_AI_FaceLandmarker");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  FaceLandmarker *face_landmarker =
      getInferenceInstance<FaceLandmarker>(CVI_AI_SUPPORTED_MODEL_FACELANDMARKER, ctx);
  if (face_landmarker == nullptr) {
    LOGE("No instance found for FaceLandmarker.\n");
    return CVI_FAILURE;
  }
  return face_landmarker->inference(frame, face);
}
