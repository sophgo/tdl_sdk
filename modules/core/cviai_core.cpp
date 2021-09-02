#include "version.hpp"

#include "core/cviai_core.h"
#include "cviai_core_internal.hpp"
#include "cviai_log.hpp"
#include "cviai_trace.hpp"

#include "alphapose/alphapose.hpp"
#include "cviai_experimental.h"
#include "cviai_perfetto.h"
#include "deepsort/cvi_deepsort.hpp"
#include "eye_classification/eye_classification.hpp"
#include "face_attribute/face_attribute.hpp"
#include "face_landmarker/face_landmarker.hpp"
#include "face_quality/face_quality.hpp"
#include "fall_detection/fall_detection.hpp"
#include "incar_object_detection/incar_object_detection.hpp"
#include "license_plate_detection/license_plate_detection.hpp"
#include "license_plate_recognition/license_plate_recognition.hpp"
#include "liveness/liveness.hpp"
#include "mask_classification/mask_classification.hpp"
#include "mask_face_recognition/mask_face_recognition.hpp"
#include "motion_detection/md.hpp"
#include "object_detection/mobiledetv2/mobiledetv2.hpp"
#include "object_detection/yolov3/yolov3.hpp"
#include "osnet/osnet.hpp"
#include "retina_face/retina_face.hpp"
#include "segmentation/deeplabv3.hpp"
#include "smoke_classification/smoke_classification.hpp"
#include "sound_classification/sound_classification.hpp"
#include "thermal_face_detection/thermal_face.hpp"
#include "yawn_classification/yawn_classification.hpp"

#include <stdarg.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;
using namespace cviai;

struct ModelParams {
  VpssEngine *vpss_engine;
  bool skip_vpss_preprocess;
  float model_threshold;
  uint32_t vpss_timeout_value;
  std::vector<uint32_t> *selected_classes;
};

using CreatorFunc = std::function<Core *(const ModelParams &)>;
using namespace std::placeholders;

template <typename C, typename... Args>
Core *create_model(const ModelParams &params, Args... arg) {
  C *instance = new C(arg...);

  instance->setVpssEngine(params.vpss_engine);
  instance->skipVpssPreprocess(params.skip_vpss_preprocess);
  if (params.model_threshold != -1) {
    instance->setModelThreshold(params.model_threshold);
  }
  instance->setVpssTimeout(params.vpss_timeout_value);
  return instance;
}

static void createIVEHandleIfNeeded(IVE_HANDLE *ive_handle) {
  if (*ive_handle == NULL) {
    *ive_handle = CVI_IVE_CreateHandle();
    if (*ive_handle == NULL) {
      LOGC("IVE handle init failed.\n");
    }
  }
}

// Convenience macros for creator
#define CREATOR(type) CreatorFunc(create_model<type>)

// Convenience macros for creator, P{NUM} stands for how many parameters for creator
#define CREATOR_P1(type, arg_type, arg1) \
  CreatorFunc(std::bind(create_model<type, arg_type>, _1, arg1))

/**
 * IMPORTANT!!
 * Creators for all DNN model. Please remember to register model creator here, or
 * AISDK cannot instantiate model correctly.
 */
unordered_map<int, CreatorFunc> MODEL_CREATORS = {
    {CVI_AI_SUPPORTED_MODEL_FACEQUALITY, CREATOR(FaceQuality)},
    {CVI_AI_SUPPORTED_MODEL_THERMALFACE, CREATOR(ThermalFace)},
    {CVI_AI_SUPPORTED_MODEL_LIVENESS, CREATOR(Liveness)},
    {CVI_AI_SUPPORTED_MODEL_MASKCLASSIFICATION, CREATOR(MaskClassification)},
    {CVI_AI_SUPPORTED_MODEL_YOLOV3, CREATOR(Yolov3)},
    {CVI_AI_SUPPORTED_MODEL_OSNET, CREATOR(OSNet)},
    {CVI_AI_SUPPORTED_MODEL_SOUNDCLASSIFICATION, CREATOR(SoundClassification)},
    {CVI_AI_SUPPORTED_MODEL_WPODNET, CREATOR(LicensePlateDetection)},
    {CVI_AI_SUPPORTED_MODEL_DEEPLABV3, CREATOR(Deeplabv3)},
    {CVI_AI_SUPPORTED_MODEL_ALPHAPOSE, CREATOR(AlphaPose)},
    {CVI_AI_SUPPORTED_MODEL_EYECLASSIFICATION, CREATOR(EyeClassification)},
    {CVI_AI_SUPPORTED_MODEL_YAWNCLASSIFICATION, CREATOR(YawnClassification)},
    {CVI_AI_SUPPORTED_MODEL_SMOKECLASSIFICATION, CREATOR(SmokeClassification)},
    {CVI_AI_SUPPORTED_MODEL_FACELANDMARKER, CREATOR(FaceLandmarker)},
    {CVI_AI_SUPPORTED_MODEL_INCAROBJECTDETECTION, CREATOR(IncarObjectDetection)},
    {CVI_AI_SUPPORTED_MODEL_MASKFACERECOGNITION, CREATOR(MaskFaceRecognition)},
    {CVI_AI_SUPPORTED_MODEL_RETINAFACE, CREATOR_P1(RetinaFace, PROCESS, CAFFE)},
    {CVI_AI_SUPPORTED_MODEL_RETINAFACE_IR, CREATOR_P1(RetinaFace, PROCESS, PYTORCH)},
    {CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE, CREATOR_P1(FaceAttribute, bool, true)},
    {CVI_AI_SUPPORTED_MODEL_FACERECOGNITION, CREATOR_P1(FaceAttribute, bool, false)},
    {CVI_AI_SUPPORTED_MODEL_LPRNET_TW, CREATOR_P1(LicensePlateRecognition, LP_FORMAT, TAIWAN)},
    {CVI_AI_SUPPORTED_MODEL_LPRNET_CN, CREATOR_P1(LicensePlateRecognition, LP_FORMAT, CHINA)},
    {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D0,
     CREATOR_P1(MobileDetV2, MobileDetV2::Model, MobileDetV2::Model::d0)},
    {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D1,
     CREATOR_P1(MobileDetV2, MobileDetV2::Model, MobileDetV2::Model::d1)},
    {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D2,
     CREATOR_P1(MobileDetV2, MobileDetV2::Model, MobileDetV2::Model::d2)},
    {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_LITE,
     CREATOR_P1(MobileDetV2, MobileDetV2::Model, MobileDetV2::Model::lite)},
    {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE_D0,
     CREATOR_P1(MobileDetV2, MobileDetV2::Model, MobileDetV2::Model::vehicle_d0)},
    {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN_D0,
     CREATOR_P1(MobileDetV2, MobileDetV2::Model, MobileDetV2::Model::pedestrian_d0)},
    {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_LITE_PERSON_PETS,
     CREATOR_P1(MobileDetV2, MobileDetV2::Model, MobileDetV2::Model::lite_person_pets)},
};

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
  delete ctx->md_model;
  delete ctx->ds_tracker;
  if (ctx->ive_handle) {
    CVI_IVE_DestroyHandle(ctx->ive_handle);
  }

  for (auto it : ctx->vec_vpss_engine) {
    delete it;
  }
  delete ctx;
}

inline Core *__attribute__((always_inline))
getInferenceInstance(const CVI_AI_SUPPORTED_MODEL_E index, cviai_context_t *ctx) {
  cviai_model_t &m_t = ctx->model_cont[index];
  if (m_t.instance == nullptr) {
    if (m_t.model_path.empty()) {
      LOGE("Model path for %s is empty.\n", CVI_AI_GetModelName(index));
      return nullptr;
    }

    if (MODEL_CREATORS.find(index) == MODEL_CREATORS.end()) {
      LOGE("Cannot find creator for %s, Please register a creator for this model!\n",
           CVI_AI_GetModelName(index));
      return nullptr;
    }

    auto creator = MODEL_CREATORS[index];
    ModelParams params = {
        .vpss_engine = ctx->vec_vpss_engine[m_t.vpss_thread],
        .skip_vpss_preprocess = m_t.skip_vpss_preprocess,
        .model_threshold = m_t.model_threshold,
        .vpss_timeout_value = ctx->vpss_timeout_value,
        .selected_classes = m_t.selected_classes,
    };

    m_t.instance = creator(params);
    if (m_t.instance->modelOpen(m_t.model_path.c_str()) != CVI_SUCCESS) {
      LOGE("Failed to open model %s (%s).\n", CVI_AI_GetModelName(index), m_t.model_path.c_str());
      return nullptr;
    }
    LOGI("model open is success: %s (%s)!\n", CVI_AI_GetModelName(index), m_t.model_path.c_str());

    // TODO: Don't setup for specific classes here, not all model should consider threshold....
    if (m_t.model_threshold == -1) {
      m_t.model_threshold = m_t.instance->getModelThreshold();
    }

    // TODO: Again, don't setup for specific classes here
    if (m_t.selected_classes) {
      // TODO: only support MobileDetV2 for now
      if (MobileDetV2 *mdetv2 = dynamic_cast<MobileDetV2 *>(m_t.instance)) {
        mdetv2->select_classes(*m_t.selected_classes);
      }
    }
  }

  return m_t.instance;
}

CVI_S32 CVI_AI_CreateHandle(cviai_handle_t *handle) { return CVI_AI_CreateHandle2(handle, -1, 0); }

CVI_S32 CVI_AI_CreateHandle2(cviai_handle_t *handle, const VPSS_GRP vpssGroupId,
                             const CVI_U8 vpssDev) {
  cviai_context_t *ctx = new cviai_context_t;
  ctx->ive_handle = NULL;

  ctx->vec_vpss_engine.push_back(new VpssEngine());
  if (ctx->vec_vpss_engine[0]->init(vpssGroupId, vpssDev) != CVI_SUCCESS) {
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
      LOGW("%s: Inference has already initialized. Please call CVI_AI_CloseModel to reset.\n",
           CVI_AI_GetModelName(config));
      return CVI_FAILURE;
    }
  }

  struct stat buffer;
  if (stat(filepath, &buffer) == 0) {
    if (S_ISREG(buffer.st_mode)) {
      m_t.model_path = filepath;
      return CVI_SUCCESS;
    } else {
      LOGE("Path of model file isn't a regular file: %s\n", filepath);
    }
  } else {
    LOGE("Model file doesn't exists: %s\n", filepath);
  }

  m_t.model_path.clear();
  return CVI_FAILURE;
}

CVI_S32 CVI_AI_OpenModel(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  cviai_model_t &m_t = ctx->model_cont[config];
  if (m_t.instance != nullptr) {
    if (!m_t.instance->isInitialized()) {
      if (m_t.model_path.empty()) {
        LOGE("Model path for %s is empty.\n", CVI_AI_GetModelName(config));
        return CVI_FAILURE;
      }

      if (m_t.instance->modelOpen(m_t.model_path.c_str()) == CVI_SUCCESS) {
        LOGI("Model opened: %s (%s)", CVI_AI_GetModelName(config), m_t.model_path.c_str());
      }
    } else {
      LOGW("%s: Inference has already initialized. Please call CVI_AI_CloseModel to reset.\n",
           CVI_AI_GetModelName(config));
      return CVI_SUCCESS;
    }
  } else {
    Core *instance = getInferenceInstance(config, ctx);
    if (!instance) {
      LOGE("Failed to open model %s\n", CVI_AI_GetModelName(config));
      return CVI_FAILURE;
    }
  }

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

  for (auto &m_inst : ctx->model_cont) {
    if (m_inst.second.instance != nullptr) {
      m_inst.second.instance->setVpssTimeout(timeout);
    }
  }
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_CloseAllModel(cviai_handle_t handle) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  for (auto &m_inst : ctx->model_cont) {
    if (m_inst.second.instance != nullptr) {
      m_inst.second.instance->modelClose();
      LOGI("Model closed: %s\n", CVI_AI_GetModelName(m_inst.first));
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
  LOGI("Model closed: %s\n", CVI_AI_GetModelName(config));
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
    // TODO: only supports MobileDetV2 for now
    if (MobileDetV2 *mdetv2 = dynamic_cast<MobileDetV2 *>(m_t.instance)) {
      mdetv2->select_classes(*m_t.selected_classes);
    } else {
      LOGW("CVI_AI_SelectDetectClass only supports MobileDetV2 family model for now.\n");
    }
  }
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_GetVpssChnConfig(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                const CVI_U32 frameWidth, const CVI_U32 frameHeight,
                                const CVI_U32 idx, cvai_vpssconfig_t *chnConfig) {
  CVI_S32 ret = CVI_SUCCESS;
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  cviai::Core *instance = getInferenceInstance(config, ctx);
  if (instance == nullptr) {
    LOGE("Instance is null.\n");
    return CVI_FAILURE;
  }

  if (instance->allowExportChannelAttribute()) {
    ret = instance->getChnConfig(frameWidth, frameHeight, idx, chnConfig);
  } else {
    LOGE("Currently model %u does not support exporting channel attribute.\n", config);
  }

  return ret;
}

CVI_S32 CVI_AI_EnalbeDumpInput(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                               const char *dump_path, bool enable) {
  CVI_S32 ret = CVI_SUCCESS;
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  cviai::Core *instance = getInferenceInstance(config, ctx);
  if (instance == nullptr) {
    LOGE("Instance is null.\n");
    return CVI_FAILURE;
  }

  instance->enableDebugger(enable);
  instance->setDebuggerOutputPath(dump_path);
  return ret;
}

/**
 *  Convenience macros for defining inference functions. F{NUM} stands for how many input frame
 *  variables, P{NUM} stands for how many input parameters in inference function. All inference
 *  function should follow same function signature, that is,
 *  CVI_S32 inference(Frame1, Frame2, ... FrameN, Param1, Param2, ..., ParamN)
 */
#define DEFINE_INF_FUNC_F1_P1(func_name, class_name, model_index, arg_type)                  \
  CVI_S32 func_name(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, arg_type arg1) { \
    TRACE_EVENT("cviai_core", #func_name);                                                   \
    cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);                           \
    class_name *obj = dynamic_cast<class_name *>(getInferenceInstance(model_index, ctx));    \
    if (obj == nullptr) {                                                                    \
      LOGE("No instance found for %s.\n", #class_name);                                      \
      return CVI_FAILURE;                                                                    \
    }                                                                                        \
    return obj->inference(frame, arg1);                                                      \
  }

#define DEFINE_INF_FUNC_F1_P2(func_name, class_name, model_index, arg1_type, arg2_type)     \
  CVI_S32 func_name(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, arg1_type arg1, \
                    arg2_type arg2) {                                                       \
    TRACE_EVENT("cviai_core", #func_name);                                                  \
    cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);                          \
    class_name *obj = dynamic_cast<class_name *>(getInferenceInstance(model_index, ctx));   \
    if (obj == nullptr) {                                                                   \
      LOGE("No instance found for %s.\n", #class_name);                                     \
      return CVI_FAILURE;                                                                   \
    }                                                                                       \
    return obj->inference(frame, arg1, arg2);                                               \
  }

#define DEFINE_INF_FUNC_F2_P1(func_name, class_name, model_index, arg_type)               \
  CVI_S32 func_name(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame1,              \
                    VIDEO_FRAME_INFO_S *frame2, arg_type arg1) {                          \
    TRACE_EVENT("cviai_core", #func_name);                                                \
    cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);                        \
    class_name *obj = dynamic_cast<class_name *>(getInferenceInstance(model_index, ctx)); \
    if (obj == nullptr) {                                                                 \
      LOGE("No instance found for %s.\n", #class_name);                                   \
      return CVI_FAILURE;                                                                 \
    }                                                                                     \
    return obj->inference(frame1, frame2, arg1);                                          \
  }

#define DEFINE_INF_FUNC_F2_P2(func_name, class_name, model_index, arg1_type, arg2_type)   \
  CVI_S32 func_name(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame1,              \
                    VIDEO_FRAME_INFO_S *frame2, arg1_type arg1, arg2_type arg2) {         \
    TRACE_EVENT("cviai_core", #func_name);                                                \
    cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);                        \
    class_name *obj = dynamic_cast<class_name *>(getInferenceInstance(model_index, ctx)); \
    if (obj == nullptr) {                                                                 \
      LOGE("No instance found for %s.\n", #class_name);                                   \
      return CVI_FAILURE;                                                                 \
    }                                                                                     \
    return obj->inference(frame1, frame2, arg1, arg2);                                    \
  }

/**
 *  Define model inference function here.
 *
 *  IMPORTANT!!
 *  Please remember to register creator function in MODEL_CREATORS first, or AISDK cannot
 *  find a correct way to create model object.
 *
 */
DEFINE_INF_FUNC_F1_P1(CVI_AI_RetinaFace, RetinaFace, CVI_AI_SUPPORTED_MODEL_RETINAFACE,
                      cvai_face_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_RetinaFace_IR, RetinaFace, CVI_AI_SUPPORTED_MODEL_RETINAFACE_IR,
                      cvai_face_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_ThermalFace, ThermalFace, CVI_AI_SUPPORTED_MODEL_THERMALFACE,
                      cvai_face_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_FaceAttribute, FaceAttribute, CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE,
                      cvai_face_t *)
DEFINE_INF_FUNC_F1_P2(CVI_AI_FaceAttributeOne, FaceAttribute, CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE,
                      cvai_face_t *, int)
DEFINE_INF_FUNC_F1_P1(CVI_AI_FaceRecognition, FaceAttribute, CVI_AI_SUPPORTED_MODEL_FACERECOGNITION,
                      cvai_face_t *)
DEFINE_INF_FUNC_F1_P2(CVI_AI_FaceRecognitionOne, FaceAttribute,
                      CVI_AI_SUPPORTED_MODEL_FACERECOGNITION, cvai_face_t *, int)
DEFINE_INF_FUNC_F1_P1(CVI_AI_MaskFaceRecognition, MaskFaceRecognition,
                      CVI_AI_SUPPORTED_MODEL_MASKFACERECOGNITION, cvai_face_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_FaceQuality, FaceQuality, CVI_AI_SUPPORTED_MODEL_FACEQUALITY,
                      cvai_face_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_MaskClassification, MaskClassification,
                      CVI_AI_SUPPORTED_MODEL_MASKCLASSIFICATION, cvai_face_t *)

DEFINE_INF_FUNC_F1_P1(CVI_AI_MobileDetV2_Vehicle_D0, MobileDetV2,
                      CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE_D0, cvai_object_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_MobileDetV2_Pedestrian_D0, MobileDetV2,
                      CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN_D0, cvai_object_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_MobileDetV2_Lite, MobileDetV2, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_LITE,
                      cvai_object_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_MobileDetV2_Lite_Person_Pets, MobileDetV2,
                      CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_LITE_PERSON_PETS, cvai_object_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_MobileDetV2_D0, MobileDetV2, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D0,
                      cvai_object_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_MobileDetV2_D1, MobileDetV2, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D1,
                      cvai_object_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_MobileDetV2_D2, MobileDetV2, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D2,
                      cvai_object_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_Yolov3, Yolov3, CVI_AI_SUPPORTED_MODEL_YOLOV3, cvai_object_t *)

DEFINE_INF_FUNC_F1_P1(CVI_AI_OSNet, OSNet, CVI_AI_SUPPORTED_MODEL_OSNET, cvai_object_t *)
DEFINE_INF_FUNC_F1_P2(CVI_AI_OSNetOne, OSNet, CVI_AI_SUPPORTED_MODEL_OSNET, cvai_object_t *, int)

DEFINE_INF_FUNC_F1_P1(CVI_AI_SoundClassification, SoundClassification,
                      CVI_AI_SUPPORTED_MODEL_SOUNDCLASSIFICATION, int *)
DEFINE_INF_FUNC_F2_P1(CVI_AI_DeeplabV3, Deeplabv3, CVI_AI_SUPPORTED_MODEL_DEEPLABV3,
                      cvai_class_filter_t *)

DEFINE_INF_FUNC_F1_P1(CVI_AI_LicensePlateRecognition_TW, LicensePlateRecognition,
                      CVI_AI_SUPPORTED_MODEL_LPRNET_TW, cvai_object_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_LicensePlateRecognition_CN, LicensePlateRecognition,
                      CVI_AI_SUPPORTED_MODEL_LPRNET_CN, cvai_object_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_LicensePlateDetection, LicensePlateDetection,
                      CVI_AI_SUPPORTED_MODEL_WPODNET, cvai_object_t *)

DEFINE_INF_FUNC_F1_P1(CVI_AI_AlphaPose, AlphaPose, CVI_AI_SUPPORTED_MODEL_ALPHAPOSE,
                      cvai_object_t *)

DEFINE_INF_FUNC_F1_P1(CVI_AI_EyeClassification, EyeClassification,
                      CVI_AI_SUPPORTED_MODEL_EYECLASSIFICATION, cvai_face_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_YawnClassification, YawnClassification,
                      CVI_AI_SUPPORTED_MODEL_YAWNCLASSIFICATION, cvai_face_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_SmokeClassification, SmokeClassification,
                      CVI_AI_SUPPORTED_MODEL_SMOKECLASSIFICATION, cvai_face_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_FaceLandmarker, FaceLandmarker, CVI_AI_SUPPORTED_MODEL_FACELANDMARKER,
                      cvai_face_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_IncarObjectDetection, IncarObjectDetection,
                      CVI_AI_SUPPORTED_MODEL_INCAROBJECTDETECTION, cvai_face_t *)

DEFINE_INF_FUNC_F2_P2(CVI_AI_Liveness, Liveness, CVI_AI_SUPPORTED_MODEL_LIVENESS, cvai_face_t *,
                      cvai_face_t *)

CVI_S32 CVI_AI_GetAlignedFace(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *srcFrame,
                              VIDEO_FRAME_INFO_S *dstFrame, cvai_face_info_t *face_info) {
  TRACE_EVENT("cviai_core", "CVI_AI_GetAlignedFace");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  FaceQuality *face_quality =
      dynamic_cast<FaceQuality *>(getInferenceInstance(CVI_AI_SUPPORTED_MODEL_FACEQUALITY, ctx));
  if (face_quality == nullptr) {
    LOGE("No instance found for FaceQuality.\n");
    return CVI_FAILURE;
  }
  return face_quality->getAlignedFace(srcFrame, dstFrame, face_info);
}

// Tracker

CVI_S32 CVI_AI_DeepSORT_Init(const cviai_handle_t handle, bool use_specific_counter) {
  TRACE_EVENT("cviai_core", "CVI_AI_DeepSORT_Init");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  DeepSORT *ds_tracker = ctx->ds_tracker;
  if (ds_tracker == nullptr) {
    LOGD("Init DeepSORT.\n");
    ctx->ds_tracker = new DeepSORT(use_specific_counter);
  } else {
    delete ds_tracker;
    LOGI("Re-init DeepSORT.\n");
    ctx->ds_tracker = new DeepSORT(use_specific_counter);
  }
  return 0;
}

CVI_S32 CVI_AI_DeepSORT_GetDefaultConfig(cvai_deepsort_config_t *ds_conf) {
  TRACE_EVENT("cviai_core", "CVI_AI_DeepSORT_GetDefaultConfig");
  cvai_deepsort_config_t default_conf = DeepSORT::get_DefaultConfig();
  memcpy(ds_conf, &default_conf, sizeof(cvai_deepsort_config_t));

  return 0;
}

CVI_S32 CVI_AI_DeepSORT_SetConfig(const cviai_handle_t handle, cvai_deepsort_config_t *ds_conf,
                                  int cviai_obj_type = -1, bool show_config = false) {
  TRACE_EVENT("cviai_core", "CVI_AI_DeepSORT_SetConf");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  DeepSORT *ds_tracker = ctx->ds_tracker;
  if (ds_tracker == nullptr) {
    LOGE("Please initialize DeepSORT first.\n");
    return CVI_FAILURE;
  }
  ds_tracker->setConfig(*ds_conf, cviai_obj_type, show_config);

  return 0;
}

CVI_S32 CVI_AI_DeepSORT_CleanCounter(const cviai_handle_t handle) {
  TRACE_EVENT("cviai_core", "CVI_AI_DeepSORT_CleanCounter");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  DeepSORT *ds_tracker = ctx->ds_tracker;
  if (ds_tracker == nullptr) {
    LOGE("Please initialize DeepSORT first.\n");
    return CVI_FAILURE;
  }
  ds_tracker->cleanCounter();

  return 0;
}

CVI_S32 CVI_AI_DeepSORT_Obj(const cviai_handle_t handle, cvai_object_t *obj,
                            cvai_tracker_t *tracker_t, bool use_reid) {
  TRACE_EVENT("cviai_core", "CVI_AI_DeepSORT_Obj");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  DeepSORT *ds_tracker = ctx->ds_tracker;
  if (ds_tracker == nullptr) {
    LOGE("Please initialize DeepSORT first.\n");
    return CVI_FAILURE;
  }
  ctx->ds_tracker->track(obj, tracker_t, use_reid);
  return 0;
}

CVI_S32 CVI_AI_DeepSORT_Face(const cviai_handle_t handle, cvai_face_t *face,
                             cvai_tracker_t *tracker_t, bool use_reid) {
  TRACE_EVENT("cviai_core", "CVI_AI_DeepSORT_Face");
  if (use_reid) {
    LOGE("Don't use DeepSORT for face tracking.\n");
    return CVI_FAILURE;
  }
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  DeepSORT *ds_tracker = ctx->ds_tracker;
  if (ds_tracker == nullptr) {
    LOGE("Please initialize DeepSORT first.\n");
    return CVI_FAILURE;
  }
  ctx->ds_tracker->track(face, tracker_t, false);
  return 0;
}

CVI_S32 CVI_AI_DeepSORT_DebugInfo_1(const cviai_handle_t handle, char *debug_info) {
  TRACE_EVENT("cviai_core", "CVI_AI_DeepSORT_DebugInfo_1");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  DeepSORT *ds_tracker = ctx->ds_tracker;
  if (ds_tracker == nullptr) {
    LOGE("Please initialize DeepSORT first.\n");
    return CVI_FAILURE;
  }
  std::string str_info;
  ctx->ds_tracker->get_TrackersInfo_UnmatchedLastTime(str_info);
  strncpy(debug_info, str_info.c_str(), 8192);

  return 0;
}

// Fall Detection

CVI_S32 CVI_AI_Fall(const cviai_handle_t handle, cvai_object_t *objects) {
  TRACE_EVENT("cviai_core", "CVI_AI_Fall");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  FallMD *fall_model = ctx->fall_model;
  if (fall_model == nullptr) {
    LOGD("Init Fall Detection Model.\n");
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
    LOGD("Init Tamper Detection Model.\n");
    createIVEHandleIfNeeded(&ctx->ive_handle);
    ctx->td_model = new TamperDetectorMD(ctx->ive_handle, frame, (float)0.05, (int)10);
    ctx->td_model->print_info();

    *moving_score = -1.0;
    return 0;
  }
  return ctx->td_model->detect(frame, moving_score);
}

CVI_S32 CVI_AI_Set_MotionDetection_Background(const cviai_handle_t handle,
                                              VIDEO_FRAME_INFO_S *frame, uint32_t threshold,
                                              double min_area) {
  TRACE_EVENT("cviai_core", "CVI_AI_Set_MotionDetection_Background");
  if (frame->stVFrame.enPixelFormat != PIXEL_FORMAT_YUV_400) {
    LOGE("Unsupported pixel format: %d. Motion Detection only support YUV400 format\n",
         frame->stVFrame.enPixelFormat);
    return CVI_FAILURE;
  }

  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  MotionDetection *md_model = ctx->md_model;
  if (md_model == nullptr) {
    LOGD("Init Motion Detection.\n");
    createIVEHandleIfNeeded(&ctx->ive_handle);
    ctx->md_model = new MotionDetection(ctx->ive_handle, frame, threshold, min_area);
    return CVI_SUCCESS;
  }
  return ctx->md_model->update_background(frame);
}

CVI_S32 CVI_AI_MotionDetection(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                               cvai_object_t *objects) {
  TRACE_EVENT("cviai_core", "CVI_AI_MotionDetection");

  if (frame->stVFrame.enPixelFormat != PIXEL_FORMAT_YUV_400) {
    LOGE("Unsupported pixel format: %d. Motion Detection only support YUV400 format\n",
         frame->stVFrame.enPixelFormat);
    return CVI_FAILURE;
  }
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  MotionDetection *md_model = ctx->md_model;
  if (md_model == nullptr) {
    LOGE("Failed to get motion detection instance\n");
    return CVI_FAILURE;
  }
  return ctx->md_model->detect(frame, objects);
}
