#include "version.hpp"

#include "core/cviai_core.h"
#include "cviai_core_internal.hpp"
#include "cviai_log.hpp"
#include "cviai_trace.hpp"

#include <stdarg.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "core/core/cvai_errno.h"
#include "core/cviai_types_mem_internal.h"
#include "cviai_experimental.h"
#include "motion_detection/md.hpp"
#include "object_detection/mobiledetv2/mobiledetv2.hpp"
#ifndef SIMPLY_MODEL
#include "cviai_perfetto.h"
#include "face_attribute/face_attribute.hpp"
#include "object_detection/yolov3/yolov3.hpp"
#include "object_detection/yolox/yolox.hpp"
#include "retina_face/retina_face.hpp"
#include "retina_face/scrfd_face.hpp"
#include "sound_classification/sound_classification.hpp"
#include "sound_classification/sound_classification_v2.hpp"
#endif
#include "utils/core_utils.hpp"
using namespace std;
using namespace cviai;
#define LOGE printf
struct ModelParams {
  VpssEngine *vpss_engine;
  uint32_t vpss_timeout_value;
};

using CreatorFunc = std::function<Core *(const ModelParams &)>;
using namespace std::placeholders;

template <typename C, typename... Args>
Core *create_model(const ModelParams &params, Args... arg) {
  C *instance = new C(arg...);

  instance->setVpssEngine(params.vpss_engine);
  instance->setVpssTimeout(params.vpss_timeout_value);
  return instance;
}

static void createIVEHandleIfNeeded(cviai_context_t *ctx) {
  if (ctx->ive_handle == nullptr) {
    ctx->ive_handle = new ive::IVE;
    if (ctx->ive_handle->init() != CVI_SUCCESS) {
      LOGC("IVE handle init failed.\n");
    }
  }
}

static CVI_S32 initVPSSIfNeeded(cviai_context_t *ctx, CVI_AI_SUPPORTED_MODEL_E model_id) {
  bool skipped;
  CVI_S32 ret = CVI_AI_GetSkipVpssPreprocess(ctx, model_id, &skipped);
  if (ret != CVIAI_SUCCESS) {
    return ret;
  }

  // Don't create vpss if preprocessing is skipped.
  if (skipped) {
    return CVIAI_SUCCESS;
  }

  uint32_t thread;
  ret = CVI_AI_GetVpssThread(ctx, model_id, &thread);
  if (ret == CVIAI_SUCCESS) {
    if (!ctx->vec_vpss_engine[thread]->isInitialized()) {
      ret = ctx->vec_vpss_engine[thread]->init();
    }
  }
  return ret;
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
#ifndef SIMPLY_MODEL
    {CVI_AI_SUPPORTED_MODEL_YOLOV3, CREATOR(Yolov3)},
    {CVI_AI_SUPPORTED_MODEL_YOLOX, CREATOR(YoloX)},
    {CVI_AI_SUPPORTED_MODEL_SOUNDCLASSIFICATION, CREATOR(SoundClassification)},
    {CVI_AI_SUPPORTED_MODEL_SOUNDCLASSIFICATION_V2, CREATOR(SoundClassificationV2)},
    {CVI_AI_SUPPORTED_MODEL_SCRFDFACE, CREATOR(ScrFDFace)},
    {CVI_AI_SUPPORTED_MODEL_RETINAFACE, CREATOR_P1(RetinaFace, PROCESS, CAFFE)},
    {CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE, CREATOR_P1(FaceAttribute, bool, true)},
    {CVI_AI_SUPPORTED_MODEL_FACERECOGNITION, CREATOR_P1(FaceAttribute, bool, false)},
#endif
    {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_COCO80,
     CREATOR_P1(MobileDetV2, MobileDetV2::Category, MobileDetV2::Category::coco80)},
    {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE,
     CREATOR_P1(MobileDetV2, MobileDetV2::Category, MobileDetV2::Category::person_vehicle)},
    {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE,
     CREATOR_P1(MobileDetV2, MobileDetV2::Category, MobileDetV2::Category::vehicle)},
    {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN,
     CREATOR_P1(MobileDetV2, MobileDetV2::Category, MobileDetV2::Category::pedestrian)},
    {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PERSON_PETS,
     CREATOR_P1(MobileDetV2, MobileDetV2::Category, MobileDetV2::Category::person_pets)},
};

#ifndef SIMPLY_MODEL
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
#endif
//*************************************************

inline void __attribute__((always_inline)) removeCtx(cviai_context_t *ctx) {
#ifndef SIMPLY_MODEL
  if (ctx->ds_tracker) {
    delete ctx->ds_tracker;
    ctx->ds_tracker = nullptr;
  }
#endif

  // delete ctx->td_model;
  if (ctx->md_model) {
    delete ctx->md_model;
    ctx->md_model = nullptr;
  }

  if (ctx->ive_handle) {
    ctx->ive_handle->destroy();
    ctx->ive_handle = nullptr;
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
    if (MODEL_CREATORS.find(index) == MODEL_CREATORS.end()) {
      LOGE("Cannot find creator for %s, Please register a creator for this model!\n",
           CVI_AI_GetModelName(index));
      return nullptr;
    }

    auto creator = MODEL_CREATORS[index];
    ModelParams params = {.vpss_engine = ctx->vec_vpss_engine[m_t.vpss_thread],
                          .vpss_timeout_value = ctx->vpss_timeout_value};

    m_t.instance = creator(params);
  }

  return m_t.instance;
}

CVI_S32 CVI_AI_CreateHandle(cviai_handle_t *handle) { return CVI_AI_CreateHandle2(handle, -1, 0); }

CVI_S32 CVI_AI_CreateHandle2(cviai_handle_t *handle, const VPSS_GRP vpssGroupId,
                             const CVI_U8 vpssDev) {
  if (vpssGroupId < -1 || vpssGroupId >= VPSS_MAX_GRP_NUM) {
    LOGE("Invalid Vpss Grp: %d.\n", vpssGroupId);
    return CVIAI_ERR_INIT_VPSS;
  }

  cviai_context_t *ctx = new cviai_context_t;
  ctx->ive_handle = NULL;
  ctx->vec_vpss_engine.push_back(new VpssEngine(vpssGroupId, vpssDev));
  const char timestamp[] = __DATE__ " " __TIME__;
  LOGI("cviai_handle_t is created, version %s-%s", CVIAI_TAG, timestamp);
  *handle = ctx;
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_DestroyHandle(cviai_handle_t handle) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  CVI_AI_CloseAllModel(handle);
  removeCtx(ctx);
  LOGI("cviai_handle_t is destroyed.");
  return CVIAI_SUCCESS;
}

static bool checkModelFile(const char *filepath) {
  struct stat buffer;
  bool ret = false;
  if (stat(filepath, &buffer) == 0) {
    if (S_ISREG(buffer.st_mode)) {
      ret = true;
    } else {
      LOGE("Path of model file isn't a regular file: %s\n", filepath);
    }
  } else {
    LOGE("Model file doesn't exists: %s\n", filepath);
  }
  return ret;
}

CVI_S32 CVI_AI_OpenModel(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                         const char *filepath) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  cviai_model_t &m_t = ctx->model_cont[config];
  Core *instance = getInferenceInstance(config, ctx);

  if (instance != nullptr) {
    if (instance->isInitialized()) {
      LOGW("%s: Inference has already initialized. Please call CVI_AI_CloseModel to reset.\n",
           CVI_AI_GetModelName(config));
      return CVIAI_ERR_MODEL_INITIALIZED;
    }
  } else {
    LOGE("Cannot create model: %s\n", CVI_AI_GetModelName(config));
    return CVIAI_ERR_OPEN_MODEL;
  }

  if (!checkModelFile(filepath)) {
    return CVIAI_ERR_INVALID_MODEL_PATH;
  }

  m_t.model_path = filepath;
  CVI_S32 ret = m_t.instance->modelOpen(m_t.model_path.c_str());
  if (ret != CVIAI_SUCCESS) {
    LOGE("Failed to open model: %s (%s)", CVI_AI_GetModelName(config), m_t.model_path.c_str());
    return ret;
  }
  LOGI("Model is opened successfully: %s \n", CVI_AI_GetModelName(config));
  return CVIAI_SUCCESS;
}

const char *CVI_AI_GetModelPath(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  return GetModelName(ctx->model_cont[config]);
}

CVI_S32 CVI_AI_SetSkipVpssPreprocess(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                     bool skip) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  Core *instance = getInferenceInstance(config, ctx);
  if (instance != nullptr) {
    instance->skipVpssPreprocess(skip);
  } else {
    LOGE("Cannot create model: %s\n", CVI_AI_GetModelName(config));
    return CVIAI_ERR_OPEN_MODEL;
  }
  return CVIAI_SUCCESS;
}
CVI_S32 CVI_AI_SetPerfEvalInterval(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                   int interval) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  Core *instance = getInferenceInstance(config, ctx);
  if (instance != nullptr) {
    instance->set_perf_eval_interval(interval);
  } else {
    LOGE("Cannot create model: %s\n", CVI_AI_GetModelName(config));
    return CVIAI_ERR_OPEN_MODEL;
  }
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_GetSkipVpssPreprocess(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                     bool *skip) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  Core *instance = getInferenceInstance(config, ctx);
  if (instance != nullptr) {
    *skip = instance->hasSkippedVpssPreprocess();
  } else {
    LOGE("Cannot create model: %s\n", CVI_AI_GetModelName(config));
    return CVIAI_ERR_OPEN_MODEL;
  }
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_SetModelThreshold(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                 float threshold) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  Core *instance = getInferenceInstance(config, ctx);
  if (instance != nullptr) {
    instance->setModelThreshold(threshold);
  } else {
    LOGE("Cannot create model: %s\n", CVI_AI_GetModelName(config));
    return CVIAI_ERR_OPEN_MODEL;
  }
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_GetModelThreshold(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                 float *threshold) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  Core *instance = getInferenceInstance(config, ctx);
  if (instance != nullptr) {
    *threshold = instance->getModelThreshold();
  } else {
    LOGE("Cannot create model: %s\n", CVI_AI_GetModelName(config));
    return CVIAI_ERR_OPEN_MODEL;
  }
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_SetVpssThread(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                             const uint32_t thread) {
  return CVI_AI_SetVpssThread2(handle, config, thread, -1, 0);
}

CVI_S32 CVI_AI_SetVpssThread2(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                              const uint32_t thread, const VPSS_GRP vpssGroupId, const CVI_U8 dev) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  Core *instance = getInferenceInstance(config, ctx);
  if (instance != nullptr) {
    return setVPSSThread(ctx->model_cont[config], ctx->vec_vpss_engine, thread, vpssGroupId, dev);
  } else {
    LOGE("Cannot create model: %s\n", CVI_AI_GetModelName(config));
    return CVIAI_ERR_OPEN_MODEL;
  }
}

CVI_S32 CVI_AI_SetVBPool(cviai_handle_t handle, uint32_t thread, VB_POOL pool_id) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  if (thread >= ctx->vec_vpss_engine.size()) {
    LOGE("Invalid vpss thread: %d, should be 0 to %d\n", thread,
         static_cast<uint32_t>(ctx->vec_vpss_engine.size() - 1));
    return CVIAI_FAILURE;
  }
  ctx->vec_vpss_engine[thread]->attachVBPool(pool_id);
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_GetVBPool(cviai_handle_t handle, uint32_t thread, VB_POOL *pool_id) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  if (thread >= ctx->vec_vpss_engine.size()) {
    LOGE("Invalid vpss thread: %d, should be 0 to %d\n", thread,
         static_cast<uint32_t>(ctx->vec_vpss_engine.size() - 1));
    return CVIAI_FAILURE;
  }
  *pool_id = ctx->vec_vpss_engine[thread]->getVBPool();
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_GetVpssThread(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                             uint32_t *thread) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  *thread = ctx->model_cont[config].vpss_thread;
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_SetVpssDepth(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model,
                            uint32_t input_id, uint32_t depth) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  Core *instance = getInferenceInstance(model, ctx);
  if (instance != nullptr) {
    return instance->setVpssDepth(input_id, depth);
  } else {
    LOGE("Cannot create model: %s\n", CVI_AI_GetModelName(model));
    return CVIAI_ERR_OPEN_MODEL;
  }
}

CVI_S32 CVI_AI_GetVpssDepth(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model,
                            uint32_t input_id, uint32_t *depth) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  Core *instance = getInferenceInstance(model, ctx);
  if (instance != nullptr) {
    return instance->getVpssDepth(input_id, depth);
  } else {
    LOGE("Cannot create model: %s\n", CVI_AI_GetModelName(model));
    return CVIAI_ERR_OPEN_MODEL;
  }
}

CVI_S32 CVI_AI_GetVpssGrpIds(cviai_handle_t handle, VPSS_GRP **groups, uint32_t *num) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  VPSS_GRP *ids = (VPSS_GRP *)malloc(ctx->vec_vpss_engine.size() * sizeof(VPSS_GRP));
  for (size_t i = 0; i < ctx->vec_vpss_engine.size(); i++) {
    ids[i] = ctx->vec_vpss_engine[i]->getGrpId();
  }
  *groups = ids;
  *num = ctx->vec_vpss_engine.size();
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_SetVpssTimeout(cviai_handle_t handle, uint32_t timeout) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  ctx->vpss_timeout_value = timeout;

  for (auto &m_inst : ctx->model_cont) {
    if (m_inst.second.instance != nullptr) {
      m_inst.second.instance->setVpssTimeout(timeout);
    }
  }
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_CloseAllModel(cviai_handle_t handle) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  for (auto &m_inst : ctx->model_cont) {
    if (m_inst.second.instance != nullptr) {
      m_inst.second.instance->modelClose();
      LOGI("Model is closed: %s\n", CVI_AI_GetModelName(m_inst.first));
      delete m_inst.second.instance;
      m_inst.second.instance = nullptr;
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
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_CloseModel(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  cviai_model_t &m_t = ctx->model_cont[config];
  if (m_t.instance == nullptr) {
    return CVIAI_ERR_CLOSE_MODEL;
  }

  m_t.instance->modelClose();
  LOGI("Model is closed: %s\n", CVI_AI_GetModelName(config));
  delete m_t.instance;
  m_t.instance = nullptr;
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_SelectDetectClass(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                 uint32_t num_selection, ...) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  va_list args;
  va_start(args, num_selection);

  std::vector<uint32_t> selected_classes;
  for (uint32_t i = 0; i < num_selection; i++) {
    uint32_t selected_class = va_arg(args, uint32_t);

    if (selected_class & CVI_AI_DET_GROUP_MASK_HEAD) {
      uint32_t group_start = (selected_class & CVI_AI_DET_GROUP_MASK_START) >> 16;
      uint32_t group_end = (selected_class & CVI_AI_DET_GROUP_MASK_END);
      for (uint32_t i = group_start; i <= group_end; i++) {
        selected_classes.push_back(i);
      }
    } else {
      if (selected_class >= CVI_AI_DET_TYPE_END) {
        LOGE("Invalid class id: %d\n", selected_class);
        return CVIAI_ERR_INVALID_ARGS;
      }
      selected_classes.push_back(selected_class);
    }
  }

  Core *instance = getInferenceInstance(config, ctx);
  if (instance != nullptr) {
    // TODO: only supports MobileDetV2 and YOLOX for now
    if (MobileDetV2 *mdetv2 = dynamic_cast<MobileDetV2 *>(instance)) {
      mdetv2->select_classes(selected_classes);
    }
#ifndef SIMPLY_MODEL
    else if (YoloX *yolox = dynamic_cast<YoloX *>(instance)) {
      yolox->select_classes(selected_classes);
    }
#endif
    else {
      LOGW("CVI_AI_SelectDetectClass only supports MobileDetV2 and YOLOX model for now.\n");
    }
  } else {
    LOGE("Failed to create model: %s\n", CVI_AI_GetModelName(config));
    return CVIAI_ERR_OPEN_MODEL;
  }
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_GetVpssChnConfig(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                const CVI_U32 frameWidth, const CVI_U32 frameHeight,
                                const CVI_U32 idx, cvai_vpssconfig_t *chnConfig) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  cviai::Core *instance = getInferenceInstance(config, ctx);
  if (instance == nullptr) {
    LOGE("Instance is null.\n");
    return CVIAI_ERR_OPEN_MODEL;
  }

  return instance->getChnConfig(frameWidth, frameHeight, idx, chnConfig);
}

CVI_S32 CVI_AI_EnalbeDumpInput(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                               const char *dump_path, bool enable) {
  CVI_S32 ret = CVIAI_SUCCESS;
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  cviai::Core *instance = getInferenceInstance(config, ctx);
  if (instance == nullptr) {
    LOGE("Instance is null.\n");
    return CVIAI_ERR_OPEN_MODEL;
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
#define DEFINE_INF_FUNC_F1_P1(func_name, class_name, model_index, arg_type)                   \
  CVI_S32 func_name(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, arg_type arg1) {  \
    TRACE_EVENT("cviai_core", #func_name);                                                    \
    cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);                            \
    class_name *obj = dynamic_cast<class_name *>(getInferenceInstance(model_index, ctx));     \
    if (obj == nullptr) {                                                                     \
      LOGE("No instance found for %s.\n", #class_name);                                       \
      return CVIAI_ERR_OPEN_MODEL;                                                            \
    }                                                                                         \
    if (obj->isInitialized()) {                                                               \
      if (initVPSSIfNeeded(ctx, model_index) != CVI_SUCCESS) {                                \
        return CVIAI_ERR_INIT_VPSS;                                                           \
      } else {                                                                                \
        return obj->inference(frame, arg1);                                                   \
      }                                                                                       \
    } else {                                                                                  \
      LOGE("Model (%s)is not yet opened! Please call CVI_AI_OpenModel to initialize model\n", \
           CVI_AI_GetModelName(model_index));                                                 \
      return CVIAI_ERR_NOT_YET_INITIALIZED;                                                   \
    }                                                                                         \
  }

#define DEFINE_INF_FUNC_F1_P2(func_name, class_name, model_index, arg1_type, arg2_type)       \
  CVI_S32 func_name(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, arg1_type arg1,   \
                    arg2_type arg2) {                                                         \
    TRACE_EVENT("cviai_core", #func_name);                                                    \
    cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);                            \
    class_name *obj = dynamic_cast<class_name *>(getInferenceInstance(model_index, ctx));     \
    if (obj == nullptr) {                                                                     \
      LOGE("No instance found for %s.\n", #class_name);                                       \
      return CVIAI_ERR_OPEN_MODEL;                                                            \
    }                                                                                         \
    if (obj->isInitialized()) {                                                               \
      if (initVPSSIfNeeded(ctx, model_index) != CVI_SUCCESS) {                                \
        return CVIAI_ERR_INIT_VPSS;                                                           \
      } else {                                                                                \
        return obj->inference(frame, arg1, arg2);                                             \
      }                                                                                       \
    } else {                                                                                  \
      LOGE("Model (%s)is not yet opened! Please call CVI_AI_OpenModel to initialize model\n", \
           CVI_AI_GetModelName(model_index));                                                 \
      return CVIAI_ERR_NOT_YET_INITIALIZED;                                                   \
    }                                                                                         \
  }

#define DEFINE_INF_FUNC_F2_P1(func_name, class_name, model_index, arg_type)                   \
  CVI_S32 func_name(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame1,                  \
                    VIDEO_FRAME_INFO_S *frame2, arg_type arg1) {                              \
    TRACE_EVENT("cviai_core", #func_name);                                                    \
    cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);                            \
    class_name *obj = dynamic_cast<class_name *>(getInferenceInstance(model_index, ctx));     \
    if (obj == nullptr) {                                                                     \
      LOGE("No instance found for %s.\n", #class_name);                                       \
      return CVIAI_ERR_OPEN_MODEL;                                                            \
    }                                                                                         \
    if (obj->isInitialized()) {                                                               \
      if (initVPSSIfNeeded(ctx, model_index) != CVI_SUCCESS) {                                \
        return CVIAI_ERR_INIT_VPSS;                                                           \
      } else {                                                                                \
        return obj->inference(frame1, frame2, arg1);                                          \
      }                                                                                       \
    } else {                                                                                  \
      LOGE("Model (%s)is not yet opened! Please call CVI_AI_OpenModel to initialize model\n", \
           CVI_AI_GetModelName(model_index));                                                 \
      return CVIAI_ERR_NOT_YET_INITIALIZED;                                                   \
    }                                                                                         \
  }

#define DEFINE_INF_FUNC_F2_P2(func_name, class_name, model_index, arg1_type, arg2_type)       \
  CVI_S32 func_name(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame1,                  \
                    VIDEO_FRAME_INFO_S *frame2, arg1_type arg1, arg2_type arg2) {             \
    TRACE_EVENT("cviai_core", #func_name);                                                    \
    cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);                            \
    class_name *obj = dynamic_cast<class_name *>(getInferenceInstance(model_index, ctx));     \
    if (obj == nullptr) {                                                                     \
      LOGE("No instance found for %s.\n", #class_name);                                       \
      return CVIAI_ERR_OPEN_MODEL;                                                            \
    }                                                                                         \
    if (obj->isInitialized()) {                                                               \
      if (initVPSSIfNeeded(ctx, model_index) != CVI_SUCCESS) {                                \
        return CVIAI_ERR_INIT_VPSS;                                                           \
      } else {                                                                                \
        return obj->inference(frame1, frame2, arg1, arg2);                                    \
      }                                                                                       \
    } else {                                                                                  \
      LOGE("Model (%s)is not yet opened! Please call CVI_AI_OpenModel to initialize model\n", \
           CVI_AI_GetModelName(model_index));                                                 \
      return CVIAI_ERR_NOT_YET_INITIALIZED;                                                   \
    }                                                                                         \
  }

/**
 *  Define model inference function here.
 *
 *  IMPORTANT!!
 *  Please remember to register creator function in MODEL_CREATORS first, or AISDK cannot
 *  find a correct way to create model object.
 *
 */
#ifndef SIMPLY_MODEL
DEFINE_INF_FUNC_F1_P1(CVI_AI_RetinaFace, RetinaFace, CVI_AI_SUPPORTED_MODEL_RETINAFACE,
                      cvai_face_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_ScrFDFace, ScrFDFace, CVI_AI_SUPPORTED_MODEL_SCRFDFACE, cvai_face_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_FaceAttribute, FaceAttribute, CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE,
                      cvai_face_t *)
DEFINE_INF_FUNC_F1_P2(CVI_AI_FaceAttributeOne, FaceAttribute, CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE,
                      cvai_face_t *, int)
DEFINE_INF_FUNC_F1_P1(CVI_AI_FaceRecognition, FaceAttribute, CVI_AI_SUPPORTED_MODEL_FACERECOGNITION,
                      cvai_face_t *)
DEFINE_INF_FUNC_F1_P2(CVI_AI_FaceRecognitionOne, FaceAttribute,
                      CVI_AI_SUPPORTED_MODEL_FACERECOGNITION, cvai_face_t *, int)
DEFINE_INF_FUNC_F1_P1(CVI_AI_Yolov3, Yolov3, CVI_AI_SUPPORTED_MODEL_YOLOV3, cvai_object_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_YoloX, YoloX, CVI_AI_SUPPORTED_MODEL_YOLOX, cvai_object_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_SoundClassification, SoundClassification,
                      CVI_AI_SUPPORTED_MODEL_SOUNDCLASSIFICATION, int *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_SoundClassification_V2, SoundClassificationV2,
                      CVI_AI_SUPPORTED_MODEL_SOUNDCLASSIFICATION_V2, int *)
#endif
DEFINE_INF_FUNC_F1_P1(CVI_AI_MobileDetV2_Vehicle, MobileDetV2,
                      CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE, cvai_object_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_MobileDetV2_Pedestrian, MobileDetV2,
                      CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN, cvai_object_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_MobileDetV2_Person_Vehicle, MobileDetV2,
                      CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE, cvai_object_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_MobileDetV2_Person_Pets, MobileDetV2,
                      CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PERSON_PETS, cvai_object_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_MobileDetV2_COCO80, MobileDetV2,
                      CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_COCO80, cvai_object_t *)

CVI_S32 CVI_AI_CropImage(VIDEO_FRAME_INFO_S *srcFrame, cvai_image_t *p_dst, cvai_bbox_t *bbox,
                         bool cvtRGB888) {
  return CVIAI_ERR_NOT_YET_INITIALIZED;
}

CVI_S32 CVI_AI_CropImage_Exten(VIDEO_FRAME_INFO_S *srcFrame, cvai_image_t *p_dst, cvai_bbox_t *bbox,
                               bool cvtRGB888, float exten_ratio, float *offset_x,
                               float *offset_y) {
  return CVIAI_ERR_NOT_YET_INITIALIZED;
}

CVI_S32 CVI_AI_CropImage_Face(VIDEO_FRAME_INFO_S *srcFrame, cvai_image_t *p_dst,
                              cvai_face_info_t *face_info, bool align, bool cvtRGB888) {
  return CVIAI_ERR_NOT_YET_INITIALIZED;
}

#ifndef SIMPLY_MODEL
// Tracker

CVI_S32 CVI_AI_DeepSORT_Init(const cviai_handle_t handle, bool use_specific_counter) {
  TRACE_EVENT("cviai_core", "CVI_AI_DeepSORT_Init");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  DeepSORT *ds_tracker = ctx->ds_tracker;
  if (ds_tracker == nullptr) {
    LOGI("Init DeepSORT.\n");
    ctx->ds_tracker = new DeepSORT(use_specific_counter);
  } else {
    delete ds_tracker;
    LOGI("Re-init DeepSORT.\n");
    ctx->ds_tracker = new DeepSORT(use_specific_counter);
  }
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_DeepSORT_GetDefaultConfig(cvai_deepsort_config_t *ds_conf) {
  TRACE_EVENT("cviai_core", "CVI_AI_DeepSORT_GetDefaultConfig");
  cvai_deepsort_config_t default_conf = DeepSORT::get_DefaultConfig();
  memcpy(ds_conf, &default_conf, sizeof(cvai_deepsort_config_t));

  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_DeepSORT_GetConfig(const cviai_handle_t handle, cvai_deepsort_config_t *ds_conf,
                                  int cviai_obj_type) {
  TRACE_EVENT("cviai_core", "CVI_AI_DeepSORT_GetConfig");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  DeepSORT *ds_tracker = ctx->ds_tracker;
  if (ds_tracker == nullptr) {
    LOGE("Please initialize DeepSORT first.\n");
    return CVIAI_FAILURE;
  }
  return ds_tracker->getConfig(ds_conf, cviai_obj_type);
}

CVI_S32 CVI_AI_DeepSORT_SetConfig(const cviai_handle_t handle, cvai_deepsort_config_t *ds_conf,
                                  int cviai_obj_type, bool show_config) {
  TRACE_EVENT("cviai_core", "CVI_AI_DeepSORT_SetConfig");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  DeepSORT *ds_tracker = ctx->ds_tracker;
  if (ds_tracker == nullptr) {
    LOGE("Please initialize DeepSORT first.\n");
    return CVIAI_FAILURE;
  }
  return ds_tracker->setConfig(ds_conf, cviai_obj_type, show_config);
}

CVI_S32 CVI_AI_DeepSORT_CleanCounter(const cviai_handle_t handle) {
  TRACE_EVENT("cviai_core", "CVI_AI_DeepSORT_CleanCounter");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  DeepSORT *ds_tracker = ctx->ds_tracker;
  if (ds_tracker == nullptr) {
    LOGE("Please initialize DeepSORT first.\n");
    return CVIAI_FAILURE;
  }
  ds_tracker->cleanCounter();

  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_DeepSORT_Obj(const cviai_handle_t handle, cvai_object_t *obj,
                            cvai_tracker_t *tracker, bool use_reid) {
  TRACE_EVENT("cviai_core", "CVI_AI_DeepSORT_Obj");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  DeepSORT *ds_tracker = ctx->ds_tracker;
  if (ds_tracker == nullptr) {
    LOGE("Please initialize DeepSORT first.\n");
    return CVIAI_FAILURE;
  }
  return ctx->ds_tracker->track(obj, tracker, use_reid);
}

CVI_S32 CVI_AI_DeepSORT_Face(const cviai_handle_t handle, cvai_face_t *face,
                             cvai_tracker_t *tracker) {
  TRACE_EVENT("cviai_core", "CVI_AI_DeepSORT_Face");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  DeepSORT *ds_tracker = ctx->ds_tracker;
  if (ds_tracker == nullptr) {
    LOGE("Please initialize DeepSORT first.\n");
    return CVIAI_FAILURE;
  }
  return ctx->ds_tracker->track(face, tracker);
}
CVI_S32 CVI_AI_DeepSORT_FaceFusePed(const cviai_handle_t handle, cvai_face_t *face,
                                    cvai_object_t *obj, cvai_tracker_t *tracker_t) {
  TRACE_EVENT("cviai_core", "CVI_AI_DeepSORT_FaceFusePed");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  DeepSORT *ds_tracker = ctx->ds_tracker;
  ds_tracker->set_image_size(face->width, face->height);
  if (ds_tracker == nullptr) {
    LOGE("Please initialize DeepSORT first.\n");
    return CVI_FAILURE;
  }
  ctx->ds_tracker->track_fuse(obj, face, tracker_t);
  return CVI_SUCCESS;
}
CVI_S32 CVI_AI_DeepSORT_UpdateOutNum(const cviai_handle_t handle, cvai_tracker_t *tracker_t) {
  TRACE_EVENT("cviai_core", "CVI_AI_DeepSORT_FaceFusePed");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  DeepSORT *ds_tracker = ctx->ds_tracker;

  if (ds_tracker == nullptr) {
    LOGE("Please initialize DeepSORT first.\n");
    return CVI_FAILURE;
  }
  ctx->ds_tracker->update_out_num(tracker_t);
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_DeepSORT_DebugInfo_1(const cviai_handle_t handle, char *debug_info) {
  TRACE_EVENT("cviai_core", "CVI_AI_DeepSORT_DebugInfo_1");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  DeepSORT *ds_tracker = ctx->ds_tracker;
  if (ds_tracker == nullptr) {
    LOGE("Please initialize DeepSORT first.\n");
    return CVIAI_FAILURE;
  }
  std::string str_info;
  ctx->ds_tracker->get_TrackersInfo_UnmatchedLastTime(str_info);
  strncpy(debug_info, str_info.c_str(), 8192);

  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_DeepSORT_GetTracker_Inactive(const cviai_handle_t handle, cvai_tracker_t *tracker) {
  TRACE_EVENT("cviai_core", "CVI_AI_DeepSORT_GetTracker_Inactive");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  DeepSORT *ds_tracker = ctx->ds_tracker;
  if (ds_tracker == nullptr) {
    LOGE("Please initialize DeepSORT first.\n");
    return CVI_FAILURE;
  }
  return ctx->ds_tracker->get_trackers_inactive(tracker);
}

// Fall Detection

CVI_S32 CVI_AI_Fall(const cviai_handle_t handle, cvai_object_t *objects) {
  return CVIAI_ERR_NOT_YET_INITIALIZED;
}

// Others
CVI_S32 CVI_AI_TamperDetection(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                               float *moving_score) {
  return CVIAI_ERR_NOT_YET_INITIALIZED;
}
#endif
CVI_S32 CVI_AI_Set_MotionDetection_Background(const cviai_handle_t handle,
                                              VIDEO_FRAME_INFO_S *frame) {
  TRACE_EVENT("cviai_core", "CVI_AI_Set_MotionDetection_Background");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  MotionDetection *md_model = ctx->md_model;
  if (md_model == nullptr) {
    LOGD("Init Motion Detection.\n");
    createIVEHandleIfNeeded(ctx);
    ctx->md_model = new MotionDetection(ctx->ive_handle);
    return ctx->md_model->init(frame);
  }
  return ctx->md_model->update_background(frame);
}
CVI_S32 CVI_AI_Set_MotionDetection_ROI(const cviai_handle_t handle, MDROI_t *roi_s) {
  TRACE_EVENT("cviai_core", "CVI_AI_Set_MotionDetection_ROI");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  MotionDetection *md_model = ctx->md_model;
  if (md_model == nullptr) {
    LOGE("MD has not been inited\n");
    return CVIAI_FAILURE;
  }
  return ctx->md_model->set_roi(roi_s);
}

CVI_S32 CVI_AI_MotionDetection(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                               cvai_object_t *obj_meta, uint8_t threshold, double min_area) {
  TRACE_EVENT("cviai_core", "CVI_AI_MotionDetection");

  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  MotionDetection *md_model = ctx->md_model;
  if (md_model == nullptr) {
    LOGE(
        "Failed to do motion detection! Please invoke CVI_AI_Set_MotionDetection_Background to set "
        "background image first\n");
    return CVIAI_FAILURE;
  }

  std::vector<std::vector<float>> boxes;
  CVI_S32 ret = ctx->md_model->detect(frame, boxes, threshold, min_area);
  memset(obj_meta, 0, sizeof(cvai_object_t));
  size_t num_boxes = boxes.size();
  if (num_boxes > 0) {
    CVI_AI_MemAllocInit(num_boxes, obj_meta);
    obj_meta->height = frame->stVFrame.u32Height;
    obj_meta->width = frame->stVFrame.u32Width;
    obj_meta->rescale_type = RESCALE_RB;
    memset(obj_meta->info, 0, sizeof(cvai_object_info_t) * num_boxes);
    for (uint32_t i = 0; i < (uint32_t)num_boxes; ++i) {
      obj_meta->info[i].bbox.x1 = boxes[i][0];
      obj_meta->info[i].bbox.y1 = boxes[i][1];
      obj_meta->info[i].bbox.x2 = boxes[i][2];
      obj_meta->info[i].bbox.y2 = boxes[i][3];
      obj_meta->info[i].bbox.score = 0;
      obj_meta->info[i].classes = -1;
      memset(obj_meta->info[i].name, 0, sizeof(obj_meta->info[i].name));
    }
  }
  return ret;
}

#ifndef SIMPLY_MODEL
CVI_S32 CVI_AI_Get_SoundClassification_ClassesNum(const cviai_handle_t handle) {
  return CVIAI_ERR_NOT_YET_INITIALIZED;
}

CVI_S32 CVI_AI_Set_SoundClassification_Threshold(const cviai_handle_t handle, const float th) {
  return CVIAI_ERR_NOT_YET_INITIALIZED;
}
CVI_S32 CVI_AI_FaceFeatureExtract(const cviai_handle_t handle, const uint8_t *p_rgb_pack, int width,
                                  int height, int stride, cvai_face_info_t *p_face_info) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  FaceAttribute *inst = dynamic_cast<FaceAttribute *>(
      getInferenceInstance(CVI_AI_SUPPORTED_MODEL_FACERECOGNITION, ctx));
  if (inst == nullptr) {
    LOGE("No instance found for FaceAttribute\n");
    return CVI_FAILURE;
  }
  return inst->extract_face_feature(p_rgb_pack, width, height, stride, p_face_info);
}

CVI_S32 CVI_AI_Change_Img(const cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model_type,
                          VIDEO_FRAME_INFO_S *frame, VIDEO_FRAME_INFO_S **dst_frame,
                          PIXEL_FORMAT_E enDstFormat) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  cviai_model_t modelt = ctx->model_cont[model_type];
  if (modelt.instance == nullptr) {
    LOGE("model not initialized:%d\n", (int)model_type);
    return CVI_FAILURE;
  }

  VpssEngine *p_vpss_inst = modelt.instance->get_vpss_instance();
  if (p_vpss_inst == nullptr) {
    LOGE("vpssmodel not initialized:%d\n", (int)model_type);
    return CVI_FAILURE;
  }

  VIDEO_FRAME_INFO_S *f = new VIDEO_FRAME_INFO_S;
  memset(f, 0, sizeof(VIDEO_FRAME_INFO_S));
  modelt.instance->vpssChangeImage(frame, f, frame->stVFrame.u32Width, frame->stVFrame.u32Height,
                                   enDstFormat);
  *dst_frame = f;
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_Delete_Img(const cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model_type,
                          VIDEO_FRAME_INFO_S *p_f) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  cviai_model_t modelt = ctx->model_cont[model_type];
  if (modelt.instance == nullptr) {
    LOGE("model not initialized:%d\n", (int)model_type);
    return CVI_FAILURE;
  }
  VpssEngine *p_vpss_inst = modelt.instance->get_vpss_instance();

  if (p_vpss_inst == nullptr) {
    LOGE("vpssmodel not initialized:%d\n", (int)model_type);
    return CVI_FAILURE;
  }
  p_vpss_inst->releaseFrame(p_f, 0);
  delete p_f;
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_CropImage_With_VPSS(const cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model_type,
                                   VIDEO_FRAME_INFO_S *frame, const cvai_bbox_t *p_crop_box,
                                   cvai_image_t *p_dst) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  cviai_model_t modelt = ctx->model_cont[model_type];
  if (modelt.instance == nullptr) {
    LOGE("model not initialized:%d\n", (int)model_type);
    return CVI_FAILURE;
  }
  VpssEngine *p_vpss_inst = modelt.instance->get_vpss_instance();

  if (p_vpss_inst == nullptr) {
    LOGE("vpssmodel not initialized:%d\n", (int)model_type);
    return CVIAI_ERR_NOT_YET_INITIALIZED;
  }
  if (p_dst->pix_format != PIXEL_FORMAT_RGB_888) {
    LOGE("only PIXEL_FORMAT_RGB_888 format supported,got:%d\n", (int)p_dst->pix_format);
    return CVI_FAILURE;
  }

  VIDEO_FRAME_INFO_S *f = new VIDEO_FRAME_INFO_S;
  memset(f, 0, sizeof(VIDEO_FRAME_INFO_S));
  modelt.instance->vpssCropImage(frame, f, *p_crop_box, p_dst->width, p_dst->height,
                                 p_dst->pix_format);
  mmap_video_frame(f);

  int ret = CVI_SUCCESS;
  for (int i = 0; i < 3; i++) {
    if ((p_dst->pix[i] == 0 && f->stVFrame.pu8VirAddr[i] != 0) ||
        (p_dst->pix[i] != 0 && f->stVFrame.pu8VirAddr[i] == 0)) {
      LOGE("error,plane:%d,dst_addr:%p,video_frame_addr:%p", i, p_dst->pix[i],
           f->stVFrame.pu8VirAddr[i]);
      ret = CVI_FAILURE;
      break;
    }
    if (f->stVFrame.u32Length[i] > p_dst->length[i]) {
      LOGE("size overflow,plane:%d,dst_len:%u,video_frame_len:%u", i, p_dst->length[i],
           f->stVFrame.u32Length[i]);
      ret = CVI_FAILURE;
      break;
    }
    memcpy(p_dst->pix[i], f->stVFrame.pu8VirAddr[i], f->stVFrame.u32Length[i]);
  }
  unmap_video_frame(f);
  if (f->stVFrame.u64PhyAddr[0] != 0) {
    p_vpss_inst->releaseFrame(f, 0);
  }
  delete f;
  return ret;
}
CVI_S32 CVI_AI_CropResizeImage(const cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model_type,
                               VIDEO_FRAME_INFO_S *frame, const cvai_bbox_t *p_crop_box,
                               int dst_width, int dst_height, PIXEL_FORMAT_E enDstFormat,
                               VIDEO_FRAME_INFO_S **p_dst_img) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  cviai_model_t modelt = ctx->model_cont[model_type];
  if (modelt.instance == nullptr) {
    LOGE("model not initialized:%d\n", (int)model_type);
    return CVI_FAILURE;
  }
  VpssEngine *p_vpss_inst = modelt.instance->get_vpss_instance();

  if (p_vpss_inst == nullptr) {
    LOGE("vpssmodel not initialized:%d\n", (int)model_type);
    return CVIAI_ERR_NOT_YET_INITIALIZED;
  }

  VIDEO_FRAME_INFO_S *f = new VIDEO_FRAME_INFO_S;
  memset(f, 0, sizeof(VIDEO_FRAME_INFO_S));
  int ret =
      modelt.instance->vpssCropImage(frame, f, *p_crop_box, dst_width, dst_width, enDstFormat);
  *p_dst_img = f;
  return ret;
}
CVI_S32 CVI_AI_Copy_VideoFrameToImage(VIDEO_FRAME_INFO_S *f, cvai_image_t *p_dst) {
  mmap_video_frame(f);
  int ret = CVI_SUCCESS;
  for (int i = 0; i < 3; i++) {
    if ((p_dst->pix[i] == 0 && f->stVFrame.pu8VirAddr[i] != 0) ||
        (p_dst->pix[i] != 0 && f->stVFrame.pu8VirAddr[i] == 0)) {
      LOGE("error,plane:%d,dst_addr:%p,video_frame_addr:%p", i, p_dst->pix[i],
           f->stVFrame.pu8VirAddr[i]);
      ret = CVI_FAILURE;
      break;
    }
    if (f->stVFrame.u32Length[i] > p_dst->length[i]) {
      LOGE("size overflow,plane:%d,dst_len:%u,video_frame_len:%u", i, p_dst->length[i],
           f->stVFrame.u32Length[i]);
      ret = CVI_FAILURE;
      break;
    }
    memcpy(p_dst->pix[i], f->stVFrame.pu8VirAddr[i], f->stVFrame.u32Length[i]);
  }
  unmap_video_frame(f);
  return ret;
}
CVI_S32 CVI_AI_Resize_VideoFrame(const cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model_type,
                                 VIDEO_FRAME_INFO_S *frame, const int dst_w, const int dst_h,
                                 PIXEL_FORMAT_E dst_format, VIDEO_FRAME_INFO_S **dst_frame) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  cviai_model_t modelt = ctx->model_cont[model_type];
  if (modelt.instance == nullptr) {
    LOGE("model not initialized:%d\n", (int)model_type);
    return CVI_FAILURE;
  }
  VpssEngine *p_vpss_inst = modelt.instance->get_vpss_instance();

  if (p_vpss_inst == nullptr) {
    LOGE("vpssmodel not initialized:%d\n", (int)model_type);
    return CVIAI_ERR_NOT_YET_INITIALIZED;
  }

  VIDEO_FRAME_INFO_S *f = new VIDEO_FRAME_INFO_S;
  memset(f, 0, sizeof(VIDEO_FRAME_INFO_S));
  cvai_bbox_t bbox;
  bbox.x1 = 0;
  bbox.y1 = 0;
  bbox.x2 = frame->stVFrame.u32Width;
  bbox.y2 = frame->stVFrame.u32Height;
  VPSS_SCALE_COEF_E scale = VPSS_SCALE_COEF_BILINEAR;
  modelt.instance->vpssCropImage(frame, f, bbox, dst_w, dst_h, dst_format, scale);
  *dst_frame = f;
  return CVI_SUCCESS;
}
DLL_EXPORT CVI_S32 CVI_AI_Release_VideoFrame(const cviai_handle_t handle,
                                             CVI_AI_SUPPORTED_MODEL_E model_type,
                                             VIDEO_FRAME_INFO_S *frame, bool del_frame) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  cviai_model_t modelt = ctx->model_cont[model_type];
  if (modelt.instance == nullptr) {
    LOGE("model not initialized:%d\n", (int)model_type);
    return CVI_FAILURE;
  }
  VpssEngine *p_vpss_inst = modelt.instance->get_vpss_instance();

  if (p_vpss_inst == nullptr) {
    LOGE("vpssmodel not initialized:%d\n", (int)model_type);
    return CVIAI_ERR_NOT_YET_INITIALIZED;
  }

  if (frame->stVFrame.u64PhyAddr[0] != 0) {
    p_vpss_inst->releaseFrame(frame, 0);
  }
  if (del_frame) {
    delete frame;
  }
  return CVI_SUCCESS;
}
#endif