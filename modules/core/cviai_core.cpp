#include "version.hpp"

#include "alphapose/alphapose.hpp"
#include "core/core/cvai_errno.h"
#include "core/cviai_core.h"
#include "core/cviai_types_mem_internal.h"
#include "cviai_core_internal.hpp"
#include "cviai_experimental.h"
#include "cviai_log.hpp"
#include "cviai_perfetto.h"
#include "cviai_trace.hpp"
#include "deepsort/cvi_deepsort.hpp"
#include "eye_classification/eye_classification.hpp"
#include "face_attribute/face_attribute.hpp"
#include "face_landmarker/face_landmark_det3.hpp"
#include "face_landmarker/face_landmarker.hpp"
#include "face_landmarker/face_landmarker_det2.hpp"
#include "face_mask_detection/retinaface_yolox.hpp"
#include "face_quality/face_quality.hpp"
#include "fall_detection/fall_det_monitor.hpp"
#include "fall_detection/fall_detection.hpp"
#include "hand_classification/hand_classification.hpp"
#include "hand_keypoint/hand_keypoint.hpp"
#include "hand_keypoint_classification/hand_keypoint_classification.hpp"
#include "human_keypoints_detection/simcc/simcc.hpp"
#include "human_keypoints_detection/yolov8_pose/yolov8_pose.hpp"
#include "image_classification/image_classification.hpp"
#include "incar_object_detection/incar_object_detection.hpp"
#include "license_plate_detection/license_plate_detection.hpp"
#include "license_plate_recognition/license_plate_recognition.hpp"
#include "license_plate_recognitionv2/license_plate_recognitionv2.hpp"
#include "liveness/ir_liveness.hpp"
#include "liveness/liveness.hpp"
#include "mask_classification/mask_classification.hpp"
#include "mask_face_recognition/mask_face_recognition.hpp"
#include "motion_detection/md.hpp"
#include "object_detection/mobiledetv2/mobiledetv2.hpp"
#include "object_detection/ppyoloe/ppyoloe.hpp"
#include "object_detection/yolo/yolo.hpp"
#include "object_detection/yolov3/yolov3.hpp"
#include "object_detection/yolov5/yolov5.hpp"
#include "object_detection/yolov6/yolov6.hpp"
#include "object_detection/yolov8/yolov8.hpp"
#include "object_detection/yolox/yolox.hpp"
#include "osnet/osnet.hpp"
#include "retina_face/retina_face.hpp"
#include "retina_face/scrfd_face.hpp"
#include "segmentation/deeplabv3.hpp"
#include "smoke_classification/smoke_classification.hpp"
#include "sound_classification/sound_classification.hpp"
#include "sound_classification/sound_classification_v2.hpp"

#include <stdarg.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "thermal_face_detection/thermal_face.hpp"
#include "thermal_person_detection/thermal_person.hpp"
#include "utils/core_utils.hpp"
#include "utils/image_utils.hpp"
#include "yawn_classification/yawn_classification.hpp"

using namespace std;
using namespace cviai;

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

static int createIVEHandleIfNeeded(cviai_context_t *ctx) {
  if (ctx->ive_handle == nullptr) {
    ctx->ive_handle = new ive::IVE;
    if (ctx->ive_handle->init() != CVI_SUCCESS) {
      LOGC("IVE handle init failed, please insmod cv18?x_ive.ko.\n");
      return CVI_FAILURE;
    }
  }
  return CVI_SUCCESS;
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
    {CVI_AI_SUPPORTED_MODEL_FACEQUALITY, CREATOR(FaceQuality)},
    {CVI_AI_SUPPORTED_MODEL_THERMALFACE, CREATOR(ThermalFace)},
    {CVI_AI_SUPPORTED_MODEL_THERMALPERSON, CREATOR(ThermalPerson)},
    {CVI_AI_SUPPORTED_MODEL_LIVENESS, CREATOR(Liveness)},
    {CVI_AI_SUPPORTED_MODEL_IRLIVENESS, CREATOR(IrLiveness)},
    {CVI_AI_SUPPORTED_MODEL_MASKCLASSIFICATION, CREATOR(MaskClassification)},
    {CVI_AI_SUPPORTED_MODEL_HANDCLASSIFICATION, CREATOR(HandClassification)},
    {CVI_AI_SUPPORTED_MODEL_HAND_KEYPOINT, CREATOR(HandKeypoint)},
    {CVI_AI_SUPPORTED_MODEL_HAND_KEYPOINT_CLASSIFICATION, CREATOR(HandKeypointClassification)},
    {CVI_AI_SUPPORTED_MODEL_YOLOV3, CREATOR(Yolov3)},
    {CVI_AI_SUPPORTED_MODEL_YOLOV5, CREATOR(Yolov5)},
    {CVI_AI_SUPPORTED_MODEL_YOLOV6, CREATOR(Yolov6)},
    {CVI_AI_SUPPORTED_MODEL_YOLO, CREATOR(Yolo)},
    {CVI_AI_SUPPORTED_MODEL_YOLOX, CREATOR(YoloX)},
    {CVI_AI_SUPPORTED_MODEL_PPYOLOE, CREATOR(PPYoloE)},
    {CVI_AI_SUPPORTED_MODEL_FACEMASKDETECTION, CREATOR(RetinafaceYolox)},
    {CVI_AI_SUPPORTED_MODEL_OSNET, CREATOR(OSNet)},
    {CVI_AI_SUPPORTED_MODEL_SOUNDCLASSIFICATION, CREATOR(SoundClassification)},
    {CVI_AI_SUPPORTED_MODEL_SOUNDCLASSIFICATION_V2, CREATOR(SoundClassificationV2)},
    {CVI_AI_SUPPORTED_MODEL_WPODNET, CREATOR(LicensePlateDetection)},
    {CVI_AI_SUPPORTED_MODEL_LP_RECONGNITION, CREATOR(LicensePlateRecognitionV2)},
    {CVI_AI_SUPPORTED_MODEL_DEEPLABV3, CREATOR(Deeplabv3)},
    {CVI_AI_SUPPORTED_MODEL_ALPHAPOSE, CREATOR(AlphaPose)},
    {CVI_AI_SUPPORTED_MODEL_EYECLASSIFICATION, CREATOR(EyeClassification)},
    {CVI_AI_SUPPORTED_MODEL_YAWNCLASSIFICATION, CREATOR(YawnClassification)},
    {CVI_AI_SUPPORTED_MODEL_SMOKECLASSIFICATION, CREATOR(SmokeClassification)},
    {CVI_AI_SUPPORTED_MODEL_FACELANDMARKER, CREATOR(FaceLandmarker)},
    {CVI_AI_SUPPORTED_MODEL_FACELANDMARKERDET2, CREATOR(FaceLandmarkerDet2)},
    {CVI_AI_SUPPORTED_MODEL_INCAROBJECTDETECTION, CREATOR(IncarObjectDetection)},
    {CVI_AI_SUPPORTED_MODEL_MASKFACERECOGNITION, CREATOR(MaskFaceRecognition)},
    {CVI_AI_SUPPORTED_MODEL_SCRFDFACE, CREATOR(ScrFDFace)},
    {CVI_AI_SUPPORTED_MODEL_RETINAFACE, CREATOR_P1(RetinaFace, PROCESS, CAFFE)},
    {CVI_AI_SUPPORTED_MODEL_RETINAFACE_IR, CREATOR_P1(RetinaFace, PROCESS, PYTORCH)},
    {CVI_AI_SUPPORTED_MODEL_RETINAFACE_HARDHAT, CREATOR_P1(RetinaFace, PROCESS, PYTORCH)},
    {CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE, CREATOR_P1(FaceAttribute, bool, true)},
    {CVI_AI_SUPPORTED_MODEL_FACERECOGNITION, CREATOR_P1(FaceAttribute, bool, false)},
    {CVI_AI_SUPPORTED_MODEL_LPRNET_TW, CREATOR_P1(LicensePlateRecognition, LP_FORMAT, TAIWAN)},
    {CVI_AI_SUPPORTED_MODEL_LPRNET_CN, CREATOR_P1(LicensePlateRecognition, LP_FORMAT, CHINA)},
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
    {CVI_AI_SUPPORTED_MODEL_SIMCC_POSE, CREATOR(Simcc)},
    {CVI_AI_SUPPORTED_MODEL_LANDMARK_DET3, CREATOR(FaceLandmarkDet3)},
    {CVI_AI_SUPPORTED_MODEL_IMAGE_CLASSIFICATION, CREATOR(ImageClassification)},
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
  delete ctx->ds_tracker;
  delete ctx->td_model;
  delete ctx->md_model;

  if (ctx->ive_handle) {
    ctx->ive_handle->destroy();
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
    // create custom instance here
    if (index == CVI_AI_SUPPORTED_MODEL_HAND_DETECTION) {
      YoloV8Detection *p_yolov8 = new YoloV8Detection();
      p_yolov8->setBranchChannel(64, 1);
      m_t.instance = p_yolov8;
      LOGI("create hand model");
    } else if (index == CVI_AI_SUPPORTED_MODEL_PERSON_PETS_DETECTION) {
      YoloV8Detection *p_yolov8 = new YoloV8Detection();
      p_yolov8->setBranchChannel(64, 3);  // three types
      m_t.instance = p_yolov8;
      LOGI("create personpet model");
    } else if (index == CVI_AI_SUPPORTED_MODEL_YOLOV8_DETECTION) {
      YoloV8Detection *p_yolov8 = new YoloV8Detection();
      p_yolov8->setBranchChannel(64, 80);  // three types
      m_t.instance = p_yolov8;
      LOGI("create yolov8 model");
    } else if (index == CVI_AI_SUPPORTED_MODEL_PERSON_VEHICLE_DETECTION) {
      YoloV8Detection *p_yolov8 = new YoloV8Detection();
      p_yolov8->setBranchChannel(64, 7);  // seven types
      m_t.instance = p_yolov8;
      LOGI("create vehicle model");
    } else if (index == CVI_AI_SUPPORTED_MODEL_HAND_FACE_PERSON_DETECTION) {
      YoloV8Detection *p_yolov8 = new YoloV8Detection();
      p_yolov8->setBranchChannel(64, 3);  // 3 types:hand,face,person
      m_t.instance = p_yolov8;
      LOGI("create vehicle model");
    } else if (index == CVI_AI_SUPPORTED_MODEL_HEAD_PERSON_DETECTION) {
      YoloV8Detection *p_yolov8 = new YoloV8Detection();
      p_yolov8->setBranchChannel(64, 2);  // 2 types:head,person
      m_t.instance = p_yolov8;
      LOGI("create headperson model");
    } else if (index == CVI_AI_SUPPORTED_MODEL_YOLOV8POSE) {
      YoloV8Pose *p_yolov8pose = new YoloV8Pose();
      p_yolov8pose->setBranchChannel(64, 17, 1);  // 17 keypoints
      m_t.instance = p_yolov8pose;
      LOGI("create yolov8 pose model");
    } else if (index == CVI_AI_SUPPORTED_MODEL_LP_DETECTION) {
      YoloV8Pose *p_yolov8pose = new YoloV8Pose();
      p_yolov8pose->setBranchChannel(64, 4, 2);  // 4 keypoints
      m_t.instance = p_yolov8pose;
      LOGI("create yolov8 pl model");
    } else {
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
  }
  m_t.instance->setVpssEngine(ctx->vec_vpss_engine[m_t.vpss_thread]);
  m_t.instance->setVpssTimeout(ctx->vpss_timeout_value);
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

CVI_S32 CVI_AI_SetModelNmsThreshold(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                    float threshold) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  Core *instance = getInferenceInstance(config, ctx);
  if (instance != nullptr) {
    instance->setModelNmsThreshold(threshold);
  } else {
    LOGE("Cannot create model: %s\n", CVI_AI_GetModelName(config));
    return CVIAI_ERR_OPEN_MODEL;
  }
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_GetModelNMmsThreshold(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                     float *threshold) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  Core *instance = getInferenceInstance(config, ctx);
  if (instance != nullptr) {
    *threshold = instance->getModelNmsThreshold();
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
    } else {
      LOGW("CVI_AI_SelectDetectClass only supports MobileDetV2for now.\n");
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
DEFINE_INF_FUNC_F1_P1(CVI_AI_RetinaFace, RetinaFace, CVI_AI_SUPPORTED_MODEL_RETINAFACE,
                      cvai_face_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_ScrFDFace, ScrFDFace, CVI_AI_SUPPORTED_MODEL_SCRFDFACE, cvai_face_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_FLDet3, FaceLandmarkDet3, CVI_AI_SUPPORTED_MODEL_LANDMARK_DET3,
                      cvai_face_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_RetinaFace_IR, RetinaFace, CVI_AI_SUPPORTED_MODEL_RETINAFACE_IR,
                      cvai_face_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_RetinaFace_Hardhat, RetinaFace,
                      CVI_AI_SUPPORTED_MODEL_RETINAFACE_HARDHAT, cvai_face_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_ThermalFace, ThermalFace, CVI_AI_SUPPORTED_MODEL_THERMALFACE,
                      cvai_face_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_ThermalPerson, ThermalPerson, CVI_AI_SUPPORTED_MODEL_THERMALPERSON,
                      cvai_object_t *)
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
DEFINE_INF_FUNC_F1_P2(CVI_AI_FaceQuality, FaceQuality, CVI_AI_SUPPORTED_MODEL_FACEQUALITY,
                      cvai_face_t *, bool *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_MaskClassification, MaskClassification,
                      CVI_AI_SUPPORTED_MODEL_MASKCLASSIFICATION, cvai_face_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_HandClassification, HandClassification,
                      CVI_AI_SUPPORTED_MODEL_HANDCLASSIFICATION, cvai_object_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_HandKeypoint, HandKeypoint, CVI_AI_SUPPORTED_MODEL_HAND_KEYPOINT,
                      cvai_handpose21_meta_ts *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_HandKeypointClassification, HandKeypointClassification,
                      CVI_AI_SUPPORTED_MODEL_HAND_KEYPOINT_CLASSIFICATION, cvai_handpose21_meta_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_FaceMaskDetection, RetinafaceYolox,
                      CVI_AI_SUPPORTED_MODEL_FACEMASKDETECTION, cvai_face_t *)
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
DEFINE_INF_FUNC_F1_P1(CVI_AI_Yolov3, Yolov3, CVI_AI_SUPPORTED_MODEL_YOLOV3, cvai_object_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_Yolov5, Yolov5, CVI_AI_SUPPORTED_MODEL_YOLOV5, cvai_object_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_Yolov6, Yolov6, CVI_AI_SUPPORTED_MODEL_YOLOV6, cvai_object_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_Yolo, Yolo, CVI_AI_SUPPORTED_MODEL_YOLO, cvai_object_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_YoloX, YoloX, CVI_AI_SUPPORTED_MODEL_YOLOX, cvai_object_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_PPYoloE, PPYoloE, CVI_AI_SUPPORTED_MODEL_PPYOLOE, cvai_object_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_OSNet, OSNet, CVI_AI_SUPPORTED_MODEL_OSNET, cvai_object_t *)
DEFINE_INF_FUNC_F1_P2(CVI_AI_OSNetOne, OSNet, CVI_AI_SUPPORTED_MODEL_OSNET, cvai_object_t *, int)
DEFINE_INF_FUNC_F1_P1(CVI_AI_SoundClassification, SoundClassification,
                      CVI_AI_SUPPORTED_MODEL_SOUNDCLASSIFICATION, int *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_SoundClassification_V2, SoundClassificationV2,
                      CVI_AI_SUPPORTED_MODEL_SOUNDCLASSIFICATION_V2, int *)
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
DEFINE_INF_FUNC_F1_P1(CVI_AI_FaceLandmarkerDet2, FaceLandmarkerDet2,
                      CVI_AI_SUPPORTED_MODEL_FACELANDMARKERDET2, cvai_face_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_IncarObjectDetection, IncarObjectDetection,
                      CVI_AI_SUPPORTED_MODEL_INCAROBJECTDETECTION, cvai_face_t *)
DEFINE_INF_FUNC_F2_P2(CVI_AI_Liveness, Liveness, CVI_AI_SUPPORTED_MODEL_LIVENESS, cvai_face_t *,
                      cvai_face_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_IrLiveness, IrLiveness, CVI_AI_SUPPORTED_MODEL_IRLIVENESS,
                      cvai_face_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_Yolov8_Pose, YoloV8Pose, CVI_AI_SUPPORTED_MODEL_YOLOV8POSE,
                      cvai_object_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_License_Plate_Detectionv2, YoloV8Pose,
                      CVI_AI_SUPPORTED_MODEL_LP_DETECTION, cvai_object_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_License_Plate_Recognitionv2, LicensePlateRecognitionV2,
                      CVI_AI_SUPPORTED_MODEL_LP_RECONGNITION, cvai_object_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_Simcc_Pose, Simcc, CVI_AI_SUPPORTED_MODEL_SIMCC_POSE, cvai_object_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_Image_Classification, ImageClassification,
                      CVI_AI_SUPPORTED_MODEL_IMAGE_CLASSIFICATION, cvai_class_meta_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_Hand_Detection, YoloV8Detection, CVI_AI_SUPPORTED_MODEL_HAND_DETECTION,
                      cvai_object_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_PersonPet_Detection, YoloV8Detection,
                      CVI_AI_SUPPORTED_MODEL_PERSON_PETS_DETECTION, cvai_object_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_YOLOV8_Detection, YoloV8Detection,
                      CVI_AI_SUPPORTED_MODEL_YOLOV8_DETECTION, cvai_object_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_HandFacePerson_Detection, YoloV8Detection,
                      CVI_AI_SUPPORTED_MODEL_HAND_FACE_PERSON_DETECTION, cvai_object_t *)
DEFINE_INF_FUNC_F1_P1(CVI_AI_HeadPerson_Detection, YoloV8Detection,
                      CVI_AI_SUPPORTED_MODEL_HEAD_PERSON_DETECTION, cvai_object_t *)

CVI_S32 CVI_AI_CropImage(VIDEO_FRAME_INFO_S *srcFrame, cvai_image_t *dst, cvai_bbox_t *bbox,
                         bool cvtRGB888) {
  TRACE_EVENT("cviai_core", "CVI_AI_CropImage");
  return crop_image(srcFrame, dst, bbox, cvtRGB888);
}

CVI_S32 CVI_AI_CropImage_Exten(VIDEO_FRAME_INFO_S *srcFrame, cvai_image_t *dst, cvai_bbox_t *bbox,
                               bool cvtRGB888, float exten_ratio, float *offset_x,
                               float *offset_y) {
  TRACE_EVENT("cviai_core", "CVI_AI_CropImage_Exten");
  return crop_image_exten(srcFrame, dst, bbox, cvtRGB888, exten_ratio, offset_x, offset_y);
}

CVI_S32 CVI_AI_CropImage_Face(VIDEO_FRAME_INFO_S *srcFrame, cvai_image_t *dst,
                              cvai_face_info_t *face_info, bool align, bool cvtRGB888) {
  TRACE_EVENT("cviai_core", "CVI_AI_CropImage_Face");
  return crop_image_face(srcFrame, dst, face_info, align, cvtRGB888);
}

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

CVI_S32 CVI_AI_DeepSORT_Head_FusePed(const cviai_handle_t handle, cvai_object_t *obj,
                                     cvai_tracker_t *tracker_t, bool use_reid, cvai_object_t *head,
                                     cvai_object_t *ped,
                                     const cvai_counting_line_t *counting_line_t,
                                     const randomRect *rect) {
  TRACE_EVENT("cviai_core", "CVI_AI_DeepSORT_HeadFusePed");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  DeepSORT *ds_tracker = ctx->ds_tracker;
  ds_tracker->set_image_size(obj->width, obj->height);
  if (ds_tracker == nullptr) {
    LOGE("Please initialize DeepSORT first.\n");
    return CVI_FAILURE;
  }
  ctx->ds_tracker->track_headfuse(obj, tracker_t, use_reid, head, ped, counting_line_t, rect);
  return CVI_SUCCESS;
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

CVI_S32 CVI_AI_DeepSORT_Byte(const cviai_handle_t handle, cvai_object_t *obj,
                             cvai_tracker_t *tracker, bool use_reid) {
  TRACE_EVENT("cviai_core", "CVI_AI_DeepSORT_Obj");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  DeepSORT *ds_tracker = ctx->ds_tracker;
  if (ds_tracker == nullptr) {
    LOGE("Please initialize DeepSORT first.\n");
    return CVIAI_FAILURE;
  }
  return ctx->ds_tracker->byte_track(obj, tracker, use_reid);
}
DLL_EXPORT CVI_S32 CVI_AI_DeepSORT_Obj_Cross(const cviai_handle_t handle, cvai_object_t *obj,
                                             cvai_tracker_t *tracker, bool use_reid,
                                             const cvai_counting_line_t *cross_line_t,
                                             const randomRect *rect) {
  TRACE_EVENT("cviai_core", "CVI_AI_DeepSORT_Obj");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  DeepSORT *ds_tracker = ctx->ds_tracker;
  if (ds_tracker == nullptr) {
    LOGE("Please initialize DeepSORT first.\n");
    return CVIAI_FAILURE;
  }
  return ctx->ds_tracker->track_cross(obj, tracker, use_reid, cross_line_t, rect);
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
DLL_EXPORT CVI_S32 CVI_AI_DeepSORT_FaceFusePed(const cviai_handle_t handle, cvai_face_t *face,
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
  TRACE_EVENT("cviai_core", "CVI_AI_Fall");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  FallMD *fall_model = ctx->fall_model;
  if (fall_model == nullptr) {
    LOGD("Init Fall Detection Model.\n");
    ctx->fall_model = new FallMD();
    ctx->fall_model->detect(objects);
    return CVIAI_SUCCESS;
  }
  return ctx->fall_model->detect(objects);
}

// New Fall Detection

CVI_S32 CVI_AI_Fall_Monitor(const cviai_handle_t handle, cvai_object_t *objects) {
  TRACE_EVENT("cviai_core", "CVI_AI_Fall_Monitor");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  FallDetMonitor *fall_monitor_model = ctx->fall_monitor_model;
  if (fall_monitor_model == nullptr) {
    LOGD("Init Fall Detection Model.\n");
    ctx->fall_monitor_model = new FallDetMonitor();
    ctx->fall_monitor_model->monitor(objects);
    return CVIAI_SUCCESS;
  }
  return ctx->fall_monitor_model->monitor(objects);
}

CVI_S32 CVI_AI_Set_Fall_FPS(const cviai_handle_t handle, float fps) {
  TRACE_EVENT("cviai_core", "CVI_AI_Fall_Monitor");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  FallDetMonitor *fall_monitor_model = ctx->fall_monitor_model;
  if (fall_monitor_model == nullptr) {
    LOGD("Init Fall Detection Model.\n");
    ctx->fall_monitor_model = new FallDetMonitor();
    ctx->fall_monitor_model->set_fps(fps);
    return CVIAI_SUCCESS;
  }
  return ctx->fall_monitor_model->set_fps(fps);
}

// Others
CVI_S32 CVI_AI_TamperDetection(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                               float *moving_score) {
  TRACE_EVENT("cviai_core", "CVI_AI_TamperDetection");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  TamperDetectorMD *td_model = ctx->td_model;
  if (td_model == nullptr) {
    LOGD("Init Tamper Detection Model.\n");
    createIVEHandleIfNeeded(ctx);
    ctx->td_model = new TamperDetectorMD(ctx->ive_handle, frame, (float)0.05, (int)10);

    *moving_score = -1.0;
    return CVIAI_SUCCESS;
  }
  return ctx->td_model->detect(frame, moving_score);
}

CVI_S32 CVI_AI_Set_MotionDetection_Background(const cviai_handle_t handle,
                                              VIDEO_FRAME_INFO_S *frame) {
  TRACE_EVENT("cviai_core", "CVI_AI_Set_MotionDetection_Background");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  MotionDetection *md_model = ctx->md_model;
  if (md_model == nullptr) {
    LOGD("Init Motion Detection.\n");
    if (createIVEHandleIfNeeded(ctx) == CVIAI_FAILURE) {
      return CVIAI_FAILURE;
    }
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
        "background image first.\n");
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

CVI_S32 CVI_AI_FaceFeatureExtract(const cviai_handle_t handle, const uint8_t *p_rgb_pack, int width,
                                  int height, int stride, cvai_face_info_t *p_face_info) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  FaceAttribute *inst = dynamic_cast<FaceAttribute *>(
      getInferenceInstance(CVI_AI_SUPPORTED_MODEL_FACERECOGNITION, ctx));
  if (inst == nullptr) {
    LOGE("No instance found for FaceAttribute\n");
    return CVI_FAILURE;
  }
  if (inst->isInitialized()) {
    if (initVPSSIfNeeded(ctx, CVI_AI_SUPPORTED_MODEL_FACERECOGNITION) != CVI_SUCCESS) {
      return CVIAI_ERR_INIT_VPSS;
    }
  } else {
    LOGE("Model (%s)is not yet opened! Please call CVI_AI_OpenModel to initialize model\n",
         CVI_AI_GetModelName(CVI_AI_SUPPORTED_MODEL_FACERECOGNITION));
    return CVIAI_ERR_NOT_YET_INITIALIZED;
  }
  return inst->extract_face_feature(p_rgb_pack, width, height, stride, p_face_info);
}

CVI_S32 CVI_AI_Get_SoundClassification_ClassesNum(const cviai_handle_t handle) {
  TRACE_EVENT("cviai_core", "CVI_AI_Get_SoundClassification_ClassesNum");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  SoundClassification *sc_model = dynamic_cast<SoundClassification *>(
      getInferenceInstance(CVI_AI_SUPPORTED_MODEL_SOUNDCLASSIFICATION, ctx));
  if (sc_model == nullptr) {
    LOGE("No instance found for SoundClassification.\n");
    return CVIAI_ERR_OPEN_MODEL;
  }
  if (sc_model->isInitialized()) {
    return sc_model->getClassesNum();
  } else {
    LOGE("Model (%s)is not yet opened! Please call CVI_AI_OpenModel to initialize model\n",
         CVI_AI_GetModelName(CVI_AI_SUPPORTED_MODEL_SOUNDCLASSIFICATION));
    return CVIAI_ERR_NOT_YET_INITIALIZED;
  }
}

CVI_S32 CVI_AI_Set_SoundClassification_Threshold(const cviai_handle_t handle, const float th) {
  TRACE_EVENT("cviai_core", "CVI_AI_Set_SoundClassification_Threshold");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  SoundClassification *sc_model = dynamic_cast<SoundClassification *>(
      getInferenceInstance(CVI_AI_SUPPORTED_MODEL_SOUNDCLASSIFICATION, ctx));
  if (sc_model == nullptr) {
    LOGE("No instance found for SoundClassification.\n");
    return CVIAI_ERR_OPEN_MODEL;
  }
  if (sc_model->isInitialized()) {
    return sc_model->setThreshold(th);
  } else {
    LOGE("Model (%s)is not yet opened! Please call CVI_AI_OpenModel to initialize model\n",
         CVI_AI_GetModelName(CVI_AI_SUPPORTED_MODEL_SOUNDCLASSIFICATION));
    return CVIAI_ERR_NOT_YET_INITIALIZED;
  }
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
  VPSS_SCALE_COEF_E scale = VPSS_SCALE_COEF_NEAREST;
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

CVI_S32 CVI_AI_PersonVehicle_Detection(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                       cvai_object_t *obj_meta) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  YoloV8Detection *yolo_model = dynamic_cast<YoloV8Detection *>(
      getInferenceInstance(CVI_AI_SUPPORTED_MODEL_PERSON_VEHICLE_DETECTION, ctx));
  if (yolo_model == nullptr) {
    LOGE("No instance found for CVI_AI_Hand_Detection.\n");
    return CVI_FAILURE;
  }
  LOGI("got yolov8 instance\n");
  if (yolo_model->isInitialized()) {
    if (initVPSSIfNeeded(ctx, CVI_AI_SUPPORTED_MODEL_PERSON_VEHICLE_DETECTION) != CVI_SUCCESS) {
      return CVIAI_ERR_INIT_VPSS;
    } else {
      int ret = yolo_model->inference(frame, obj_meta);
      if (ret == CVIAI_SUCCESS) {
        for (uint32_t i = 0; i < obj_meta->size; i++) {
          if (obj_meta->info[i].classes == 4) {
            obj_meta->info[i].classes = 0;  // person
          } else if (obj_meta->info[i].classes == 0 || obj_meta->info[i].classes == 1 ||
                     obj_meta->info[i].classes == 2) {
            obj_meta->info[i].classes = 1;  // motor vehicle
          } else {
            obj_meta->info[i].classes = 2;  // non-motor vehicle
          }
        }
      }
      return ret;
    }
  } else {
    LOGE("Model (%s)is not yet opened! Please call CVI_AI_OpenModel to initialize model\n",
         CVI_AI_GetModelName(CVI_AI_SUPPORTED_MODEL_PERSON_VEHICLE_DETECTION));
    return CVIAI_ERR_NOT_YET_INITIALIZED;
  }
}

CVI_S32 CVI_AI_Set_YOLOV5_Param(const cviai_handle_t handle, YoloPreParam *p_preprocess_cfg,
                                YoloAlgParam *p_yolo_param) {
  printf("enter CVI_AI_Set_YOLOV5_Param...\n");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  Yolov5 *yolov5_model =
      dynamic_cast<Yolov5 *>(getInferenceInstance(CVI_AI_SUPPORTED_MODEL_YOLOV5, ctx));
  if (yolov5_model == nullptr) {
    LOGE("No instance found for yolov5 detection.\n");
    return CVI_FAILURE;
  }
  LOGI("got yolov5 instance\n");
  if (p_preprocess_cfg == nullptr || p_yolo_param == nullptr) {
    LOGE("p_preprocess_cfg or p_yolov5_param can not be nullptr.\n");
    return CVI_FAILURE;
  }

  yolov5_model->set_param(p_preprocess_cfg, p_yolo_param);
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_Set_YOLOV6_Param(const cviai_handle_t handle, YoloPreParam *p_preprocess_cfg,
                                YoloAlgParam *p_yolo_param) {
  printf("enter CVI_AI_Set_YOLOV6_Param...\n");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  Yolov6 *yolov6_model =
      dynamic_cast<Yolov6 *>(getInferenceInstance(CVI_AI_SUPPORTED_MODEL_YOLOV6, ctx));
  if (yolov6_model == nullptr) {
    LOGE("No instance found for yolov6 detection.\n");
    return CVI_FAILURE;
  }
  LOGI("got yolov6 instance\n");
  if (p_preprocess_cfg == nullptr || p_yolo_param == nullptr) {
    LOGE("p_preprocess_cfg or p_yolov6_param can not be nullptr.\n");
    return CVI_FAILURE;
  }

  yolov6_model->set_param(p_preprocess_cfg, p_yolo_param);
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_Set_YOLOX_Param(const cviai_handle_t handle, YoloPreParam *p_preprocess_cfg,
                               YoloAlgParam *p_yolo_param) {
  printf("enter CVI_AI_Set_YOLO_Param...\n");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  YoloX *yolox_model =
      dynamic_cast<YoloX *>(getInferenceInstance(CVI_AI_SUPPORTED_MODEL_YOLOX, ctx));
  if (yolox_model == nullptr) {
    LOGE("No instance found for yolox detection.\n");
    return CVI_FAILURE;
  }
  LOGI("got yolo instance\n");
  if (p_preprocess_cfg == nullptr || p_yolo_param == nullptr) {
    LOGE("p_preprocess_cfg or p_yolo_param can not be nullptr.\n");
    return CVI_FAILURE;
  }

  yolox_model->set_param(p_preprocess_cfg, p_yolo_param);
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_Set_YOLO_Param(const cviai_handle_t handle, YoloPreParam *p_preprocess_cfg,
                              YoloAlgParam *p_yolo_param) {
  printf("enter CVI_AI_Set_YOLO_Param...\n");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  Yolo *yolo_model = dynamic_cast<Yolo *>(getInferenceInstance(CVI_AI_SUPPORTED_MODEL_YOLO, ctx));
  if (yolo_model == nullptr) {
    LOGE("No instance found for yolo detection.\n");
    return CVI_FAILURE;
  }
  LOGI("got yolo instance\n");
  if (p_preprocess_cfg == nullptr || p_yolo_param == nullptr) {
    LOGE("p_preprocess_cfg or p_yolo_param can not be nullptr.\n");
    return CVI_FAILURE;
  }

  yolo_model->set_param(p_preprocess_cfg, p_yolo_param);
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_Set_PPYOLOE_Param(const cviai_handle_t handle, YoloPreParam *p_preprocess_cfg,
                                 YoloAlgParam *p_yolo_param) {
  printf("enter CVI_AI_Set_YOLO_Param...\n");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  PPYoloE *ppyoloe_model =
      dynamic_cast<PPYoloE *>(getInferenceInstance(CVI_AI_SUPPORTED_MODEL_PPYOLOE, ctx));
  if (ppyoloe_model == nullptr) {
    LOGE("No instance found for ppyoloe detection.\n");
    return CVI_FAILURE;
  }
  LOGI("got ppyoloe instance\n");
  if (p_preprocess_cfg == nullptr || p_yolo_param == nullptr) {
    LOGE("p_preprocess_cfg or p_yolo_param can not be nullptr.\n");
    return CVI_FAILURE;
  }

  ppyoloe_model->set_param(p_preprocess_cfg, p_yolo_param);
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_Set_Image_Cls_Param(const cviai_handle_t handle, VpssPreParam *p_preprocess_cfg) {
  printf("enter CVI_AI_Set_Image_Classification_Param...\n");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  ImageClassification *image_cls_model = dynamic_cast<ImageClassification *>(
      getInferenceInstance(CVI_AI_SUPPORTED_MODEL_IMAGE_CLASSIFICATION, ctx));
  if (image_cls_model == nullptr) {
    LOGE("No instance found for image classification.\n");
    return CVI_FAILURE;
  }
  LOGI("got image_cls_model instance\n");
  if (p_preprocess_cfg == nullptr) {
    LOGE("p_preprocess_cfg can not be nullptr.\n");
    return CVI_FAILURE;
  }

  image_cls_model->set_param(p_preprocess_cfg);
  return CVI_SUCCESS;
}
