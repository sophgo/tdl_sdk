#include "core/cviai_core.h"
#include "cviai_core_internal.hpp"

#include "cviai_experimental.h"
#include "cviai_perfetto.h"
#include "face_attribute/face_attribute.hpp"
#include "face_quality/face_quality.hpp"
#include "liveness/liveness.hpp"
#include "mask_classification/mask_classification.hpp"
#include "mask_face_recognition/mask_face_recognition.hpp"
#include "object_detection/mobiledetv2/mobiledetv2.hpp"
#include "object_detection/yolov3/yolov3.hpp"
#include "retina_face/retina_face.hpp"
#include "thermal_face_detection/thermal_face.hpp"

#include "cviai_trace.hpp"
#include "opencv2/opencv.hpp"

#include <stdio.h>
#include <unistd.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;
using namespace cviai;

void CVI_AI_PerfettoInit() {
#if __GNUC__ >= 7
  perfetto::TracingInitArgs args;
  args.backends |= perfetto::kInProcessBackend;
  args.backends |= perfetto::kSystemBackend;

  perfetto::Tracing::Initialize(args);
  perfetto::TrackEvent::Register();
#endif
}

//*************************************************
// Experimental features
void CVI_AI_EnableGDC(cviai_handle_t handle, bool use_gdc) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  ctx->use_gdc_wrap = use_gdc;
}
//*************************************************

int CVI_AI_CreateHandle(cviai_handle_t *handle) {
  cviai_context_t *ctx = new cviai_context_t;
  ctx->ive_handle = CVI_IVE_CreateHandle();
  ctx->vec_vpss_engine.push_back(new VpssEngine());
  if (ctx->vec_vpss_engine[0]->init(false) != CVI_SUCCESS) {
    delete ctx->vec_vpss_engine[0];
    delete ctx;
    return CVI_FAILURE;
  }
  *handle = ctx;
  return CVI_SUCCESS;
}

int CVI_AI_CreateHandle2(cviai_handle_t *handle, const VPSS_GRP vpssGroupId) {
  if (vpssGroupId == (VPSS_GRP)-1) {
    return CVI_FAILURE;
  }
  cviai_context_t *ctx = new cviai_context_t;
  ctx->vec_vpss_engine.push_back(new VpssEngine());
  if (ctx->vec_vpss_engine[0]->init(false, vpssGroupId) != CVI_SUCCESS) {
    delete ctx->vec_vpss_engine[0];
    delete ctx;
    return CVI_FAILURE;
  }
  *handle = ctx;
  return CVI_SUCCESS;
}

int CVI_AI_DestroyHandle(cviai_handle_t handle) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  CVI_AI_CloseAllModel(handle);
  CVI_IVE_DestroyHandle(ctx->ive_handle);
  for (auto it : ctx->vec_vpss_engine) {
    delete it;
  }
  delete ctx->td_model;
  delete ctx;
  return CVI_SUCCESS;
}

int CVI_AI_SetModelPath(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                        const char *filepath) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  ctx->model_cont[config].model_path = filepath;
  return CVI_SUCCESS;
}

int CVI_AI_GetModelPath(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config, char **filepath) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  char *path = (char *)malloc(ctx->model_cont[config].model_path.size());
  snprintf(path, strlen(path), "%s", ctx->model_cont[config].model_path.c_str());
  *filepath = path;
  return CVI_SUCCESS;
}

int CVI_AI_SetSkipVpssPreprocess(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                 bool skip) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  auto &m_t = ctx->model_cont[config];
  m_t.skip_vpss_preprocess = skip;
  if (m_t.instance != nullptr) {
    m_t.instance->skipVpssPreprocess(m_t.skip_vpss_preprocess);
  }
  return CVI_SUCCESS;
}

int CVI_AI_GetSkipVpssPreprocess(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                 bool *skip) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  *skip = ctx->model_cont[config].skip_vpss_preprocess;
  return CVI_SUCCESS;
}

int CVI_AI_SetModelThreshold(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                             float threshold) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  auto &m_t = ctx->model_cont[config];
  m_t.model_threshold = threshold;
  if (m_t.instance != nullptr) {
    m_t.instance->setModelThreshold(m_t.model_threshold);
  }
  return CVI_SUCCESS;
}

int CVI_AI_GetModelThreshold(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                             float *threshold) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  *threshold = ctx->model_cont[config].model_threshold;
  return CVI_SUCCESS;
}

int CVI_AI_SetVpssThread(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                         const uint32_t thread) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  uint32_t vpss_thread = thread;
  if (thread >= ctx->vec_vpss_engine.size()) {
    auto inst = new VpssEngine();
    if (inst->init(false) != CVI_SUCCESS) {
      printf("Vpss init failed\n");
      delete inst;
      return CVI_FAILURE;
    }

    ctx->vec_vpss_engine.push_back(inst);
    if (thread != ctx->vec_vpss_engine.size() - 1) {
      printf(
          "Thread %u is not in use, thus %u is changed to %u automatically. Used vpss group id is "
          "%u.\n",
          vpss_thread, thread, vpss_thread, inst->getGrpId());
      vpss_thread = ctx->vec_vpss_engine.size() - 1;
    }
  }
  ctx->model_cont[config].vpss_thread = vpss_thread;
  return CVI_SUCCESS;
}

int CVI_AI_SetVpssThread2(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                          const uint32_t thread, const VPSS_GRP vpssGroupId) {
  if (vpssGroupId == (VPSS_GRP)-1) {
    return CVI_FAILURE;
  }
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  uint32_t vpss_thread = thread;
  if (thread >= ctx->vec_vpss_engine.size()) {
    auto inst = new VpssEngine();
    if (inst->init(false, vpssGroupId) != CVI_SUCCESS) {
      printf("Vpss init failed\n");
      delete inst;
      return CVI_FAILURE;
    }

    ctx->vec_vpss_engine.push_back(inst);
    if (thread != ctx->vec_vpss_engine.size() - 1) {
      printf(
          "Thread %u is not in use, thus %u is changed to %u automatically. Used vpss group id is "
          "%u.\n",
          vpss_thread, thread, vpss_thread, inst->getGrpId());
      vpss_thread = ctx->vec_vpss_engine.size() - 1;
    }
  } else {
    printf("Thread %u already exists, given group id %u will not be used.\n", thread, vpssGroupId);
  }
  auto &m_t = ctx->model_cont[config];
  m_t.vpss_thread = vpss_thread;
  if (m_t.instance != nullptr) {
    m_t.instance->setVpssEngine(ctx->vec_vpss_engine[m_t.vpss_thread]);
  }
  return CVI_SUCCESS;
}

int CVI_AI_GetVpssThread(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config, uint32_t *thread) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  *thread = ctx->model_cont[config].vpss_thread;
  return CVI_SUCCESS;
}

int CVI_AI_GetVpssGrpIds(cviai_handle_t handle, VPSS_GRP **groups, uint32_t *num) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  VPSS_GRP *ids = (VPSS_GRP *)malloc(ctx->vec_vpss_engine.size() * sizeof(VPSS_GRP));
  for (size_t i = 0; i < ctx->vec_vpss_engine.size(); i++) {
    ids[i] = ctx->vec_vpss_engine[i]->getGrpId();
  }
  *groups = ids;
  *num = ctx->vec_vpss_engine.size();
  return CVI_SUCCESS;
}

int CVI_AI_CloseAllModel(cviai_handle_t handle) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  for (auto &m_inst : ctx->model_cont) {
    if (m_inst.second.instance != nullptr) {
      m_inst.second.instance->modelClose();
      delete m_inst.second.instance;
      m_inst.second.instance = nullptr;
    }
  }
  ctx->model_cont.clear();
  return CVI_SUCCESS;
}

int CVI_AI_CloseModel(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  cviai_model_t &m_t = ctx->model_cont[config];
  if (m_t.instance == nullptr) {
    return CVI_FAILURE;
  }
  m_t.instance->modelClose();
  delete m_t.instance;
  m_t.instance = nullptr;
  return CVI_SUCCESS;
}

inline void BufferRGBPackedCopy(const uint8_t *buffer, uint32_t width, uint32_t height,
                                uint32_t stride, VIDEO_FRAME_INFO_S *frame, bool invert) {
  VIDEO_FRAME_S *vFrame = &frame->stVFrame;
  if (invert) {
    for (uint32_t j = 0; j < height; j++) {
      const uint8_t *ptr = buffer + j * stride;
      for (uint32_t i = 0; i < width; i++) {
        uint32_t offset = i * 3 + j * vFrame->u32Stride[0];
        const uint8_t *ptr_pxl = i * 3 + ptr;
        vFrame->pu8VirAddr[0][offset] = ptr_pxl[2];
        vFrame->pu8VirAddr[0][offset + 1] = ptr_pxl[1];
        vFrame->pu8VirAddr[0][offset + 2] = ptr_pxl[0];
      }
    }
  } else {
    for (uint32_t j = 0; j < height; j++) {
      const uint8_t *ptr = buffer + j * stride;
      for (uint32_t i = 0; i < width; i++) {
        uint32_t offset = i * 3 + j * vFrame->u32Stride[0];
        const uint8_t *ptr_pxl = i * 3 + ptr;
        vFrame->pu8VirAddr[0][offset] = ptr_pxl[0];
        vFrame->pu8VirAddr[0][offset + 1] = ptr_pxl[1];
        vFrame->pu8VirAddr[0][offset + 2] = ptr_pxl[2];
      }
    }
  }
}

inline void BufferRGBPacked2PlanarCopy(const uint8_t *buffer, uint32_t width, uint32_t height,
                                       uint32_t stride, VIDEO_FRAME_INFO_S *frame, bool invert) {
  VIDEO_FRAME_S *vFrame = &frame->stVFrame;
  if (invert) {
    for (uint32_t j = 0; j < height; j++) {
      const uint8_t *ptr = buffer + j * stride;
      for (uint32_t i = 0; i < width; i++) {
        const uint8_t *ptr_pxl = i * 3 + ptr;
        vFrame->pu8VirAddr[0][i + j * vFrame->u32Stride[0]] = ptr_pxl[0];
        vFrame->pu8VirAddr[1][i + j * vFrame->u32Stride[1]] = ptr_pxl[1];
        vFrame->pu8VirAddr[2][i + j * vFrame->u32Stride[2]] = ptr_pxl[2];
      }
    }
  } else {
    for (uint32_t j = 0; j < height; j++) {
      const uint8_t *ptr = buffer + j * stride;
      for (uint32_t i = 0; i < width; i++) {
        const uint8_t *ptr_pxl = i * 3 + ptr;
        vFrame->pu8VirAddr[0][i + j * vFrame->u32Stride[0]] = ptr_pxl[2];
        vFrame->pu8VirAddr[1][i + j * vFrame->u32Stride[1]] = ptr_pxl[1];
        vFrame->pu8VirAddr[2][i + j * vFrame->u32Stride[2]] = ptr_pxl[0];
      }
    }
  }
}

int CVI_AI_Buffer2VBFrame(const uint8_t *buffer, uint32_t width, uint32_t height, uint32_t stride,
                          const PIXEL_FORMAT_E inFormat, VB_BLK *blk, VIDEO_FRAME_INFO_S *frame,
                          const PIXEL_FORMAT_E outFormat) {
  if (CREATE_VBFRAME_HELPER(blk, frame, width, height, outFormat) != CVI_SUCCESS) {
    printf("Create VBFrame failed.\n");
    return CVI_FAILURE;
  }

  int ret = CVI_SUCCESS;
  if ((inFormat == PIXEL_FORMAT_RGB_888 && outFormat == PIXEL_FORMAT_BGR_888) ||
      (inFormat == PIXEL_FORMAT_BGR_888 && outFormat == PIXEL_FORMAT_RGB_888)) {
    BufferRGBPackedCopy(buffer, width, height, stride, frame, true);
  } else if ((inFormat == PIXEL_FORMAT_RGB_888 && outFormat == PIXEL_FORMAT_RGB_888) ||
             (inFormat == PIXEL_FORMAT_BGR_888 && outFormat == PIXEL_FORMAT_BGR_888)) {
    BufferRGBPackedCopy(buffer, width, height, stride, frame, false);
  } else if (inFormat == PIXEL_FORMAT_BGR_888 && outFormat == PIXEL_FORMAT_RGB_888_PLANAR) {
    BufferRGBPacked2PlanarCopy(buffer, width, height, stride, frame, true);
  } else if (inFormat == PIXEL_FORMAT_RGB_888 && outFormat == PIXEL_FORMAT_RGB_888_PLANAR) {
    BufferRGBPacked2PlanarCopy(buffer, width, height, stride, frame, false);
  } else {
    printf("Unsupported convert format: %u -> %u.\n", inFormat, outFormat);
    ret = CVI_FAILURE;
  }
  uint32_t image_size =
      frame->stVFrame.u32Length[0] + frame->stVFrame.u32Length[1] + frame->stVFrame.u32Length[2];
  CVI_SYS_IonFlushCache(frame->stVFrame.u64PhyAddr[0], frame->stVFrame.pu8VirAddr[0], image_size);
  CVI_SYS_Munmap(
      (void *)frame->stVFrame.pu8VirAddr[0],
      frame->stVFrame.u32Length[0] + frame->stVFrame.u32Length[1] + frame->stVFrame.u32Length[2]);
  frame->stVFrame.pu8VirAddr[0] = NULL;
  frame->stVFrame.pu8VirAddr[1] = NULL;
  frame->stVFrame.pu8VirAddr[2] = NULL;

  return ret;
}

int CVI_AI_ReadImage(const char *filepath, VB_BLK *blk, VIDEO_FRAME_INFO_S *frame,
                     PIXEL_FORMAT_E format) {
  cv::Mat img = cv::imread(filepath);
  if (CREATE_VBFRAME_HELPER(blk, frame, img.cols, img.rows, format) != CVI_SUCCESS) {
    printf("Create VBFrame failed.\n");
    return CVI_FAILURE;
  }

  int ret = CVI_SUCCESS;
  VIDEO_FRAME_S *vFrame = &frame->stVFrame;
  switch (format) {
    case PIXEL_FORMAT_RGB_888: {
      for (int j = 0; j < img.rows; j++) {
        cv::Vec3b *ptr = img.ptr<cv::Vec3b>(j);
        for (int i = 0; i < img.cols; i++) {
          uint32_t offset = i * 3 + j * vFrame->u32Stride[0];
          vFrame->pu8VirAddr[0][offset] = ptr[i][2];
          vFrame->pu8VirAddr[0][offset + 1] = ptr[i][1];
          vFrame->pu8VirAddr[0][offset + 2] = ptr[i][0];
        }
      }
    } break;
    case PIXEL_FORMAT_BGR_888: {
      for (int j = 0; j < img.rows; j++) {
        cv::Vec3b *ptr = img.ptr<cv::Vec3b>(j);
        for (int i = 0; i < img.cols; i++) {
          uint32_t offset = i * 3 + j * vFrame->u32Stride[0];
          vFrame->pu8VirAddr[0][offset] = ptr[i][0];
          vFrame->pu8VirAddr[0][offset + 1] = ptr[i][1];
          vFrame->pu8VirAddr[0][offset + 2] = ptr[i][2];
        }
      }
    } break;
    case PIXEL_FORMAT_RGB_888_PLANAR: {
      for (int j = 0; j < img.rows; j++) {
        cv::Vec3b *ptr = img.ptr<cv::Vec3b>(j);
        for (int i = 0; i < img.cols; i++) {
          vFrame->pu8VirAddr[0][i + j * vFrame->u32Stride[0]] = ptr[i][2];
          vFrame->pu8VirAddr[1][i + j * vFrame->u32Stride[1]] = ptr[i][1];
          vFrame->pu8VirAddr[2][i + j * vFrame->u32Stride[2]] = ptr[i][0];
        }
      }
    } break;
    default:
      printf("Unsupported format: %u.\n", format);
      ret = CVI_FAILURE;
      break;
  }
  uint32_t image_size = vFrame->u32Length[0] + vFrame->u32Length[1] + vFrame->u32Length[2];
  CVI_SYS_IonFlushCache(vFrame->u64PhyAddr[0], vFrame->pu8VirAddr[0], image_size);
  CVI_SYS_Munmap((void *)vFrame->pu8VirAddr[0],
                 vFrame->u32Length[0] + vFrame->u32Length[1] + vFrame->u32Length[2]);
  // FIXME: Middleware bug
  vFrame->pu8VirAddr[0] = NULL;
  vFrame->pu8VirAddr[1] = NULL;
  vFrame->pu8VirAddr[2] = NULL;
  return ret;
}

template <class C, typename... Arguments>
inline C *__attribute__((always_inline))
getInferenceInstance(const CVI_AI_SUPPORTED_MODEL_E index, cviai_context_t *ctx,
                     Arguments &&... arg) {
  cviai_model_t &m_t = ctx->model_cont[index];
  if (m_t.instance == nullptr) {
    if (m_t.model_path.empty()) {
      printf("Model path for FaceAttribute is empty.\n");
      return nullptr;
    }
    m_t.instance = new C(arg...);
    if (m_t.instance->modelOpen(m_t.model_path.c_str()) != CVI_SUCCESS) {
      printf("Open model failed (%s).\n", m_t.model_path.c_str());
      return nullptr;
    }
    m_t.instance->setVpssEngine(ctx->vec_vpss_engine[m_t.vpss_thread]);
    m_t.instance->skipVpssPreprocess(m_t.skip_vpss_preprocess);
    if (m_t.model_threshold == -1) {
      m_t.model_threshold = m_t.instance->getModelThreshold();
    } else {
      m_t.instance->setModelThreshold(m_t.model_threshold);
    }
  }
  C *class_inst = dynamic_cast<C *>(m_t.instance);
  return class_inst;
}

inline int __attribute__((always_inline))
CVI_AI_FaceAttributeBase(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_face_t *faces,
                         int face_idx, bool set_attribute) {
  TRACE_EVENT("cviai_core", "CVI_AI_FaceRecognitionOne");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  FaceAttribute *face_attr = getInferenceInstance<FaceAttribute>(
      CVI_AI_SUPPORTED_MODEL_FACERECOGNITION, ctx, ctx->use_gdc_wrap);
  if (face_attr == nullptr) {
    printf("No instance found for FaceAttribute.\n");
    return CVI_FAILURE;
  }
  face_attr->setWithAttribute(set_attribute);
  return face_attr->inference(frame, faces, face_idx);
}

int CVI_AI_FaceRecognition(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                           cvai_face_t *faces) {
  TRACE_EVENT("cviai_core", "CVI_AI_FaceRecognition");
  return CVI_AI_FaceAttributeBase(handle, frame, faces, -1, false);
}

int CVI_AI_FaceRecognitionOne(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                              cvai_face_t *faces, int face_idx) {
  TRACE_EVENT("cviai_core", "CVI_AI_FaceRecognitionOne");
  return CVI_AI_FaceAttributeBase(handle, frame, faces, face_idx, false);
}

int CVI_AI_FaceAttribute(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                         cvai_face_t *faces) {
  TRACE_EVENT("cviai_core", "CVI_AI_FaceAttribute");
  return CVI_AI_FaceAttributeBase(handle, frame, faces, -1, true);
}

int CVI_AI_FaceAttributeOne(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                            cvai_face_t *faces, int face_idx) {
  TRACE_EVENT("cviai_core", "CVI_AI_FaceAttributeOne");
  return CVI_AI_FaceAttributeBase(handle, frame, faces, face_idx, true);
}

int CVI_AI_Yolov3(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj,
                  cvai_obj_det_type_t det_type) {
  TRACE_EVENT("cviai_core", "CVI_AI_Yolov3");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  Yolov3 *yolov3 = getInferenceInstance<Yolov3>(CVI_AI_SUPPORTED_MODEL_YOLOV3, ctx);
  if (yolov3 == nullptr) {
    printf("No instance found for Yolov3.\n");
    return CVI_FAILURE;
  }
  return yolov3->inference(frame, obj, det_type);
}

int CVI_AI_MobileDetV2_D0(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj,
                          cvai_obj_det_type_t det_type) {
  TRACE_EVENT("cviai_core", "CVI_AI_MobileDetV2_D0");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  MobileDetV2 *detector = getInferenceInstance<MobileDetV2>(CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D0,
                                                            ctx, MobileDetV2::Model::d0);
  if (detector == nullptr) {
    printf("No instance found for MobileDetV2.\n");
    return CVI_RC_FAILURE;
  }
  return detector->inference(frame, obj, det_type);
}

int CVI_AI_MobileDetV2_D1(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj,
                          cvai_obj_det_type_t det_type) {
  TRACE_EVENT("cviai_core", "CVI_AI_MobileDetV2_D1");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  MobileDetV2 *detector = getInferenceInstance<MobileDetV2>(CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D1,
                                                            ctx, MobileDetV2::Model::d1);
  if (detector == nullptr) {
    printf("No instance found for MobileDetV2.\n");
    return CVI_RC_FAILURE;
  }
  return detector->inference(frame, obj, det_type);
}

int CVI_AI_MobileDetV2_D2(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj,
                          cvai_obj_det_type_t det_type) {
  TRACE_EVENT("cviai_core", "CVI_AI_MobileDetV2_D2");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  MobileDetV2 *detector = getInferenceInstance<MobileDetV2>(CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D2,
                                                            ctx, MobileDetV2::Model::d2);
  if (detector == nullptr) {
    printf("No instance found for MobileDetV2.\n");
    return CVI_RC_FAILURE;
  }
  return detector->inference(frame, obj, det_type);
}

int CVI_AI_RetinaFace(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_face_t *faces,
                      int *face_count) {
  TRACE_EVENT("cviai_core", "CVI_AI_RetinaFace");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  RetinaFace *retina_face =
      getInferenceInstance<RetinaFace>(CVI_AI_SUPPORTED_MODEL_RETINAFACE, ctx);
  if (retina_face == nullptr) {
    printf("No instance found for RetinaFace.\n");
    return CVI_FAILURE;
  }
  return retina_face->inference(frame, faces, face_count);
}

int CVI_AI_Liveness(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *rgbFrame,
                    VIDEO_FRAME_INFO_S *irFrame, cvai_face_t *face,
                    cvai_liveness_ir_position_e ir_position) {
  TRACE_EVENT("cviai_core", "CVI_AI_Liveness");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  Liveness *liveness =
      getInferenceInstance<Liveness>(CVI_AI_SUPPORTED_MODEL_LIVENESS, ctx, ir_position);
  if (liveness == nullptr) {
    printf("No instance found for Liveness.\n");
    return CVI_FAILURE;
  }
  return liveness->inference(rgbFrame, irFrame, face);
}

int CVI_AI_FaceQuality(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_face_t *face) {
  TRACE_EVENT("cviai_core", "CVI_AI_FaceQuality");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  FaceQuality *face_quality =
      getInferenceInstance<FaceQuality>(CVI_AI_SUPPORTED_MODEL_FACEQUALITY, ctx);
  if (face_quality == nullptr) {
    printf("No instance found for FaceQuality.\n");
    return CVI_FAILURE;
  }
  return face_quality->inference(frame, face);
}

int CVI_AI_MaskClassification(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                              cvai_face_t *face) {
  TRACE_EVENT("cviai_core", "CVI_AI_MaskClassification");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  MaskClassification *mask_classification =
      getInferenceInstance<MaskClassification>(CVI_AI_SUPPORTED_MODEL_MASKCLASSIFICATION, ctx);
  if (mask_classification == nullptr) {
    printf("No instance found for MaskClassification.\n");
    return CVI_FAILURE;
  }
  return mask_classification->inference(frame, face);
}

int CVI_AI_ThermalFace(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_face_t *faces) {
  TRACE_EVENT("cviai_core", "CVI_AI_ThermalFace");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  ThermalFace *thermal_face =
      getInferenceInstance<ThermalFace>(CVI_AI_SUPPORTED_MODEL_THERMALFACE, ctx);
  if (thermal_face == nullptr) {
    printf("No instance found for ThermalFace.\n");
    return CVI_FAILURE;
  }
  return thermal_face->inference(frame, faces);
}

int CVI_AI_MaskFaceRecognition(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                               cvai_face_t *faces) {
  TRACE_EVENT("cviai_core", "CVI_AI_MaskFaceRecognition");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  MaskFaceRecognition *mask_face_rec =
      getInferenceInstance<MaskFaceRecognition>(CVI_AI_SUPPORTED_MODEL_MASKFACERECOGNITION, ctx);
  if (mask_face_rec == nullptr) {
    printf("No instance found for MaskFaceRecognition.\n");
    return CVI_FAILURE;
  }

  return mask_face_rec->inference(frame, faces);
}

int CVI_AI_TamperDetection(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                           float *moving_score) {
  TRACE_EVENT("cviai_core", "CVI_AI_TamperDetection");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  TamperDetectorMD *td_model = ctx->td_model;
  if (td_model == nullptr) {
    printf("Init Tamper Detection Model.\n");
    ctx->td_model = new TamperDetectorMD(ctx->ive_handle, frame, (float)0.05, (int)10);
    ctx->td_model->print_info();

    *moving_score = -1.0;
    return 0;
  }
  return ctx->td_model->detect(frame, moving_score);
}
