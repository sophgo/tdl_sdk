#ifndef _CVIAI_CORE_H_
#define _CVIAI_CORE_H_
#include "core/core/cvai_core_types.h"
#include "core/core/cvai_vpss_types.h"
#include "core/cviai_rescale_bbox.h"
#include "core/cviai_types_mem.h"
#include "core/deepsort/cvai_deepsort_types.h"
#include "core/face/cvai_face_helper.h"
#include "core/face/cvai_face_types.h"
#include "core/object/cvai_object_types.h"

#include <cvi_comm_vb.h>
#include <cvi_comm_vpss.h>
#include <cvi_sys.h>

/** @def CVI_AI_Free
 *  @ingroup core_cviaicore
 * @brief Free the content inside the structure, not the structure itself.
 *        Support the following structure types written in _Generic.
 *
 * @param X Input data structure.
 */
#ifdef __cplusplus
#define CVI_AI_Free(X) CVI_AI_FreeCpp(X)
#else
// clang-format off
#define CVI_AI_Free(X) _Generic((X),                   \
           cvai_feature_t*: CVI_AI_FreeFeature,        \
           cvai_pts_t*: CVI_AI_FreePts,                \
           cvai_tracker_t*: CVI_AI_FreeTracker,        \
           cvai_face_info_t*: CVI_AI_FreeFaceInfo,     \
           cvai_face_t*: CVI_AI_FreeFace,              \
           cvai_object_info_t*: CVI_AI_FreeObjectInfo, \
           cvai_object_t*: CVI_AI_FreeObject)(X)
// clang-format on
#endif

/** @def CVI_AI_CopyInfo
 *  @ingroup core_cviaicore
 * @brief Fully copy the info structure. (including allocating new memory for you.)
 *
 * @param IN Input info structure.
 * @param OUT Output info structure (uninitialized structure required).
 */
#ifdef __cplusplus
#define CVI_AI_CopyInfo(IN, OUT) CVI_AI_CopyInfoCpp(IN, OUT)
#else
// clang-format off
#define CVI_AI_CopyInfoG(OUT) _Generic((OUT),                       \
           cvai_face_info_t*: CVI_AI_CopyFaceInfo,                  \
           cvai_object_info_t*: CVI_AI_CopyObjectInfo)
#define CVI_AI_CopyInfo(IN, OUT) _Generic((IN),                     \
           cvai_face_info_t*: CVI_AI_CopyInfoG(OUT),                \
           cvai_object_info_t*: CVI_AI_CopyInfoG(OUT))((IN), (OUT))
// clang-format on
#endif

/** @def CVI_AI_RescaleMetaCenter
 * @ingroup core_cviaicore
 * @brief Rescale the output coordinate to original image. Padding in four directions. Support the
 * following structure types written in _Generic.
 *
 * @param videoFrame Original input image.
 * @param X Input data structure.
 */

/** @def CVI_AI_RescaleMetaRB
 * @ingroup core_cviaicore
 * @brief Rescale the output coordinate to original image. Padding in right, bottom directions.
 * Support the following structure types written in _Generic.
 *
 * @param videoFrame Original input image.
 * @param X Input data structure.
 */
#ifdef __cplusplus
#define CVI_AI_RescaleMetaCenter(videoFrame, X) CVI_AI_RescaleMetaCenterCpp(videoFrame, X);
#define CVI_AI_RescaleMetaRB(videoFrame, X) CVI_AI_RescaleMetaRBCpp(videoFrame, X);
#else
// clang-format off
#define CVI_AI_RescaleMetaCenter(videoFrame, X) _Generic((X), \
           cvai_face_t*: CVI_AI_RescaleMetaCenterFace,        \
           cvai_object_t*: CVI_AI_RescaleMetaCenterObj)(videoFrame, X);
#define CVI_AI_RescaleMetaRB(videoFrame, X) _Generic((X),     \
           cvai_face_t*: CVI_AI_RescaleMetaRBFace,            \
           cvai_object_t*: CVI_AI_RescaleMetaRBObj)(videoFrame, X);
// clang-format on
#endif

/** @typedef cviai_handle_t
 * @ingroup core_cviaicore
 * @brief An cviai handle
 */
typedef void *cviai_handle_t;

/**
 * \addtogroup core_ai AI Inference Functions
 * \ingroup core_cviaicore
 */

/**
 * \addtogroup core_ai_settings AI Inference Setting Functions
 * \ingroup core_ai
 */
/**@{*/

/**
 * IMPORTENT!! Add supported model here!
 */
// clang-format off
#define CVI_AI_MODEL_LIST \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_RETINAFACE)                       \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_THERMALFACE)                      \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE)                    \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_FACERECOGNITION)                  \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_MASKFACERECOGNITION)              \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_FACEQUALITY)                      \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_LIVENESS)                         \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_MASKCLASSIFICATION)               \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_LITE)                 \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE_D0)           \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D0)                   \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D1)                   \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D2)                   \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_YOLOV3)                           \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_OSNET)                            \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_ESCLASSIFICATION)                 \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_WPODNET)                          \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_LPRNET)                           \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_DEEPLABV3)                        \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_ALPHAPOSE)                        \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_EYECLASSIFICATION)                \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_YAWNCLASSIFICATION)               \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_FACELANDMARKER)
// clang-format on

#define CVI_AI_NAME_WRAP(x) x,

/** @enum CVI_AI_SUPPORTED_MODEL_E
 * @brief Supported NN model list. Can be used to config function behavior.
 *
 */
typedef enum { CVI_AI_MODEL_LIST CVI_AI_SUPPORTED_MODEL_END } CVI_AI_SUPPORTED_MODEL_E;
#undef CVI_AI_NAME_WRAP

#define CVI_AI_NAME_WRAP(x) #x,

static inline const char *CVI_AI_GetModelName(CVI_AI_SUPPORTED_MODEL_E index) {
  static const char *model_names[] = {CVI_AI_MODEL_LIST};
  int length = sizeof(model_names) / sizeof(model_names[0]);
  if (index < length) {
    return model_names[index];
  } else {
    return "Unknown";
  }
}

#undef CVI_AI_NAME_WRAP
#undef CVI_AI_MODEL_LIST

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief Create a cviai_handle_t, will automatically find a vpss group id.
 *
 * @param handle An AI SDK handle.
 * @return int Return CVI_SUCCESS if succeed.
 */
DLL_EXPORT CVI_S32 CVI_AI_CreateHandle(cviai_handle_t *handle);

/**
 * @brief Create a cviai_handle_t, need to manually assign a vpss group id.
 *
 * @param handle An AI SDK handle.
 * @param vpssGroupId Assign a group id to cviai_handle_t.
 * @return int Return CVI_SUCCESS if succeed.
 */
DLL_EXPORT CVI_S32 CVI_AI_CreateHandle2(cviai_handle_t *handle, const VPSS_GRP vpssGroupId);

/**
 * @brief Destroy a cviai_handle_t.
 *
 * @param handle An AI SDK handle.
 * @return int Return CVI_SUCCESS if success to destroy handle.
 */
DLL_EXPORT CVI_S32 CVI_AI_DestroyHandle(cviai_handle_t handle);

/**
 * @brief Set the model path for supported networks.
 *
 * @param handle An AI SDK handle.
 * @param config Supported model type config.
 * @param filepath File path to the cvimodel file.
 * @return int Return CVI_SUCCESS if load model succeed.
 */
DLL_EXPORT CVI_S32 CVI_AI_SetModelPath(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                       const char *filepath);

/**
 * @brief Get set model path from supported models.
 *
 * @param handle An AI SDK handle.
 * @param config Supported model type config.
 * @param filepath Output model path.
 * @return int Return CVI_SUCCESS.
 */
DLL_EXPORT CVI_S32 CVI_AI_GetModelPath(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                       char **filepath);

/**
 * @brief Set skip vpss preprocess for supported networks.
 *
 * @param handle An AI SDK handle.
 * @param config Supported model type config.
 * @param skip To skip preprocess or not.
 * @return int Return CVI_SUCCESS if load model succeed.
 */
DLL_EXPORT CVI_S32 CVI_AI_SetSkipVpssPreprocess(cviai_handle_t handle,
                                                CVI_AI_SUPPORTED_MODEL_E config, bool skip);

/**
 * @brief Get skip preprocess value for given supported model.
 *
 * @param handle An AI SDK handle.
 * @param config Supported model type config.
 * @param skip Output setting value.
 * @return int Return CVI_SUCCESS.
 */
DLL_EXPORT CVI_S32 CVI_AI_GetSkipVpssPreprocess(cviai_handle_t handle,
                                                CVI_AI_SUPPORTED_MODEL_E config, bool *skip);

/**
 * @brief Set the threshold of an AI inference.
 *
 * @param handle An AI SDK handle.
 * @param config Supported model type config.
 * @param threshold Threshold in float, usually a number between 0 and 1.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_SetModelThreshold(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                            float threshold);

/**
 * @brief Get the threshold of an AI Inference
 *
 * @param handle An AI SDK handle.
 * @param config Supported model type config.
 * @param threshold Threshold in float.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_GetModelThreshold(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                            float *threshold);
/**
 * @brief Set different vpss thread for each model. Vpss group id is not thread safe. We recommended
 * to change a thread if the process is not sequential.
 *
 * @param handle An AI SDK handle.
 * @param config Supported model type config.
 * @param thread The vpss thread index user desired. Note this param will changed if previous index
 * is not used.
 * @return int Return CVI_SUCCESS if successfully changed.
 */
DLL_EXPORT CVI_S32 CVI_AI_SetVpssThread(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                        const uint32_t thread);

/**
 * @brief Set different vpss thread for each model. Vpss group id is not thread safe. We recommended
 * to change a thread if the process is not sequential. This function requires manually assigning a
 * vpss group id.
 *
 * @param handle An AI SDK handle.
 * @param config Supported model type config.
 * @param thread The vpss thread index user desired. Note this param will changed if previous index
 * is not used.
 * @param vpssGroupId Assign a vpss group id if a new vpss instance needs to be created.
 * @return int Return CVI_SUCCESS if successfully changed.
 */
DLL_EXPORT CVI_S32 CVI_AI_SetVpssThread2(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                         const uint32_t thread, const VPSS_GRP vpssGroupId);

/**
 * @brief Get the set thread index for given supported model.
 *
 * @param handle An AI SDK handle.
 * @param config Supported model type config.
 * @param thread Output thread index.
 * @return int Return CVI_SUCCESS.
 */
DLL_EXPORT CVI_S32 CVI_AI_GetVpssThread(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                        uint32_t *thread);

/**
 * @brief Get the vpss group ids used by the handle.
 *
 * @param handle An AI SDK handle.
 * @param groups Return the list of used vpss group id.
 * @param num Return the length of the list.
 * @return int Return CVI_SUCCESS.
 */
DLL_EXPORT CVI_S32 CVI_AI_GetVpssGrpIds(cviai_handle_t handle, VPSS_GRP **groups, uint32_t *num);

/**
 * @brief Set VPSS waiting time.
 *
 * @param handle An AI SDK handle.
 * @param timeout Timeout value.
 * @return int Return CVI_SUCCESS.
 */
DLL_EXPORT CVI_S32 CVI_AI_SetVpssTimeout(cviai_handle_t handle, uint32_t timeout);

/**
 * @brief Close all opened models and delete the model instances.
 *
 * @param handle An AI SDK handle.
 * @return int Return CVI_SUCCESS if succeed.
 */
DLL_EXPORT CVI_S32 CVI_AI_CloseAllModel(cviai_handle_t handle);

/**
 * @brief Close the chosen model and delete its model instance.
 *
 * @param handle An AI SDK handle.
 * @param config Supported model type config.
 * @return int Return CVI_SUCCESS if close succeed.
 */
DLL_EXPORT CVI_S32 CVI_AI_CloseModel(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config);

/**
 * @brief Export vpss channel attribute.
 *
 * @param handle An AI SDK handle.
 * @param config Supported model type config.
 * @param frameWidth The input frame width.
 * @param frameHeight The input frame height.
 * @param idx The index of the input tensors.
 * @param chnAttr Exported VPSS channel config settings.
 * @return int Return CVI_SUCCESS on success, CVI_FAILURE if exporting not supported.
 */
DLL_EXPORT CVI_S32 CVI_AI_GetVpssChnConfig(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                           const CVI_U32 frameWidth, const CVI_U32 frameHeight,
                                           const CVI_U32 idx, cvai_vpssconfig_t *chnConfig);

/**@}*/

/**
 * \addtogroup core_fd Face Detection AI Inference
 * \ingroup core_ai
 */
/**@{*/

/**
 * @brief RetinaFace face detection.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param faces Output detect result. The name, bbox, and face points will be given.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_RetinaFace(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                     cvai_face_t *faces);

/**
 * @brief Detect face with thermal images.
 *
 * @param handle An AI SDK handle.
 * @param frame Input thermal video frame.
 * @param faces Output detect result. The bbox will be given.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_ThermalFace(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                      cvai_face_t *faces);

/**@}*/

/**
 * \addtogroup core_fr Face Recognition AI Inference
 * \ingroup core_ai
 */
/**@{*/

/**
 * @brief Do face recognition and attribute with bbox info stored in faces.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param faces cvai_face_t structure, the cvai_face_info_t and cvai_bbox_t must be set.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_FaceAttribute(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                        cvai_face_t *faces);

/**
 * @brief Do face recognition and attribute with bbox info stored in faces. Only do inference on the
 * given index of cvai_face_info_t.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param faces cvai_face_t structure, the cvai_face_info_t and cvai_bbox_t must be set.
 * @param face_idx The index of cvai_face_info_t inside cvai_face_t.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_FaceAttributeOne(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                           cvai_face_t *faces, int face_idx);

/**
 * @brief Do face recognition with bbox info stored in faces.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param faces cvai_face_t structure, the cvai_face_info_t and cvai_bbox_t must be set.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_FaceRecognition(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                          cvai_face_t *faces);

/**
 * @brief Do face recognition with bbox info stored in faces. Only do inference on the given index
 * of cvai_face_info_t.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param faces cvai_face_t structure, the cvai_face_info_t and cvai_bbox_t must be set.
 * @param face_idx The index of cvai_face_info_t inside cvai_face_t.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_FaceRecognitionOne(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                             cvai_face_t *faces, int face_idx);

/**
 * @brief Do face recognition with mask wearing.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param faces cvai_face_t structure, the cvai_face_info_t and cvai_bbox_t must be set.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_MaskFaceRecognition(const cviai_handle_t handle,
                                              VIDEO_FRAME_INFO_S *frame, cvai_face_t *faces);

/**@}*/

/**
 * \addtogroup core_fc Face classification AI Inference
 * \ingroup core_ai
 */
/**@{*/

/**
 * @brief FaceQuality. Gives a score to present how good the image quality of a face is.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param face cvai_face_t structure, the cvai_face_info_t and cvai_bbox_t must be set.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_FaceQuality(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                      cvai_face_t *face);

/**
 * @brief Liveness. Gives a score to present how real the face is. The score will be low if the face
 * is not directly taken by a camera.
 *
 * @param handle An AI SDK handle.
 * @param rgbFrame Input RGB video frame.
 * @param irFrame Input IR video frame.
 * @param face cvai_face_t structure, the cvai_face_info_t and cvai_bbox_t must be set.
 * @param ir_position The position relationship netween the ir and the rgb camera.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_Liveness(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *rgbFrame,
                                   VIDEO_FRAME_INFO_S *irFrame, cvai_face_t *rgb_face,
                                   cvai_face_t *ir_face);

/**
 * @brief Mask classification. Tells if a face is wearing a mask.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param face cvai_face_t structure, the cvai_face_info_t and cvai_bbox_t must be set.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_MaskClassification(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                             cvai_face_t *face);

/**@}*/

DLL_EXPORT CVI_S32 CVI_AI_SelectDetectClass(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                            uint32_t num_classes, ...);

/**
 * \addtogroup core_od Object Detection AI Inference
 * \ingroup core_ai
 */
/**@{*/

/**
 * @brief MobileDetV2 Vehicle object detection, This model cat detect "car", "truck", "motorbike"
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param obj Output detect result. The name, bbox, and classes will be given.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_MobileDetV2_Vehicle_D0(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                                 cvai_object_t *obj);

/**
 * @brief MobileDetV2 Lite object detection, the most lightweight MobileDetV2 with 9 classes
 * detection. This model cat detect "person", "bicycle", "car", "motorbike", "aeroplane", "bus",
 * "train", "truck", and "boat"
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param obj Output detect result. The name, bbox, and classes will be given.
 * @param det_type Specify detection type.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_MobileDetV2_Lite(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                           cvai_object_t *obj, cvai_obj_det_type_e det_type);

/**
 * @brief MobileDetV2 D0 object detection, the most lightweight MobileDetV2.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param obj Output detect result. The name, bbox, and classes will be given.
 * @param det_type Specify detection type.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_MobileDetV2_D0(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                         cvai_object_t *obj, cvai_obj_det_type_e det_type);

/**
 * @brief MobileDetV2 D1 object detection, the not so lightweight MobileDetV2.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param obj Output detect result. The name, bbox, and classes will be given.
 * @param det_type Specify detection type.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_MobileDetV2_D1(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                         cvai_object_t *obj, cvai_obj_det_type_e det_type);

/**
 * @brief MobileDetV2 D2 object detection, the heaviest MobileDetV2.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param obj Output detect result. The name, bbox, and classes will be given.
 * @param det_type Specify detection type.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_MobileDetV2_D2(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                         cvai_object_t *obj, cvai_obj_det_type_e det_type);

/**
 * @brief Yolov3 object detection.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param obj Output detect result. The name, bbox, and classes will be given.
 * @param det_type Specify detection type.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_Yolov3(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                 cvai_object_t *obj, cvai_obj_det_type_e det_type);

/**@}*/

/**
 * \addtogroup core_pr Person Re-Id AI Inference
 * \ingroup core_ai
 */
/**@{*/

/**
 * @brief Do person Re-Id with bbox info stored in obj.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param obj cvai_object_t structure, the cvai_object_info_t and cvai_bbox_t must be set.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_OSNet(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                cvai_object_t *obj);

/**
 * @brief Do person Re-Id with bbox info stored in obj. Only do inference on the given index of
 * cvai_object_info_t.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param obj cvai_object_t structure, the cvai_object_info_t and cvai_bbox_t must be set.
 * @param obj_idx The index of cvai_object_info_t inside cvai_object_t.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_OSNetOne(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                   cvai_object_t *obj, int obj_idx);

/**@}*/

/**
 * \addtogroup core_audio Audio AI Inference
 * \ingroup core_ai
 */
/**@{*/

/**
 * @brief Do Environment sound detection.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param index The index of environment sound classes.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_ESClassification(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                           int *index);

/**@}*/

/**
 * \addtogroup core_tracker Tracker
 * \ingroup core_ai
 */
/**@{*/

/**
 * @brief Initialize deepsort.
 *
 * @param handle An AI SDK handle.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_Deepsort_Init(const cviai_handle_t handle);

/**
 * @brief Get default deepsort config.
 *
 * @param handle An AI SDK handle.
 * @param ds_conf A deepsort config.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_Deepsort_GetDefaultConfig(cvai_deepsort_config_t *ds_conf);

/**
 * @brief Set deepsort with specific config.
 *
 * @param handle An AI SDK handle.
 * @param ds_conf The specific config.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_Deepsort_SetConfig(const cviai_handle_t handle,
                                             cvai_deepsort_config_t *ds_conf);

/**
 * @brief Run deepsort track.
 *
 * @param handle An AI SDK handle.
 * @param obj Input detected object with feature.
 * @param tracker_t Output tracker results.
 * @param use_reid If true, track by DeepSORT algorithm, else SORT.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_Deepsort(const cviai_handle_t handle, cvai_object_t *obj,
                                   cvai_tracker_t *tracker_t, bool use_reid);

DLL_EXPORT CVI_S32 CVI_AI_Deepsort_DebugInfo_1(const cviai_handle_t handle, char *debug_info);

/**@}*/

/**
 * \addtogroup core_segmentation Segmentation Inference
 * \ingroup core_ai
 */
/**@{*/

/**
 * @brief Deeplabv3 segmentation.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param out_frame Output frame which each pixel represents class label.
 * @param filter Class id filter. Set NULL to ignore.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_DeeplabV3(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                    VIDEO_FRAME_INFO_S *out_frame, cvai_class_filter_t *filter);
/**@}*/

/**
 * @brief LicensePlateRecognition.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param license_plate_meta License plate object info
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_LicensePlateRecognition(const cviai_handle_t handle,
                                                  VIDEO_FRAME_INFO_S *frame,
                                                  cvai_object_t *license_plate_meta);
/**@}*/

/**
 * @brief LicensePlateDetection.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param vehicle_meta Vehicle object info
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_LicensePlateDetection(const cviai_handle_t handle,
                                                VIDEO_FRAME_INFO_S *frame,
                                                cvai_object_t *vehicle_meta);
/**@}*/

/**
 * \addtogroup core_pose Pose Detection
 * \ingroup core_ai
 */
/**@{*/
/**
 * @brief Alphapose.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param vehicle_meta Detected object info
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_AlphaPose(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                    cvai_object_t *objects);
/**@}*/

/**
 * \addtogroup core_fall Fall Detection
 * \ingroup core_ai
 */
/**@{*/
/**
 * @brief Fall.
 *
 * @param handle An AI SDK handle.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_Fall(const cviai_handle_t handle, cvai_object_t *objects);
/**@{*/

/**
 * \addtogroup core_others Others
 * \ingroup core_ai
 */
/**@{*/

/**
 * @brief Do background subtraction.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param moving_score Check the unit diff sum of a frame.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_TamperDetection(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                          float *moving_score);

/**
 * @brief Do eye classification.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param cvai_face_t structure. Calculate the eye_score in cvai_dms_t.
 * @return int Return CVI_SUCCESS on success.
 */

DLL_EXPORT CVI_S32 CVI_AI_EyeClassification(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                            cvai_face_t *face);

/**
 * @brief Do yawn classification.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param cvai_face_t structure. Calculate the yawn_score in cvai_dms_t.
 * @return int Return CVI_SUCCESS on success.
 */

DLL_EXPORT CVI_S32 CVI_AI_YawnClassification(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                             cvai_face_t *face);
/**
 * @brief Do face landmark.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param cvai_face_t structure. Calculate the landmarks in cvai_dms_t.
 * @return int Return CVI_SUCCESS on success.
 */

DLL_EXPORT CVI_S32 CVI_AI_FaceLandmarker(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                         cvai_face_t *face);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif  // End of _CVIAI_CORE_H_
