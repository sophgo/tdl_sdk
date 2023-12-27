#ifndef _CVI_TDL_CORE_H_
#define _CVI_TDL_CORE_H_
#include "core/core/cvtdl_core_types.h"
#include "core/core/cvtdl_errno.h"
#include "core/core/cvtdl_vpss_types.h"
#include "core/cvi_tdl_rescale_bbox.h"
#include "core/cvi_tdl_types_mem.h"
#include "core/deepsort/cvtdl_deepsort_types.h"
#include "core/face/cvtdl_face_helper.h"
#include "core/face/cvtdl_face_types.h"
#include "core/object/cvtdl_object_types.h"
#include "cvi_comm.h"

/** @def CVI_TDL_Free
 *  @ingroup core_cvitdlcore
 * @brief Free the content inside the structure, not the structure itself.
 *        Support the following structure types written in _Generic.
 *
 * @param X Input data structure.
 */
#ifdef __cplusplus
#define CVI_TDL_Free(X) CVI_TDL_FreeCpp(X)
#else
// clang-format off
#define CVI_TDL_Free(X) _Generic((X),                   \
           cvtdl_feature_t*: CVI_TDL_FreeFeature,        \
           cvtdl_pts_t*: CVI_TDL_FreePts,                \
           cvtdl_tracker_t*: CVI_TDL_FreeTracker,        \
           cvtdl_face_info_t*: CVI_TDL_FreeFaceInfo,     \
           cvtdl_face_t*: CVI_TDL_FreeFace,              \
           cvtdl_object_info_t*: CVI_TDL_FreeObjectInfo, \
           cvtdl_object_t*: CVI_TDL_FreeObject,          \
           cvtdl_handpose21_meta_ts*: CVI_TDL_FreeHandPoses, \
           cvtdl_class_meta_t*: CVI_TDL_FreeClassMeta, \
           cvtdl_image_t*: CVI_TDL_FreeImage)(X)
// clang-format on
#endif

/** @def CVI_TDL_CopyInfo
 *  @ingroup core_cvitdlcore
 * @brief Fully copy the info structure. (including allocating new memory for you.)
 *
 * @param IN Input info structure.
 * @param OUT Output info structure (uninitialized structure required).
 */
#ifdef __cplusplus
#define CVI_TDL_CopyInfo(IN, OUT) CVI_TDL_CopyInfoCpp(IN, OUT)
#else
// clang-format off
#define CVI_TDL_CopyInfoG(OUT) _Generic((OUT),                       \
           cvtdl_face_info_t*: CVI_TDL_CopyFaceInfo,                  \
           cvtdl_object_info_t*: CVI_TDL_CopyObjectInfo,              \
           cvtdl_image_t*: CVI_TDL_CopyImage)
#define CVI_TDL_CopyInfo(IN, OUT) _Generic((IN),                     \
           cvtdl_face_info_t*: CVI_TDL_CopyInfoG(OUT),                \
           cvtdl_object_info_t*: CVI_TDL_CopyInfoG(OUT),              \
           cvtdl_image_t*: CVI_TDL_CopyInfoG(OUT))((IN), (OUT))
// clang-format on
#endif

/** @def CVI_TDL_RescaleMetaCenter
 * @ingroup core_cvitdlcore
 * @brief Rescale the output coordinate to original image. Padding in four directions. Support the
 * following structure types written in _Generic.
 *
 * @param videoFrame Original input image.
 * @param X Input data structure.
 */

/** @def CVI_TDL_RescaleMetaRB
 * @ingroup core_cvitdlcore
 * @brief Rescale the output coordinate to original image. Padding in right, bottom directions.
 * Support the following structure types written in _Generic.
 *
 * @param videoFrame Original input image.
 * @param X Input data structure.
 */
#ifdef __cplusplus
#define CVI_TDL_RescaleMetaCenter(videoFrame, X) CVI_TDL_RescaleMetaCenterCpp(videoFrame, X);
#define CVI_TDL_RescaleMetaRB(videoFrame, X) CVI_TDL_RescaleMetaRBCpp(videoFrame, X);
#else
// clang-format off
#define CVI_TDL_RescaleMetaCenter(videoFrame, X) _Generic((X), \
           cvtdl_face_t*: CVI_TDL_RescaleMetaCenterFace,        \
           cvtdl_object_t*: CVI_TDL_RescaleMetaCenterObj)(videoFrame, X);
#define CVI_TDL_RescaleMetaRB(videoFrame, X) _Generic((X),     \
           cvtdl_face_t*: CVI_TDL_RescaleMetaRBFace,            \
           cvtdl_object_t*: CVI_TDL_RescaleMetaRBObj)(videoFrame, X);
// clang-format on
#endif

/** @typedef cvitdl_handle_t
 * @ingroup core_cvitdlcore
 * @brief An cvitdl handle
 */
typedef void *cvitdl_handle_t;

/**
 * \addtogroup core Inference Functions
 * \ingroup core_cvitdlcore
 */

/**
 * \addtogroup core_tdl_settings TDL Inference Setting Functions
 * \ingroup core_tdl
 */
/**@{*/

/**
 * IMPORTENT!! Add supported model here!
 */
// clang-format off
#define CVI_TDL_MODEL_LIST \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_RETINAFACE)                       \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_SCRFDFACE)                       \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_RETINAFACE_IR)                    \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_RETINAFACE_HARDHAT)               \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_THERMALFACE)                      \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_THERMALPERSON)                    \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_FACEATTRIBUTE)                    \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_FACERECOGNITION)                  \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_MASKFACERECOGNITION)              \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_FACEQUALITY)                      \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_MASKCLASSIFICATION)               \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_HANDCLASSIFICATION)               \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_HAND_KEYPOINT)                    \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_HAND_KEYPOINT_CLASSIFICATION)     \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_LIVENESS)                         \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_HAND_DETECTION)                    \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE)       \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE)              \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN)           \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_PERSON_PETS)          \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_COCO80)               \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_YOLOV3)                           \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_YOLOV5)                           \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_YOLOV6)                           \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_YOLOV7)                           \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_YOLO)                           \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_YOLOX)                           \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_PPYOLOE)                           \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_OSNET)                            \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_SOUNDCLASSIFICATION)              \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_SOUNDCLASSIFICATION_V2)            \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_WPODNET)                          \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_LPRNET_TW)                        \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_LPRNET_CN)                        \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_DEEPLABV3)                        \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_ALPHAPOSE)                        \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_EYECLASSIFICATION)                \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_YAWNCLASSIFICATION)               \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_FACELANDMARKER)                   \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_FACELANDMARKERDET2)               \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_INCAROBJECTDETECTION)             \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_SMOKECLASSIFICATION)              \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_FACEMASKDETECTION)                \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_IRLIVENESS)                       \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_PERSON_PETS_DETECTION)            \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_YOLOV8_DETECTION)                 \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_PERSON_VEHICLE_DETECTION)         \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_HAND_FACE_PERSON_DETECTION)       \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_HEAD_PERSON_DETECTION)            \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_YOLOV8POSE)                       \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_SIMCC_POSE)                       \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_LANDMARK_DET3)                    \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_LP_RECONGNITION)                  \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_LP_DETECTION)                     \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_IMAGE_CLASSIFICATION)             \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_HRNET_POSE)                       \
  CVI_TDL_NAME_WRAP(CVI_TDL_SUPPORTED_MODEL_DMSLANDMARKERDET)                 \
// clang-format on

#define CVI_TDL_NAME_WRAP(x) x,

/** @enum CVI_TDL_SUPPORTED_MODEL_E
 * @brief Supported NN model list. Can be used to config function behavior.
 *
 */
typedef enum { CVI_TDL_MODEL_LIST CVI_TDL_SUPPORTED_MODEL_END } CVI_TDL_SUPPORTED_MODEL_E;
#undef CVI_TDL_NAME_WRAP

#define CVI_TDL_NAME_WRAP(x) #x,

static inline const char *CVI_TDL_GetModelName(CVI_TDL_SUPPORTED_MODEL_E index) {
  static const char *model_names[] = {CVI_TDL_MODEL_LIST};
  uint32_t length = sizeof(model_names) / sizeof(model_names[0]);
  if (index < length) {
    return model_names[index];
  } else {
    return "Unknown";
  }
}

#undef CVI_TDL_NAME_WRAP
#undef CVI_TDL_MODEL_LIST

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief Create a cvitdl_handle_t, will automatically find a vpss group id.
 *
 * @param handle An TDL SDK handle.
 * @return int Return CVI_TDL_SUCCESS if succeed.
 */
DLL_EXPORT CVI_S32 CVI_TDL_CreateHandle(cvitdl_handle_t *handle);

/**
 * @brief Create a cvitdl_handle_t, need to manually assign a vpss group id.
 *
 * @param handle An TDL SDK handle.
 * @param vpssGroupId Assign a group id to cvitdl_handle_t.
 * @param vpssDev Assign a device id to cvitdl_handle_t.
 * @return int Return CVI_TDL_SUCCESS if succeed.
 */
DLL_EXPORT CVI_S32 CVI_TDL_CreateHandle2(cvitdl_handle_t *handle, const VPSS_GRP vpssGroupId,
                                        const CVI_U8 vpssDev);

DLL_EXPORT CVI_S32 CVI_TDL_CreateHandle3(cvitdl_handle_t *handle);

/**
 * @brief Destroy a cvitdl_handle_t.
 *
 * @param handle An TDL SDK handle.
 * @return int Return CVI_TDL_SUCCESS if success to destroy handle.
 */
DLL_EXPORT CVI_S32 CVI_TDL_DestroyHandle(cvitdl_handle_t handle);

/**
 * @brief Open model with given file path.
 *
 * @param handle An TDL SDK handle.
 * @param model Supported model id.
 * @param filepath File path to the cvimodel file.
 * @return int Return CVI_TDL_SUCCESS if load model succeed.
 */
DLL_EXPORT CVI_S32 CVI_TDL_OpenModel(cvitdl_handle_t handle, CVI_TDL_SUPPORTED_MODEL_E model,
                                    const char *filepath);

/**
 * @brief Get set model path from supported models.
 *
 * @param handle An TDL SDK handle.
 * @param model Supported model id.
 * @return model path.
 */
DLL_EXPORT const char *CVI_TDL_GetModelPath(cvitdl_handle_t handle, CVI_TDL_SUPPORTED_MODEL_E model);

/**
 * @brief Set skip vpss preprocess for supported networks.
 *
 * @param handle An TDL SDK handle.
 * @param model Supported model id.
 * @param skip To skip preprocess or not.
 * @return int Return CVI_TDL_SUCCESS if load model succeed.
 */
DLL_EXPORT CVI_S32 CVI_TDL_SetSkipVpssPreprocess(cvitdl_handle_t handle,
                                                CVI_TDL_SUPPORTED_MODEL_E model, bool skip);

/**
 * @brief Set skip vpss preprocess for supported networks.
 *
 * @param handle An TDL SDK handle.
 * @param model Supported model id.
 * @param interval number of frames used to performance evaluation
 * @return int Return CVI_TDL_SUCCESS if load model succeed.
 */
DLL_EXPORT CVI_S32 CVI_TDL_SetPerfEvalInterval(cvitdl_handle_t handle,
                                              CVI_TDL_SUPPORTED_MODEL_E config, int interval);

/**
 * @brief Set list depth for VPSS.
 *
 * @param handle An TDL SDK handle.
 * @param model Supported model id.
 * @param input_id input index of model.
 * @param depth list depth of VPSS.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_SetVpssDepth(cvitdl_handle_t handle, CVI_TDL_SUPPORTED_MODEL_E model,
                                       uint32_t input_id, uint32_t depth);

/**
 * @brief Get list depth for VPSS.
 *
 * @param handle An TDL SDK handle.
 * @param model Supported model id.
 * @param input_id input index of model.
 * @param depth list depth of VPSS.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_GetVpssDepth(cvitdl_handle_t handle, CVI_TDL_SUPPORTED_MODEL_E model,
                                       uint32_t input_id, uint32_t *depth);

/**
 * @brief Get skip preprocess value for given supported model.
 *
 * @param handle An TDL SDK handle.
 * @param model Supported model id.
 * @param skip Output setting value.
 * @return int Return CVI_TDL_SUCCESS.
 */
DLL_EXPORT CVI_S32 CVI_TDL_GetSkipVpssPreprocess(cvitdl_handle_t handle,
                                                CVI_TDL_SUPPORTED_MODEL_E model, bool *skip);

/**
 * @brief Set the threshold of an TDL inference.
 *
 * @param handle An TDL SDK handle.
 * @param model Supported model id.
 * @param threshold Threshold in float, usually a number between 0 and 1.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_SetModelThreshold(cvitdl_handle_t handle, CVI_TDL_SUPPORTED_MODEL_E model,
                                            float threshold);

/**
 * @brief Set the nms threshold of yolo inference.
 *
 * @param handle An TDL SDK handle.
 * @param model Supported model id.
 * @param threshold Threshold in float, usually a number between 0 and 1.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_SetModelNmsThreshold(cvitdl_handle_t handle,
                                               CVI_TDL_SUPPORTED_MODEL_E model, float threshold);

/**
 * @brief Get the threshold of an TDL Inference
 *
 * @param handle An TDL SDK handle.
 * @param model Supported model id.
 * @param threshold Threshold in float.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_GetModelThreshold(cvitdl_handle_t handle, CVI_TDL_SUPPORTED_MODEL_E model,
                                            float *threshold);

/**
 * @brief Get the nms threshold of yolo Inference
 *
 * @param handle An TDL SDK handle.
 * @param model Supported model id.
 * @param threshold Threshold in float.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_GetModelNmsThreshold(cvitdl_handle_t handle,
                                               CVI_TDL_SUPPORTED_MODEL_E model, float *threshold);
/**
 * @brief Set different vpss thread for each model. Vpss group id is not thread safe. We recommended
 * to change a thread if the process is not sequential.
 *
 * @param handle An TDL SDK handle.
 * @param model Supported model id.
 * @param thread The vpss thread index user desired. Note this param will changed if previous index
 * is not used.
 * @return int Return CVI_TDL_SUCCESS if successfully changed.
 */
DLL_EXPORT CVI_S32 CVI_TDL_SetVpssThread(cvitdl_handle_t handle, CVI_TDL_SUPPORTED_MODEL_E model,
                                        const uint32_t thread);

/**
 * @brief Set different vpss thread for each model. Vpss group id is not thread safe. We recommended
 * to change a thread if the process is not sequential. This function requires manually assigning a
 * vpss group id and device id.
 *
 * @param handle An TDL SDK handle.
 * @param model Supported model id.
 * @param thread The vpss thread index user desired. Note this param will changed if previous index
 * is not used.
 * @param vpssGroupId Assign a vpss group id if a new vpss instance needs to be created.
 * @param dev Assign Vpss device id to Vpss group
 * @return int Return CVI_TDL_SUCCESS if successfully changed.
 */
DLL_EXPORT CVI_S32 CVI_TDL_SetVpssThread2(cvitdl_handle_t handle, CVI_TDL_SUPPORTED_MODEL_E model,
                                         const uint32_t thread, const VPSS_GRP vpssGroupId,
                                         const CVI_U8 dev);

/**
 * @brief Get the set thread index for given supported model.
 *
 * @param handle An TDL SDK handle.
 * @param model Supported model id.
 * @param thread Output thread index.
 * @return int Return CVI_TDL_SUCCESS.
 */
DLL_EXPORT CVI_S32 CVI_TDL_GetVpssThread(cvitdl_handle_t handle, CVI_TDL_SUPPORTED_MODEL_E model,
                                        uint32_t *thread);

/**
 * @brief Set VB pool id to VPSS in TDLSDK. By default, VPSS in TDLSDK acquires VB from all
 * system-wide VB_POOLs which are created via CVI_VB_Init. In this situation, system decides which
 * VB_POOL is used according to VB allocation mechanism. The size of aquired VB maybe not optimal
 * and it could cause resource competition. To prevents this problem, you can assign a specific
 * VB_POOL to TDLSDK via this function. The VB_POOL created by CVI_VB_Init or CVI_VB_CreatePool are
 * both accepted.
 *
 * @param handle An TDL SDK handle.
 * @param thread VPSS thread index.
 * @param pool_id vb pool id. if pool id is VB_INVALID_POOLID than VPSS will get VB from all
 * system-wide VB_POOLs like default.
 * @return int Return CVI_TDL_SUCCESS.
 */
DLL_EXPORT CVI_S32 CVI_TDL_SetVBPool(cvitdl_handle_t handle, uint32_t thread, VB_POOL pool_id);

/**
 * @brief Get VB pool id used by internal VPSS.
 *
 * @param handle An TDL SDK handle.
 * @param thread VPSS thread index.
 * @param pool_id Output vb pool id.
 * @return int Return CVI_TDL_SUCCESS.
 */
DLL_EXPORT CVI_S32 CVI_TDL_GetVBPool(cvitdl_handle_t handle, uint32_t thread, VB_POOL *pool_id);

/**
 * @brief Get the vpss group ids used by the handle.
 *
 * @param handle An TDL SDK handle.
 * @param groups Return the list of used vpss group id.
 * @param num Return the length of the list.
 * @return int Return CVI_TDL_SUCCESS.
 */
DLL_EXPORT CVI_S32 CVI_TDL_GetVpssGrpIds(cvitdl_handle_t handle, VPSS_GRP **groups, uint32_t *num);

/**
 * @brief Set VPSS waiting time.
 *
 * @param handle An TDL SDK handle.
 * @param timeout Timeout value.
 * @return int Return CVI_TDL_SUCCESS.
 */
DLL_EXPORT CVI_S32 CVI_TDL_SetVpssTimeout(cvitdl_handle_t handle, uint32_t timeout);

/**
 * @brief Close all opened models and delete the model instances.
 *
 * @param handle An TDL SDK handle.
 * @return int Return CVI_TDL_SUCCESS if succeed.
 */
DLL_EXPORT CVI_S32 CVI_TDL_CloseAllModel(cvitdl_handle_t handle);

/**
 * @brief Close the chosen model and delete its model instance.
 *
 * @param handle An TDL SDK handle.
 * @param model Supported model id.
 * @return int Return CVI_TDL_SUCCESS if close succeed.
 */
DLL_EXPORT CVI_S32 CVI_TDL_CloseModel(cvitdl_handle_t handle, CVI_TDL_SUPPORTED_MODEL_E model);

/**
 * @brief Export vpss channel attribute.
 *
 * @param handle An TDL SDK handle.
 * @param model Supported model id.
 * @param frameWidth The input frame width.
 * @param frameHeight The input frame height.
 * @param idx The index of the input tensors.
 * @param chnAttr Exported VPSS channel config settings.
 * @return int Return CVI_TDL_SUCCESS on success, CVI_TDL_FAILURE if exporting not supported.
 */
DLL_EXPORT CVI_S32 CVI_TDL_GetVpssChnConfig(cvitdl_handle_t handle, CVI_TDL_SUPPORTED_MODEL_E model,
                                           const CVI_U32 frameWidth, const CVI_U32 frameHeight,
                                           const CVI_U32 idx, cvtdl_vpssconfig_t *chnConfig);

/**@}*/

/**
 * \addtogroup core_fd Face Detection TDL Inference
 * \ingroup core_tdl
 */
/**@{*/

/**
 * @brief RetinaFace face detection.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param faces Output detect result. The name, bbox, and face points will be given.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_RetinaFace(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                     cvtdl_face_t *faces);

/**
 * @brief ScrFD face detection.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param faces Output detect result. The name, bbox, and face points will be given.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_ScrFDFace(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                    cvtdl_face_t *faces);

DLL_EXPORT CVI_S32 CVI_TDL_FLDet3(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                 cvtdl_face_t *faces);
/**
 * @brief RetinaFaceIR face detection.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param faces Output detect result. The name, bbox, and face points will be given.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_RetinaFace_IR(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                        cvtdl_face_t *faces);
/**
 * @brief RetinaFace hardhat face detection.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param faces Output detect result. The name, bbox, and face points will be given.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_RetinaFace_Hardhat(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                             cvtdl_face_t *faces);

/**
 * @brief Detect face with thermal images.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input thermal video frame.
 * @param faces Output detect result. The bbox will be given.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_ThermalFace(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                      cvtdl_face_t *faces);

/**
 * @brief Detect person with thermal images.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input thermal video frame.
 * @param obj Output detect result. The bbox will be given.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_ThermalPerson(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                        cvtdl_object_t *obj);

/**
 * @brief Detect face with mask score.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param face_meta Output detect result. The bbox will be given.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_FaceMaskDetection(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                            cvtdl_face_t *face_meta);
/**@}*/

/**
 * \addtogroup core_fr Face Recognition TDL Inference
 * \ingroup core_tdl
 */
/**@{*/

/**
 * @brief Do face recognition and attribute with bbox info stored in faces.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param faces cvtdl_face_t structure, the cvtdl_face_info_t and cvtdl_bbox_t must be set.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_FaceAttribute(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                        cvtdl_face_t *faces);

/**
 * @brief Do face recognition and attribute with bbox info stored in faces. Only do inference on the
 * given index of cvtdl_face_info_t.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param faces cvtdl_face_t structure, the cvtdl_face_info_t and cvtdl_bbox_t must be set.
 * @param face_idx The index of cvtdl_face_info_t inside cvtdl_face_t.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_FaceAttributeOne(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                           cvtdl_face_t *faces, int face_idx);

/**
 * @brief Do face recognition with bbox info stored in faces.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param faces cvtdl_face_t structure, the cvtdl_face_info_t and cvtdl_bbox_t must be set.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_FaceRecognition(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                          cvtdl_face_t *faces);

/**
 * @brief Do face recognition with bbox info stored in faces. Only do inference on the given index
 * of cvtdl_face_info_t.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param faces cvtdl_face_t structure, the cvtdl_face_info_t and cvtdl_bbox_t must be set.
 * @param face_idx The index of cvtdl_face_info_t inside cvtdl_face_t.
 * @return int Return CVI_TDL_SUCCESS on success.
 */

/**
 * @brief Do face recognition with bbox info stored in faces.
 *
 * @param handle An TDL SDK handle.
 * @param p_rgb_pack Input video frame.
 * @param p_face_info, if no data in p_face_info,p_rgb_pack means aligned packed rgb data
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_FaceFeatureExtract(const cvitdl_handle_t handle, const uint8_t *p_rgb_pack,
                                             int width, int height, int stride,
                                             cvtdl_face_info_t *p_face_info);

DLL_EXPORT CVI_S32 CVI_TDL_FaceRecognitionOne(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                             cvtdl_face_t *faces, int face_idx);

/**
 * @brief Do face recognition with mask wearing.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param faces cvtdl_face_t structure, the cvtdl_face_info_t and cvtdl_bbox_t must be set.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_MaskFaceRecognition(const cvitdl_handle_t handle,
                                              VIDEO_FRAME_INFO_S *frame, cvtdl_face_t *faces);

/**@}*/

/**
 * \addtogroup core_fc Face classification TDL Inference
 * \ingroup core_tdl
 */
/**@{*/

/**
 * @brief FaceQuality. Assess the quality of the faces.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param face cvtdl_face_t structure, the cvtdl_face_info_t and cvtdl_bbox_t must be set.
 * @param skip bool array, whether skip quailty assessment at corresponding index (NULL for running
 * without skip)
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_FaceQuality(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                      cvtdl_face_t *face, bool *skip);

/**
 * @brief Crop image in given frame.
 *
 * @param srcFrame Input frame. (only support RGB Packed format)
 * @param dst Output image.
 * @param bbox The bounding box.
 * @param cvtRGB888 convert to RGB888 format.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_CropImage(VIDEO_FRAME_INFO_S *srcFrame, cvtdl_image_t *dst,
                                    cvtdl_bbox_t *bbox, bool cvtRGB888);

/**
 * @brief Crop image (extended) in given frame.
 *
 * @param srcFrame Input frame. (only support RGB Packed format)
 * @param dst Output image.
 * @param bbox The bounding box.
 * @param cvtRGB888 convert to RGB888 format.
 * @param exten_ratio extension ration.
 * @param offset_x original bounding box x offset.
 * @param offset_y original bounding box y offset.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_CropImage_Exten(VIDEO_FRAME_INFO_S *srcFrame, cvtdl_image_t *dst,
                                          cvtdl_bbox_t *bbox, bool cvtRGB888, float exten_ratio,
                                          float *offset_x, float *offset_y);

/**
 * @brief Crop face image in given frame.
 *
 * @param srcFrame Input frame. (only support RGB Packed format)
 * @param dst Output image.
 * @param face_info Face information, contain bbox and 5 landmark.
 * @param align Align face to standard size if true.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_CropImage_Face(VIDEO_FRAME_INFO_S *srcFrame, cvtdl_image_t *dst,
                                         cvtdl_face_info_t *face_info, bool align, bool cvtRGB888);

/**
 * @brief Liveness. Gives a score to present how real the face is. The score will be low if the face
 * is not directly taken by a camera.
 *
 * @param handle An TDL SDK handle.
 * @param rgbFrame Input RGB video frame.
 * @param irFrame Input IR video frame.
 * @param face cvtdl_face_t structure, the cvtdl_face_info_t and cvtdl_bbox_t must be set.
 * @param ir_position The position relationship netween the ir and the rgb camera.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_Liveness(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *rgbFrame,
                                   VIDEO_FRAME_INFO_S *irFrame, cvtdl_face_t *rgb_face,
                                   cvtdl_face_t *ir_face);

DLL_EXPORT CVI_S32 CVI_TDL_IrLiveness(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *irFrame,
                                     cvtdl_face_t *ir_face);

/**
 * @brief Mask classification. Tells if a face is wearing a mask.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param face cvtdl_face_t structure, the cvtdl_face_info_t and cvtdl_bbox_t must be set.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_MaskClassification(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                             cvtdl_face_t *face);

/**
 * @brief Hand classification.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param meta cvtdl_object_t structure.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_HandClassification(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                             cvtdl_object_t *meta);

/**
 * @brief 2D hand keypoints.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param meta cvtdl_handpose21_meta_ts structure.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_HandKeypoint(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                       cvtdl_handpose21_meta_ts *meta);

/**
 * @brief Hand classification by hand keypoints.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param meta cvtdl_handpose21_meta_t structure.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_HandKeypointClassification(const cvitdl_handle_t handle,
                                                     VIDEO_FRAME_INFO_S *frame,
                                                     cvtdl_handpose21_meta_t *meta);

/**
 * @brief Hand Detection. Tells if a object is wearing a mask.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param object cvtdl_object_t structure, the cvtdl_object_info_t and cvtdl_bbox_t must be set.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_Hand_Detection(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                         cvtdl_object_t *obj_meta);

/**
 * @brief Hand Detection. Tells if a object is wearing a mask.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param object cvtdl_object_t structure, the cvtdl_object_info_t and cvtdl_bbox_t must be set.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_HeadPerson_Detection(const cvitdl_handle_t handle,
                                               VIDEO_FRAME_INFO_S *frame, cvtdl_object_t *obj_meta);

/**@}*/

/**
 * \addtogroup core_od Object Detection TDL Inference
 * \ingroup core_tdl
 */
/**@{*/

/**
 * @brief Select classes for detection model. Model will output objects belong to these classes.
 * Currently only support MobileDetV2 family.
 *
 * @param handle An TDL SDK handle.
 * @param model model id.
 * @param num_classes number of classes you want to select.
 * @param ... class indexs
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_SelectDetectClass(cvitdl_handle_t handle, CVI_TDL_SUPPORTED_MODEL_E model,
                                            uint32_t num_classes, ...);

/**
 * @brief MobileDetV2 Vehicle object detectior, which can be used to detect "car", "truck", and
 * "motorbike" classes.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param obj Output detect result. The name, bbox, and classes will be given.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_MobileDetV2_Vehicle(cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                              cvtdl_object_t *obj);

/**
 * @brief MobileDetV2 pedestrian object detector, which can be used to detect "person" class
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param obj Output detect result. The name, bbox, and classes will be given.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_MobileDetV2_Pedestrian(cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                                 cvtdl_object_t *obj);

/**
 * @brief MobileDetV2 object detector, which can be used to detect "person", "bicycle", "car",
 * "motorbike", "bus", and "truck" classes.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param obj Output detect result. The name, bbox, and classes will be given.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_MobileDetV2_Person_Vehicle(cvitdl_handle_t handle,
                                                     VIDEO_FRAME_INFO_S *frame, cvtdl_object_t *obj);

/**
 * @brief MobileDetV2 object detector, which can be used to detect "person", "cat", and "dog"
 * classes.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param obj Output detect result. The name, bbox, and classes will be given.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_MobileDetV2_Person_Pets(cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                                  cvtdl_object_t *obj);

/**
 * @brief MobileDetV2 object detector which can be used to detect coco 80 classes objects.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param obj Output detect result. The name, bbox, and classes will be given.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_MobileDetV2_COCO80(cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                             cvtdl_object_t *obj);

/**
 * @brief Yolov3 object detection.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param obj Output detect result. The name, bbox, and classes will be given.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_Yolov3(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                 cvtdl_object_t *obj);

/**
 * @brief Yolov5 object detection.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param obj Output detect result. The name, bbox, and classes will be given.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_Yolov5(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                 cvtdl_object_t *obj);

/**
 * @brief Yolov6 object detection.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param obj Output detect result. The name, bbox, and classes will be given.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_Yolov6(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                 cvtdl_object_t *obj);

/**
 * @brief Yolov7 object detection.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param obj Output detect result. The name, bbox, and classes will be given.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_Yolov7(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                 cvtdl_object_t *obj);

/**
 * @brief Yolo object detection.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param obj Output detect result. The name, bbox, and classes will be given.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_Yolo(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                               cvtdl_object_t *obj);

/**
 * @brief YoloX object detection.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param obj Output detect result. The name, bbox, and classes will be given.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_YoloX(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                cvtdl_object_t *obj);

/**
 * @brief PP-Yoloe object detection.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param obj Output detect result. The name, bbox, and classes will be given.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_PPYoloE(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                  cvtdl_object_t *obj);

/**@}*/
/**
 * \addtogroup core_pr Person Re-Id TDL Inference
 * \ingroup core_tdl
 */
/**@{*/

/**
 * @brief Do person Re-Id with bbox info stored in obj.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param obj cvtdl_object_t structure, the cvtdl_object_info_t and cvtdl_bbox_t must be set.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_OSNet(cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                cvtdl_object_t *obj);

/**
 * @brief Do person Re-Id with bbox info stored in obj. Only do inference on the given index of
 * cvtdl_object_info_t.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param obj cvtdl_object_t structure, the cvtdl_object_info_t and cvtdl_bbox_t must be set.
 * @param obj_idx The index of cvtdl_object_info_t inside cvtdl_object_t.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_OSNetOne(cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                   cvtdl_object_t *obj, int obj_idx);

/**@}*/

/**
 * \addtogroup core_audio Audio TDL Inference
 * \ingroup core_tdl
 */
/**@{*/

/**
 * @brief Do sound classification.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param index The index of sound classes.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_SoundClassification(const cvitdl_handle_t handle,
                                              VIDEO_FRAME_INFO_S *frame, int *index);
/**
 * @brief Do sound classification.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param index The index of sound classes.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_SoundClassification_V2(const cvitdl_handle_t handle,
                                                 VIDEO_FRAME_INFO_S *frame, int *index);
/**
 * @brief Get sound classification classes num.
 *
 * @param handle An TDL SDK handle.
 * @return int Return CVI_TDL_SUCCESS on success.
 */

DLL_EXPORT CVI_S32 CVI_TDL_Get_SoundClassification_ClassesNum(const cvitdl_handle_t handle);

/**
 * @brief Set sound classification threshold.
 *
 * @param handle An TDL SDK handle.
 * @param th Sound classifiction threshold
 * @return int Return CVI_TDL_SUCCESS on success.
 */

DLL_EXPORT CVI_S32 CVI_TDL_Set_SoundClassification_Threshold(const cvitdl_handle_t handle,
                                                            const float th);

/**@}*/

/**
 * \addtogroup core_tracker Tracker
 * \ingroup core_tdl
 */
/**@{*/

/**
 * @brief Initialize DeepSORT.
 *
 * @param handle An TDL SDK handle.
 * @param use_specific_counter true for using individual id counter for each class
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_DeepSORT_Init(const cvitdl_handle_t handle, bool use_specific_counter);

/**
 * @brief Get default DeepSORT config.
 *
 * @param ds_conf Output config.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_DeepSORT_GetDefaultConfig(cvtdl_deepsort_config_t *ds_conf);

/**
 * @brief Get DeepSORT config.
 *
 * @param handle An TDL SDK handle.
 * @param ds_conf Output config.
 * @param cvitdl_obj_type The specific class type (-1 for setting default config).
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_DeepSORT_GetConfig(const cvitdl_handle_t handle,
                                             cvtdl_deepsort_config_t *ds_conf, int cvitdl_obj_type);

/**
 * @brief Set DeepSORT with specific config.
 *
 * @param handle An TDL SDK handle.
 * @param ds_conf The specific config.
 * @param cvitdl_obj_type The specific class type (-1 for setting default config).
 * @param show_config show detail information or not.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_DeepSORT_SetConfig(const cvitdl_handle_t handle,
                                             cvtdl_deepsort_config_t *ds_conf, int cvitdl_obj_type,
                                             bool show_config);

/**
 * @brief clean DeepSORT ID counter.
 *
 * @param handle An TDL SDK handle.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_DeepSORT_CleanCounter(const cvitdl_handle_t handle);

/**
 * @brief Run DeepSORT/SORT track for object.
 *
 * @param handle An TDL SDK handle.
 * @param obj Input detected object with feature.
 * @param tracker_t Output tracker results.
 * @param use_reid If true, track by DeepSORT algorithm, else SORT.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_DeepSORT_Obj(const cvitdl_handle_t handle, cvtdl_object_t *obj,
                                       cvtdl_tracker_t *tracker, bool use_reid);

/**
 * @brief Run DeepSORT/SORT track for object, add function to judge cross the border.
 *
 * @param handle An TDL SDK handle.
 * @param obj Input detected object with feature.
 * @param tracker_t Output tracker results.
 * @param use_reid If true, track by DeepSORT algorithm, else SORT.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_DeepSORT_Obj_Cross(const cvitdl_handle_t handle, cvtdl_object_t *obj,
                                             cvtdl_tracker_t *tracker, bool use_reid,
                                             const cvtdl_counting_line_t *cross_line_t,
                                             const randomRect *rect);

/**
 * @brief Run DeepSORT/SORT track for object to Consumer counting.
 *
 * @param handle An TDL SDK handle.
 * @param obj Input detected object with feature.
 * @param tracker_t Output tracker results.
 * @param use_reid If true, track by DeepSORT algorithm, else SORT.
 * @param head save head data
 * @param ped save ped data
 * @param counting_line_t the line of consumer counting
 * @param counting_line_t the buffer rectangle
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_DeepSORT_Head_FusePed(const cvitdl_handle_t handle, cvtdl_object_t *obj,
                                                cvtdl_tracker_t *tracker_t, bool use_reid,
                                                cvtdl_object_t *head, cvtdl_object_t *ped,
                                                const cvtdl_counting_line_t *counting_line_t,
                                                const randomRect *rect);

/**
 * @brief Run SORT track for face.
 *
 * @param handle An TDL SDK handle.
 * @param face Input detected face with feature.
 * @param tracker_t Output tracker results.
 * @param use_reid Set false for SORT.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_DeepSORT_Face(const cvitdl_handle_t handle, cvtdl_face_t *face,
                                        cvtdl_tracker_t *tracker);

DLL_EXPORT CVI_S32 CVI_TDL_DeepSORT_FaceFusePed(const cvitdl_handle_t handle, cvtdl_face_t *face,
                                               cvtdl_object_t *obj, cvtdl_tracker_t *tracker_t);

DLL_EXPORT CVI_S32 CVI_TDL_DeepSORT_Set_Timestamp(const cvitdl_handle_t handle, uint32_t ts);

DLL_EXPORT CVI_S32 CVI_TDL_DeepSORT_UpdateOutNum(const cvitdl_handle_t handle,
                                                cvtdl_tracker_t *tracker_t);
DLL_EXPORT CVI_S32 CVI_TDL_DeepSORT_DebugInfo_1(const cvitdl_handle_t handle, char *debug_info);

DLL_EXPORT CVI_S32 CVI_TDL_DeepSORT_GetTracker_Inactive(const cvitdl_handle_t handle,
                                                       cvtdl_tracker_t *tracker);

/**@}*/

/**
 * \addtogroup core_segmentation Segmentation Inference
 * \ingroup core_tdl
 */
/**@{*/

/**
 * @brief Deeplabv3 segmentation.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param out_frame Output frame which each pixel represents class label.
 * @param filter Class id filter. Set NULL to ignore.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_DeeplabV3(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                    VIDEO_FRAME_INFO_S *out_frame, cvtdl_class_filter_t *filter);
/**@}*/

/**
 * @brief CVI_TDL_License_Plate_Recognitionv2.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param vehicle License plate object info
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_License_Plate_Recognitionv2(const cvitdl_handle_t handle,
                                                      VIDEO_FRAME_INFO_S *frame,
                                                      cvtdl_object_t *vehicle);
/**
 * @brief LicensePlateDetection
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param vehicle License plate object info
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_License_Plate_Detectionv2(const cvitdl_handle_t handle,
                                                    VIDEO_FRAME_INFO_S *frame,
                                                    cvtdl_object_t *vehicle);

/*useless: old license plate detection and recongition*/
DLL_EXPORT CVI_S32 CVI_TDL_LicensePlateDetection(const cvitdl_handle_t handle,
                                                VIDEO_FRAME_INFO_S *frame,
                                                cvtdl_object_t *vehicle_meta);
DLL_EXPORT CVI_S32 CVI_TDL_LicensePlateRecognition_CN(const cvitdl_handle_t handle,
                                                     VIDEO_FRAME_INFO_S *frame,
                                                     cvtdl_object_t *vehicle);
DLL_EXPORT CVI_S32 CVI_TDL_LicensePlateRecognition_TW(const cvitdl_handle_t handle,
                                                     VIDEO_FRAME_INFO_S *frame,
                                                     cvtdl_object_t *vehicle);

/**@}*/

/**
 * \addtogroup core_pose Pose Detection
 * \ingroup core_tdl
 */
/**@{*/
/**
 * @brief Alphapose.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param vehicle_meta Detected object info
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_AlphaPose(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                    cvtdl_object_t *objects);
/**@}*/

/**
 * \addtogroup core_fall Fall Detection
 * \ingroup core_tdl
 */
/**@{*/
/**
 * @brief Fall.
 *
 * @param handle An TDL SDK handle.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_Fall(const cvitdl_handle_t handle, cvtdl_object_t *objects);
/**@{*/

/**
 * \addtogroup core_fall Fall Detection new API
 * \ingroup core_tdl
 */
/**@{*/

/**
 * @brief Fall Detection.
 *
 * @param handle An TDL SDK handle.
 * @param objects Detected object info.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_Fall_Monitor(const cvitdl_handle_t handle, cvtdl_object_t *objects);

/**
 * @brief Set fall detection FPS.
 *
 * @param handle An TDL SDK handle.
 * @param fps Current frame fps.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_Set_Fall_FPS(const cvitdl_handle_t handle, float fps);

/**
 * \addtogroup core_others Others
 * \ingroup core_tdl
 */
/**@{*/

/**
 * @brief Do background subtraction.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param moving_score Check the unit diff sum of a frame.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_TamperDetection(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                          float *moving_score);

/**
 * @brief Set background frame for motion detection.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame, should be YUV420 format.
 * be returned.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_Set_MotionDetection_Background(const cvitdl_handle_t handle,
                                                         VIDEO_FRAME_INFO_S *frame);

/**
 * @brief Set ROI frame for motion detection.
 *
 * @param handle An TDL SDK handle.
 * @param x1 left x coordinate of roi
 * @param y1 top y coordinate of roi
 * @param x2 right x coordinate of roi
 * @param y2 bottom y coordinate of roi
 *
 * be returned.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_Set_MotionDetection_ROI(const cvitdl_handle_t handle, MDROI_t *roi_s);
/**
 * @brief Do Motion Detection with background subtraction method.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame, should be YUV420 format.
 * @param objects Detected object info
 * @param threshold Threshold of motion detection, the range between 0 and 255.
 * @param min_area Minimal pixel area. The bounding box whose area is larger than this value would
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_MotionDetection(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                          cvtdl_object_t *objects, uint8_t threshold,
                                          double min_area);

/**@}*/

/**
 * \addtogroup core_dms Driving Monitor System
 * \ingroup core_tdl
 */
/**@{*/

/**
 * @brief Do eye classification.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param cvtdl_face_t structure. Calculate the eye_score in cvtdl_dms_t.
 * @return int Return CVI_TDL_SUCCESS on success.
 */

DLL_EXPORT CVI_S32 CVI_TDL_EyeClassification(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                            cvtdl_face_t *face);

/**
 * @brief Do yawn classification.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param cvtdl_face_t structure. Calculate the yawn_score in cvtdl_dms_t.
 * @return int Return CVI_TDL_SUCCESS on success.
 */

DLL_EXPORT CVI_S32 CVI_TDL_YawnClassification(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                             cvtdl_face_t *face);
/**
 * @brief Do face landmark.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param cvtdl_face_t structure. Calculate the landmarks in cvtdl_dms_t.
 * @return int Return CVI_TDL_SUCCESS on success.
 */

DLL_EXPORT CVI_S32 CVI_TDL_IncarObjectDetection(const cvitdl_handle_t handle,
                                               VIDEO_FRAME_INFO_S *frame, cvtdl_face_t *face);

/**@}*/

/**
 * \addtogroup core_face_landmark Face Landmark
 * \ingroup core_tdl
 */
/**@{*/

DLL_EXPORT CVI_S32 CVI_TDL_FaceLandmarker(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                         cvtdl_face_t *face);

/**@}*/

/**
 * \addtogroup core_face_landmarkdet3 Face Landmark
 * \ingroup core_tdl
 */
/**@{*/

DLL_EXPORT CVI_S32 CVI_TDL_FaceLandmarkerDet2(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                             cvtdl_face_t *face);

/**@}*/

/**
 * \addtogroup core_ Dms face Landmark
 * \ingroup core_tdl
 */
/**@{*/
DLL_EXPORT CVI_S32 CVI_TDL_DMSLDet(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                  cvtdl_face_t *face);

/**
 * @brief Do smoke classification.
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param cvtdl_face_t structure. Calculate the smoke_score in cvtdl_dms_t.
 * @return int Return CVI_TDL_SUCCESS on success.
 */

DLL_EXPORT CVI_S32 CVI_TDL_SmokeClassification(const cvitdl_handle_t handle,
                                              VIDEO_FRAME_INFO_S *frame, cvtdl_face_t *face);

/**
 * @brief Dump model input frame to npz.
 *
 * @param handle An TDL SDK handle.
 * @param model Model id.
 * @param dump_path Output path.
 * @param enable Whether enable or not.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
/**@}*/
DLL_EXPORT CVI_S32 CVI_TDL_EnalbeDumpInput(cvitdl_handle_t handle, CVI_TDL_SUPPORTED_MODEL_E model,
                                          const char *dump_path, bool enable);

DLL_EXPORT CVI_S32 CVI_TDL_CropImage_With_VPSS(const cvitdl_handle_t handle,
                                              CVI_TDL_SUPPORTED_MODEL_E model,
                                              VIDEO_FRAME_INFO_S *frame,
                                              const cvtdl_bbox_t *p_crop_box, cvtdl_image_t *p_dst);

DLL_EXPORT CVI_S32 CVI_TDL_CropResizeImage(const cvitdl_handle_t handle,
                                          CVI_TDL_SUPPORTED_MODEL_E model_type,
                                          VIDEO_FRAME_INFO_S *frame, const cvtdl_bbox_t *p_crop_box,
                                          int dst_width, int dst_height, PIXEL_FORMAT_E enDstFormat,
                                          VIDEO_FRAME_INFO_S **p_dst_img);

DLL_EXPORT CVI_S32 CVI_TDL_Copy_VideoFrameToImage(VIDEO_FRAME_INFO_S *frame, cvtdl_image_t *p_dst);
DLL_EXPORT CVI_S32 CVI_TDL_Resize_VideoFrame(const cvitdl_handle_t handle,
                                            CVI_TDL_SUPPORTED_MODEL_E model,
                                            VIDEO_FRAME_INFO_S *frame, const int dst_w,
                                            const int dst_h, PIXEL_FORMAT_E dst_format,
                                            VIDEO_FRAME_INFO_S **dst_frame);
DLL_EXPORT CVI_S32 CVI_TDL_Release_VideoFrame(const cvitdl_handle_t handle,
                                             CVI_TDL_SUPPORTED_MODEL_E model,
                                             VIDEO_FRAME_INFO_S *frame, bool del_frame);
DLL_EXPORT CVI_S32 CVI_TDL_Change_Img(const cvitdl_handle_t handle,
                                     CVI_TDL_SUPPORTED_MODEL_E model_type, VIDEO_FRAME_INFO_S *frame,
                                     VIDEO_FRAME_INFO_S **dst_frame, PIXEL_FORMAT_E enDstFormat);

DLL_EXPORT CVI_S32 CVI_TDL_Delete_Img(const cvitdl_handle_t handle,
                                     CVI_TDL_SUPPORTED_MODEL_E model_type, VIDEO_FRAME_INFO_S *p_f);
/**
 * @brief person and pet(cat,dog) Detection
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param object cvtdl_object_t structure, the cvtdl_object_info_t and cvtdl_bbox_t must be set.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_PersonPet_Detection(const cvitdl_handle_t handle,
                                              VIDEO_FRAME_INFO_S *frame, cvtdl_object_t *obj_meta);

/**
 * @brief Yolov8 Detection
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param object cvtdl_object_t structure, the cvtdl_object_info_t and cvtdl_bbox_t must be set.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_YOLOV8_Detection(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                           cvtdl_object_t *obj_meta);

/**
 * @brief car,bus,truck,person,bike,motor Detection
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param object cvtdl_object_t structure, the cvtdl_object_info_t and cvtdl_bbox_t must be set.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_PersonVehicle_Detection(const cvitdl_handle_t handle,
                                                  VIDEO_FRAME_INFO_S *frame,
                                                  cvtdl_object_t *obj_meta);
/**
 * @brief hand,face,person Detection
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param object cvtdl_object_t structure, the cvtdl_object_info_t and cvtdl_bbox_t must be set.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_HandFacePerson_Detection(const cvitdl_handle_t handle,
                                                   VIDEO_FRAME_INFO_S *frame,
                                                   cvtdl_object_t *obj_meta);
/**
 * @brief human keypoints detection
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param object cvtdl_object_t structure, the cvtdl_object_info_t and cvtdl_bbox_t and
 * pedestrian_properity must be set.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_Yolov8_Pose(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                      cvtdl_object_t *obj_meta);
/**
 * @brief human keypoints detection
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param object cvtdl_object_t structure, the cvtdl_object_info_t and cvtdl_bbox_t and
 * pedestrian_properity must be set.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_Simcc_Pose(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                     cvtdl_object_t *obj_meta);
/**
 * @brief human keypoints detection
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param object cvtdl_object_t structure, the cvtdl_object_info_t and cvtdl_bbox_t and
 * pedestrian_properity must be set.
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_Hrnet_Pose(const cvitdl_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                     cvtdl_object_t *obj_meta);
/**
 * @brief image classification
 *
 * @param handle An TDL SDK handle.
 * @param frame Input video frame.
 * @param object cvtdl_class_meta_t structure, top 5 class info and score
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_Image_Classification(const cvitdl_handle_t handle,
                                               VIDEO_FRAME_INFO_S *frame,
                                               cvtdl_class_meta_t *obj_meta);

/**
 * @brief get yolo preprocess param struct
 *
 * @param handle An TDL SDK handle.
 * @param model_index Supported model list.
 * @return  YoloPreparam yolo preprocess param struct.
 */
DLL_EXPORT YoloPreParam CVI_TDL_Get_YOLO_Preparam(const cvitdl_handle_t handle, const CVI_TDL_SUPPORTED_MODEL_E model_index);

/**
 * @brief set yolo preprocess param struct
 *
 * @param handle An TDL SDK handle.
 * @param model_index Supported model list.
 * @param pre_param Yolo preprocess struct
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_Set_YOLO_Preparam(const cvitdl_handle_t handle, const CVI_TDL_SUPPORTED_MODEL_E model_index, YoloPreParam pre_param);

/**
 * @brief get yolo algorithm param struct
 *
 * @param handle An TDL SDK handle.
 * @param model_index Supported model list.
 * @return  YoloAlgparam yolo algorthm param struct.
 */
DLL_EXPORT YoloAlgParam CVI_TDL_Get_YOLO_Algparam(const cvitdl_handle_t handle, const CVI_TDL_SUPPORTED_MODEL_E model_index);

/**
 * @brief set yolo algorithm param struct
 *
 * @param handle An TDL SDK handle.
 * @param model_index Supported model list.
 * @param alg_param Yolo algorithm struct
 * @return int Return CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_Set_YOLO_Algparam(const cvitdl_handle_t handle, const CVI_TDL_SUPPORTED_MODEL_E model_index, YoloAlgParam alg_param);

DLL_EXPORT CVI_S32 CVI_TDL_Set_Yolov5_ROI(const cvitdl_handle_t handle, Point_t roi_s);
/**
 * @brief image_classification setup function
 *
 * @param handle An TDL SDK handle.
 * @param p_preprocess_cfg Input preprocess setup config.
 * @int Reture CVI_TDL_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_TDL_DeepSORT_Byte(const cvitdl_handle_t handle, cvtdl_object_t *obj,
                                        cvtdl_tracker_t *tracker, bool use_reid);

DLL_EXPORT CVI_S32 CVI_TDL_Set_Image_Cls_Param(const cvitdl_handle_t handle,
                                              VpssPreParam *p_preprocess_cfg);
DLL_EXPORT CVI_S32 CVI_TDL_Set_YOLO_Param(const cvitdl_handle_t handle,
                                         YoloPreParam *p_preprocess_cfg,
                                         YoloAlgParam *p_yolo_param);
#ifdef __cplusplus
}
#endif

#endif  // End of _CVI_TDL_CORE_H_
