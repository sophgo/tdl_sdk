#ifndef _CVIAI_CORE_H_
#define _CVIAI_CORE_H_
#include "core/core/cvai_core_types.h"
#include "core/cviai_rescale_bbox.h"
#include "core/cviai_types_mem.h"
#include "core/deepsort/cvai_deepsort_types.h"
#include "core/face/cvai_face_helper.h"
#include "core/face/cvai_face_types.h"
#include "core/object/cvai_object_types.h"

#include <cvi_comm_vb.h>
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

/** @def CVI_AI_RescaleBBoxCenter
 * @ingroup core_cviaicore
 * @brief Rescale the output coordinate to original image. Padding in four directions. Support the
 * following structure types written in _Generic.
 *
 * @param videoFrame Original input image.
 * @param X Input data structure.
 */

/** @def CVI_AI_RescaleBBoxRB
 * @ingroup core_cviaicore
 * @brief Rescale the output coordinate to original image. Padding in right, bottom directions.
 * Support the following structure types written in _Generic.
 *
 * @param videoFrame Original input image.
 * @param X Input data structure.
 */
#ifdef __cplusplus
#define CVI_AI_RescaleBBoxCenter(videoFrame, X) CVI_AI_RescaleBBoxCenterCpp(videoFrame, X);
#define CVI_AI_RescaleBBoxRB(videoFrame, X) CVI_AI_RescaleBBoxRBCpp(videoFrame, X);
#else
// clang-format off
#define CVI_AI_RescaleBBoxCenter(videoFrame, X) _Generic((X), \
           cvai_face_t*: CVI_AI_RescaleBBoxCenterFace,        \
           cvai_object_t*: CVI_AI_RescaleBBoxCenterObj)(videoFrame, X);
#define CVI_AI_RescaleBBoxRB(videoFrame, X) _Generic((X),     \
           cvai_face_t*: CVI_AI_RescaleBBoxRBFace,            \
           cvai_object_t*: CVI_AI_RescaleBBoxRBObj)(videoFrame, X);
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

/** @enum CVI_AI_SUPPORTED_MODEL_E
 * @brief Supported NN model list. Can be used to config function behavior.
 *
 */
typedef enum {
  CVI_AI_SUPPORTED_MODEL_RETINAFACE,
  CVI_AI_SUPPORTED_MODEL_THERMALFACE,
  CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE,
  CVI_AI_SUPPORTED_MODEL_FACERECOGNITION,
  CVI_AI_SUPPORTED_MODEL_MASKFACERECOGNITION,
  CVI_AI_SUPPORTED_MODEL_FACEQUALITY,
  CVI_AI_SUPPORTED_MODEL_LIVENESS,
  CVI_AI_SUPPORTED_MODEL_MASKCLASSIFICATION,
  CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D0,
  CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D1,
  CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D2,
  CVI_AI_SUPPORTED_MODEL_YOLOV3,
  CVI_AI_SUPPORTED_MODEL_OSNET,
  CVI_AI_SUPPORTED_MODEL_ESCLASSIFICATION,
  CVI_AI_SUPPORTED_MODEL_END
} CVI_AI_SUPPORTED_MODEL_E;

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief Create a cviai_handle_t, will automatically find a vpss group id.
 *
 * @param handle An AI SDK handle.
 * @return int Return CVI_SUCCESS if succeed.
 */
DLL_EXPORT int CVI_AI_CreateHandle(cviai_handle_t *handle);

/**
 * @brief Create a cviai_handle_t, need to manually assign a vpss group id.
 *
 * @param handle An AI SDK handle.
 * @param vpssGroupId Assign a group id to cviai_handle_t.
 * @return int Return CVI_SUCCESS if succeed.
 */
DLL_EXPORT int CVI_AI_CreateHandle2(cviai_handle_t *handle, const VPSS_GRP vpssGroupId);

/**
 * @brief Destroy a cviai_handle_t.
 *
 * @param handle An AI SDK handle.
 * @return int Return CVI_SUCCESS if success to destroy handle.
 */
DLL_EXPORT int CVI_AI_DestroyHandle(cviai_handle_t handle);

/**
 * @brief Set the model path for supported networks.
 *
 * @param handle An AI SDK handle.
 * @param config Supported model type config.
 * @param filepath File path to the cvimodel file.
 * @return int Return CVI_SUCCESS if load model succeed.
 */
DLL_EXPORT int CVI_AI_SetModelPath(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                   const char *filepath);

/**
 * @brief Get set model path from supported models.
 *
 * @param handle An AI SDK handle.
 * @param config Supported model type config.
 * @param filepath Output model path.
 * @return int Return CVI_SUCCESS.
 */
DLL_EXPORT int CVI_AI_GetModelPath(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                   char **filepath);

/**
 * @brief Set skip vpss preprocess for supported networks.
 *
 * @param handle An AI SDK handle.
 * @param config Supported model type config.
 * @param skip To skip preprocess or not.
 * @return int Return CVI_SUCCESS if load model succeed.
 */
DLL_EXPORT int CVI_AI_SetSkipVpssPreprocess(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                            bool skip);

/**
 * @brief Get skip preprocess value for given supported model.
 *
 * @param handle An AI SDK handle.
 * @param config Supported model type config.
 * @param skip Output setting value.
 * @return int Return CVI_SUCCESS.
 */
DLL_EXPORT int CVI_AI_GetSkipVpssPreprocess(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                            bool *skip);

/**
 * @brief Set the threshold of an AI inference.
 *
 * @param handle An AI SDK handle.
 * @param config Supported model type config.
 * @param threshold Threshold in float, usually a number between 0 and 1.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT int CVI_AI_SetModelThreshold(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                        float threshold);

/**
 * @brief Get the threshold of an AI Inference
 *
 * @param handle An AI SDK handle.
 * @param config Supported model type config.
 * @param threshold Threshold in float.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT int CVI_AI_GetModelThreshold(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
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
DLL_EXPORT int CVI_AI_SetVpssThread(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
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
DLL_EXPORT int CVI_AI_SetVpssThread2(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                     const uint32_t thread, const VPSS_GRP vpssGroupId);

/**
 * @brief Get the set thread index for given supported model.
 *
 * @param handle An AI SDK handle.
 * @param config Supported model type config.
 * @param thread Output thread index.
 * @return int Return CVI_SUCCESS.
 */
DLL_EXPORT int CVI_AI_GetVpssThread(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                    uint32_t *thread);

/**
 * @brief Get the vpss group ids used by the handle.
 *
 * @param handle An AI SDK handle.
 * @param groups Return the list of used vpss group id.
 * @param num Return the length of the list.
 * @return int Return CVI_SUCCESS.
 */
DLL_EXPORT int CVI_AI_GetVpssGrpIds(cviai_handle_t handle, VPSS_GRP **groups, uint32_t *num);

/**
 * @brief Close all opened models and delete the model instances.
 *
 * @param handle An AI SDK handle.
 * @return int Return CVI_SUCCESS if succeed.
 */
DLL_EXPORT int CVI_AI_CloseAllModel(cviai_handle_t handle);

/**
 * @brief Close the chosen model and delete its model instance.
 *
 * @param handle An AI SDK handle.
 * @param config Supported model type config.
 * @return int Return CVI_SUCCESS if close succeed.
 */
DLL_EXPORT int CVI_AI_CloseModel(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config);

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
DLL_EXPORT int CVI_AI_RetinaFace(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                 cvai_face_t *faces);

/**
 * @brief Detect face with thermal images.
 *
 * @param handle An AI SDK handle.
 * @param frame Input thermal video frame.
 * @param faces Output detect result. The bbox will be given.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT int CVI_AI_ThermalFace(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
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
DLL_EXPORT int CVI_AI_FaceAttribute(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
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
DLL_EXPORT int CVI_AI_FaceAttributeOne(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                       cvai_face_t *faces, int face_idx);

/**
 * @brief Do face recognition with bbox info stored in faces.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param faces cvai_face_t structure, the cvai_face_info_t and cvai_bbox_t must be set.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT int CVI_AI_FaceRecognition(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
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
DLL_EXPORT int CVI_AI_FaceRecognitionOne(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                         cvai_face_t *faces, int face_idx);

/**
 * @brief Do face recognition with mask wearing.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param faces cvai_face_t structure, the cvai_face_info_t and cvai_bbox_t must be set.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT int CVI_AI_MaskFaceRecognition(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                          cvai_face_t *faces);

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
DLL_EXPORT int CVI_AI_FaceQuality(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
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
DLL_EXPORT int CVI_AI_Liveness(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *rgbFrame,
                               VIDEO_FRAME_INFO_S *irFrame, cvai_face_t *face,
                               cvai_liveness_ir_position_e ir_position);

/**
 * @brief Mask classification. Tells if a face is wearing a mask.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param face cvai_face_t structure, the cvai_face_info_t and cvai_bbox_t must be set.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT int CVI_AI_MaskClassification(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                         cvai_face_t *face);

/**@}*/

/**
 * \addtogroup core_od Object Detection AI Inference
 * \ingroup core_ai
 */
/**@{*/

/**
 * @brief MobileDetV2 D0 object detection, the most lightweight MobileDetV2.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param obj Output detect result. The name, bbox, and classes will be given.
 * @param det_type Specify detection type.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT int CVI_AI_MobileDetV2_D0(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
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
DLL_EXPORT int CVI_AI_MobileDetV2_D1(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
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
DLL_EXPORT int CVI_AI_MobileDetV2_D2(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
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
DLL_EXPORT int CVI_AI_Yolov3(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
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
DLL_EXPORT int CVI_AI_OSNet(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj);

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
DLL_EXPORT int CVI_AI_OSNetOne(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj,
                               int obj_idx);

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
DLL_EXPORT int CVI_AI_ESClassification(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
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
DLL_EXPORT int CVI_AI_Deepsort_Init(const cviai_handle_t handle);

/**
 * @brief Get default deepsort config.
 *
 * @param handle An AI SDK handle.
 * @param ds_conf A deepsort config.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT int CVI_AI_Deepsort_GetDefaultConfig(cvai_deepsort_config_t *ds_conf);

/**
 * @brief Set deepsort with specific config.
 *
 * @param handle An AI SDK handle.
 * @param ds_conf The specific config.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT int CVI_AI_Deepsort_SetConfig(const cviai_handle_t handle,
                                         cvai_deepsort_config_t *ds_conf);

/**
 * @brief Run deepsort track.
 *
 * @param handle An AI SDK handle.
 * @param obj Input detected object with feature.
 * @param tracker_t Output tracker results.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT int CVI_AI_Deepsort(const cviai_handle_t handle, cvai_object_t *obj,
                               cvai_tracker_t *tracker_t);

/**@}*/

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
DLL_EXPORT int CVI_AI_TamperDetection(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                      float *moving_score);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif  // End of _CVIAI_CORE_H_
