#ifndef _CVIAI_CORE_H_
#define _CVIAI_CORE_H_
#include "core/core/cvai_core_types.h"
#include "core/core/cvai_errno.h"
#include "core/core/cvai_vpss_types.h"
#include "core/cviai_rescale_bbox.h"
#include "core/cviai_types_mem.h"
#include "core/deepsort/cvai_deepsort_types.h"
#include "core/face/cvai_face_helper.h"
#include "core/face/cvai_face_types.h"
#include "core/object/cvai_object_types.h"

#include <cvi_comm_vb.h>
#include <cvi_sys.h>
#include "cvi_comm.h"

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
           cvai_object_t*: CVI_AI_FreeObject,          \
           cvai_image_t*: CVI_AI_FreeImage)(X)
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
           cvai_object_info_t*: CVI_AI_CopyObjectInfo,              \
           cvai_image_t*: CVI_AI_CopyImage)
#define CVI_AI_CopyInfo(IN, OUT) _Generic((IN),                     \
           cvai_face_info_t*: CVI_AI_CopyInfoG(OUT),                \
           cvai_object_info_t*: CVI_AI_CopyInfoG(OUT),              \
           cvai_image_t*: CVI_AI_CopyInfoG(OUT))((IN), (OUT))
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
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_RETINAFACE_IR)                    \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_RETINAFACE_HARDHAT)               \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_THERMALFACE)                      \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_THERMALPERSON)                    \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE)                    \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_FACERECOGNITION)                  \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_MASKFACERECOGNITION)              \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_FACEQUALITY)                      \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_LIVENESS)                         \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_MASKCLASSIFICATION)               \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE)       \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE)              \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN)           \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PERSON_PETS)          \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_COCO80)               \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_YOLOV3)                           \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_YOLOX)                           \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_OSNET)                            \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_SOUNDCLASSIFICATION)              \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_WPODNET)                          \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_LPRNET_TW)                        \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_LPRNET_CN)                        \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_DEEPLABV3)                        \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_ALPHAPOSE)                        \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_EYECLASSIFICATION)                \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_YAWNCLASSIFICATION)               \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_FACELANDMARKER)                   \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_INCAROBJECTDETECTION)             \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_SMOKECLASSIFICATION)              \
  CVI_AI_NAME_WRAP(CVI_AI_SUPPORTED_MODEL_FACEMASKDETECTION)

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
  uint32_t length = sizeof(model_names) / sizeof(model_names[0]);
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
 * @return int Return CVIAI_SUCCESS if succeed.
 */
DLL_EXPORT CVI_S32 CVI_AI_CreateHandle(cviai_handle_t *handle);

/**
 * @brief Create a cviai_handle_t, need to manually assign a vpss group id.
 *
 * @param handle An AI SDK handle.
 * @param vpssGroupId Assign a group id to cviai_handle_t.
 * @param vpssDev Assign a device id to cviai_handle_t.
 * @return int Return CVIAI_SUCCESS if succeed.
 */
DLL_EXPORT CVI_S32 CVI_AI_CreateHandle2(cviai_handle_t *handle, const VPSS_GRP vpssGroupId,
                                        const CVI_U8 vpssDev);

/**
 * @brief Destroy a cviai_handle_t.
 *
 * @param handle An AI SDK handle.
 * @return int Return CVIAI_SUCCESS if success to destroy handle.
 */
DLL_EXPORT CVI_S32 CVI_AI_DestroyHandle(cviai_handle_t handle);

/**
 * @brief Open model with given file path.
 *
 * @param handle An AI SDK handle.
 * @param model Supported model id.
 * @param filepath File path to the cvimodel file.
 * @return int Return CVIAI_SUCCESS if load model succeed.
 */
DLL_EXPORT CVI_S32 CVI_AI_OpenModel(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model,
                                    const char *filepath);

/**
 * @brief Get set model path from supported models.
 *
 * @param handle An AI SDK handle.
 * @param model Supported model id.
 * @return model path.
 */
DLL_EXPORT const char *CVI_AI_GetModelPath(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model);

/**
 * @brief Set skip vpss preprocess for supported networks.
 *
 * @param handle An AI SDK handle.
 * @param model Supported model id.
 * @param skip To skip preprocess or not.
 * @return int Return CVIAI_SUCCESS if load model succeed.
 */
DLL_EXPORT CVI_S32 CVI_AI_SetSkipVpssPreprocess(cviai_handle_t handle,
                                                CVI_AI_SUPPORTED_MODEL_E model, bool skip);

/**
 * @brief Set list depth for VPSS.
 *
 * @param handle An AI SDK handle.
 * @param model Supported model id.
 * @param input_id input index of model.
 * @param depth list depth of VPSS.
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_SetVpssDepth(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model,
                                       uint32_t input_id, uint32_t depth);

/**
 * @brief Get list depth for VPSS.
 *
 * @param handle An AI SDK handle.
 * @param model Supported model id.
 * @param input_id input index of model.
 * @param depth list depth of VPSS.
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_GetVpssDepth(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model,
                                       uint32_t input_id, uint32_t *depth);

/**
 * @brief Get skip preprocess value for given supported model.
 *
 * @param handle An AI SDK handle.
 * @param model Supported model id.
 * @param skip Output setting value.
 * @return int Return CVIAI_SUCCESS.
 */
DLL_EXPORT CVI_S32 CVI_AI_GetSkipVpssPreprocess(cviai_handle_t handle,
                                                CVI_AI_SUPPORTED_MODEL_E model, bool *skip);

/**
 * @brief Set the threshold of an AI inference.
 *
 * @param handle An AI SDK handle.
 * @param model Supported model id.
 * @param threshold Threshold in float, usually a number between 0 and 1.
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_SetModelThreshold(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model,
                                            float threshold);

/**
 * @brief Get the threshold of an AI Inference
 *
 * @param handle An AI SDK handle.
 * @param model Supported model id.
 * @param threshold Threshold in float.
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_GetModelThreshold(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model,
                                            float *threshold);
/**
 * @brief Set different vpss thread for each model. Vpss group id is not thread safe. We recommended
 * to change a thread if the process is not sequential.
 *
 * @param handle An AI SDK handle.
 * @param model Supported model id.
 * @param thread The vpss thread index user desired. Note this param will changed if previous index
 * is not used.
 * @return int Return CVIAI_SUCCESS if successfully changed.
 */
DLL_EXPORT CVI_S32 CVI_AI_SetVpssThread(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model,
                                        const uint32_t thread);

/**
 * @brief Set different vpss thread for each model. Vpss group id is not thread safe. We recommended
 * to change a thread if the process is not sequential. This function requires manually assigning a
 * vpss group id and device id.
 *
 * @param handle An AI SDK handle.
 * @param model Supported model id.
 * @param thread The vpss thread index user desired. Note this param will changed if previous index
 * is not used.
 * @param vpssGroupId Assign a vpss group id if a new vpss instance needs to be created.
 * @param dev Assign Vpss device id to Vpss group
 * @return int Return CVIAI_SUCCESS if successfully changed.
 */
DLL_EXPORT CVI_S32 CVI_AI_SetVpssThread2(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model,
                                         const uint32_t thread, const VPSS_GRP vpssGroupId,
                                         const CVI_U8 dev);

/**
 * @brief Get the set thread index for given supported model.
 *
 * @param handle An AI SDK handle.
 * @param model Supported model id.
 * @param thread Output thread index.
 * @return int Return CVIAI_SUCCESS.
 */
DLL_EXPORT CVI_S32 CVI_AI_GetVpssThread(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model,
                                        uint32_t *thread);

/**
 * @brief Set VB pool id to VPSS in AISDK. By default, VPSS in AISDK acquires VB from all
 * system-wide VB_POOLs which are created via CVI_VB_Init. In this situation, system decides which
 * VB_POOL is used according to VB allocation mechanism. The size of aquired VB maybe not optimal
 * and it could cause resource competition. To prevents this problem, you can assign a specific
 * VB_POOL to AISDK via this function. The VB_POOL created by CVI_VB_Init or CVI_VB_CreatePool are
 * both accepted.
 *
 * @param handle An AI SDK handle.
 * @param thread VPSS thread index.
 * @param pool_id vb pool id. if pool id is VB_INVALID_POOLID than VPSS will get VB from all
 * system-wide VB_POOLs like default.
 * @return int Return CVIAI_SUCCESS.
 */
DLL_EXPORT CVI_S32 CVI_AI_SetVBPool(cviai_handle_t handle, uint32_t thread, VB_POOL pool_id);

/**
 * @brief Get VB pool id used by internal VPSS.
 *
 * @param handle An AI SDK handle.
 * @param thread VPSS thread index.
 * @param pool_id Output vb pool id.
 * @return int Return CVIAI_SUCCESS.
 */
DLL_EXPORT CVI_S32 CVI_AI_GetVBPool(cviai_handle_t handle, uint32_t thread, VB_POOL *pool_id);

/**
 * @brief Get the vpss group ids used by the handle.
 *
 * @param handle An AI SDK handle.
 * @param groups Return the list of used vpss group id.
 * @param num Return the length of the list.
 * @return int Return CVIAI_SUCCESS.
 */
DLL_EXPORT CVI_S32 CVI_AI_GetVpssGrpIds(cviai_handle_t handle, VPSS_GRP **groups, uint32_t *num);

/**
 * @brief Set VPSS waiting time.
 *
 * @param handle An AI SDK handle.
 * @param timeout Timeout value.
 * @return int Return CVIAI_SUCCESS.
 */
DLL_EXPORT CVI_S32 CVI_AI_SetVpssTimeout(cviai_handle_t handle, uint32_t timeout);

/**
 * @brief Close all opened models and delete the model instances.
 *
 * @param handle An AI SDK handle.
 * @return int Return CVIAI_SUCCESS if succeed.
 */
DLL_EXPORT CVI_S32 CVI_AI_CloseAllModel(cviai_handle_t handle);

/**
 * @brief Close the chosen model and delete its model instance.
 *
 * @param handle An AI SDK handle.
 * @param model Supported model id.
 * @return int Return CVIAI_SUCCESS if close succeed.
 */
DLL_EXPORT CVI_S32 CVI_AI_CloseModel(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model);

/**
 * @brief Export vpss channel attribute.
 *
 * @param handle An AI SDK handle.
 * @param model Supported model id.
 * @param frameWidth The input frame width.
 * @param frameHeight The input frame height.
 * @param idx The index of the input tensors.
 * @param chnAttr Exported VPSS channel config settings.
 * @return int Return CVIAI_SUCCESS on success, CVIAI_FAILURE if exporting not supported.
 */
DLL_EXPORT CVI_S32 CVI_AI_GetVpssChnConfig(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model,
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
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_RetinaFace(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                     cvai_face_t *faces);

/**
 * @brief RetinaFaceIR face detection.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param faces Output detect result. The name, bbox, and face points will be given.
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_RetinaFace_IR(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                        cvai_face_t *faces);
/**
 * @brief RetinaFace hardhat face detection.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param faces Output detect result. The name, bbox, and face points will be given.
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_RetinaFace_Hardhat(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                             cvai_face_t *faces);

/**
 * @brief Detect face with thermal images.
 *
 * @param handle An AI SDK handle.
 * @param frame Input thermal video frame.
 * @param faces Output detect result. The bbox will be given.
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_ThermalFace(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                      cvai_face_t *faces);

/**
 * @brief Detect person with thermal images.
 *
 * @param handle An AI SDK handle.
 * @param frame Input thermal video frame.
 * @param obj Output detect result. The bbox will be given.
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_ThermalPerson(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                        cvai_object_t *obj);

/**
 * @brief Detect face with mask score.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param face_meta Output detect result. The bbox will be given.
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_FaceMaskDetection(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                            cvai_face_t *face_meta);
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
 * @return int Return CVIAI_SUCCESS on success.
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
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_FaceAttributeOne(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                           cvai_face_t *faces, int face_idx);

/**
 * @brief Do face recognition with bbox info stored in faces.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param faces cvai_face_t structure, the cvai_face_info_t and cvai_bbox_t must be set.
 * @return int Return CVIAI_SUCCESS on success.
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
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_FaceRecognitionOne(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                             cvai_face_t *faces, int face_idx);

/**
 * @brief Do face recognition with mask wearing.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param faces cvai_face_t structure, the cvai_face_info_t and cvai_bbox_t must be set.
 * @return int Return CVIAI_SUCCESS on success.
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
 * @brief FaceQuality. Assess the quality of the faces.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param face cvai_face_t structure, the cvai_face_info_t and cvai_bbox_t must be set.
 * @param skip bool array, whether skip quailty assessment at corresponding index (NULL for running
 * without skip)
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_FaceQuality(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                      cvai_face_t *face, bool *skip);

/**
 * @brief Crop image in given frame.
 *
 * @param srcFrame Input frame. (only support RGB Packed format)
 * @param dst Output image.
 * @param bbox The bounding box.
 * @param cvtRGB888 convert to RGB888 format.
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_CropImage(VIDEO_FRAME_INFO_S *srcFrame, cvai_image_t *dst,
                                    cvai_bbox_t *bbox, bool cvtRGB888);

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
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_CropImage_Exten(VIDEO_FRAME_INFO_S *srcFrame, cvai_image_t *dst,
                                          cvai_bbox_t *bbox, bool cvtRGB888, float exten_ratio,
                                          float *offset_x, float *offset_y);

/**
 * @brief Crop face image in given frame.
 *
 * @param srcFrame Input frame. (only support RGB Packed format)
 * @param dst Output image.
 * @param face_info Face information, contain bbox and 5 landmark.
 * @param align Align face to standard size if true.
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_CropImage_Face(VIDEO_FRAME_INFO_S *srcFrame, cvai_image_t *dst,
                                         cvai_face_info_t *face_info, bool align, bool cvtRGB888);

/**
 * @brief Liveness. Gives a score to present how real the face is. The score will be low if the face
 * is not directly taken by a camera.
 *
 * @param handle An AI SDK handle.
 * @param rgbFrame Input RGB video frame.
 * @param irFrame Input IR video frame.
 * @param face cvai_face_t structure, the cvai_face_info_t and cvai_bbox_t must be set.
 * @param ir_position The position relationship netween the ir and the rgb camera.
 * @return int Return CVIAI_SUCCESS on success.
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
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_MaskClassification(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                             cvai_face_t *face);

/**@}*/

/**
 * \addtogroup core_od Object Detection AI Inference
 * \ingroup core_ai
 */
/**@{*/

/**
 * @brief Select classes for detection model. Model will output objects belong to these classes.
 * Currently only support MobileDetV2 family.
 *
 * @param handle An AI SDK handle.
 * @param model model id.
 * @param num_classes number of classes you want to select.
 * @param ... class indexs
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_SelectDetectClass(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model,
                                            uint32_t num_classes, ...);

/**
 * @brief MobileDetV2 Vehicle object detectior, which can be used to detect "car", "truck", and
 * "motorbike" classes.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param obj Output detect result. The name, bbox, and classes will be given.
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_MobileDetV2_Vehicle(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                              cvai_object_t *obj);

/**
 * @brief MobileDetV2 pedestrian object detector, which can be used to detect "person" class
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param obj Output detect result. The name, bbox, and classes will be given.
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_MobileDetV2_Pedestrian(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                                 cvai_object_t *obj);

/**
 * @brief MobileDetV2 object detector, which can be used to detect "person", "bicycle", "car",
 * "motorbike", "bus", and "truck" classes.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param obj Output detect result. The name, bbox, and classes will be given.
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_MobileDetV2_Person_Vehicle(cviai_handle_t handle,
                                                     VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj);

/**
 * @brief MobileDetV2 object detector, which can be used to detect "person", "cat", and "dog"
 * classes.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param obj Output detect result. The name, bbox, and classes will be given.
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_MobileDetV2_Person_Pets(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                                  cvai_object_t *obj);

/**
 * @brief MobileDetV2 object detector which can be used to detect coco 80 classes objects.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param obj Output detect result. The name, bbox, and classes will be given.
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_MobileDetV2_COCO80(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                             cvai_object_t *obj);

/**
 * @brief Yolov3 object detection.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param obj Output detect result. The name, bbox, and classes will be given.
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_Yolov3(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                 cvai_object_t *obj);

/**
 * @brief YoloX object detection.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param obj Output detect result. The name, bbox, and classes will be given.
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_YoloX(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                cvai_object_t *obj);

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
 * @return int Return CVIAI_SUCCESS on success.
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
 * @return int Return CVIAI_SUCCESS on success.
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
 * @brief Do sound classification.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param index The index of sound classes.
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_SoundClassification(const cviai_handle_t handle,
                                              VIDEO_FRAME_INFO_S *frame, int *index);

/**
 * @brief Get sound classification classes num.
 *
 * @param handle An AI SDK handle.
 * @return int Return CVIAI_SUCCESS on success.
 */

DLL_EXPORT CVI_S32 CVI_AI_Get_SoundClassification_ClassesNum(const cviai_handle_t handle);

/**
 * @brief Set sound classification threshold.
 *
 * @param handle An AI SDK handle.
 * @param th Sound classifiction threshold
 * @return int Return CVIAI_SUCCESS on success.
 */

DLL_EXPORT CVI_S32 CVI_AI_Set_SoundClassification_Threshold(const cviai_handle_t handle,
                                                            const float th);

/**@}*/

/**
 * \addtogroup core_tracker Tracker
 * \ingroup core_ai
 */
/**@{*/

/**
 * @brief Initialize DeepSORT.
 *
 * @param handle An AI SDK handle.
 * @param use_specific_counter true for using individual id counter for each class
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_DeepSORT_Init(const cviai_handle_t handle, bool use_specific_counter);

/**
 * @brief Get default DeepSORT config.
 *
 * @param ds_conf Output config.
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_DeepSORT_GetDefaultConfig(cvai_deepsort_config_t *ds_conf);

/**
 * @brief Get DeepSORT config.
 *
 * @param handle An AI SDK handle.
 * @param ds_conf Output config.
 * @param cviai_obj_type The specific class type (-1 for setting default config).
 * @return int Return CVI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_DeepSORT_GetConfig(const cviai_handle_t handle,
                                             cvai_deepsort_config_t *ds_conf, int cviai_obj_type);

/**
 * @brief Set DeepSORT with specific config.
 *
 * @param handle An AI SDK handle.
 * @param ds_conf The specific config.
 * @param cviai_obj_type The specific class type (-1 for setting default config).
 * @param show_config show detail information or not.
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_DeepSORT_SetConfig(const cviai_handle_t handle,
                                             cvai_deepsort_config_t *ds_conf, int cviai_obj_type,
                                             bool show_config);

/**
 * @brief clean DeepSORT ID counter.
 *
 * @param handle An AI SDK handle.
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_DeepSORT_CleanCounter(const cviai_handle_t handle);

/**
 * @brief Run DeepSORT/SORT track for object.
 *
 * @param handle An AI SDK handle.
 * @param obj Input detected object with feature.
 * @param tracker_t Output tracker results.
 * @param use_reid If true, track by DeepSORT algorithm, else SORT.
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_DeepSORT_Obj(const cviai_handle_t handle, cvai_object_t *obj,
                                       cvai_tracker_t *tracker, bool use_reid);

/**
 * @brief Run SORT track for face.
 *
 * @param handle An AI SDK handle.
 * @param face Input detected face with feature.
 * @param tracker_t Output tracker results.
 * @param use_reid Set false for SORT.
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_DeepSORT_Face(const cviai_handle_t handle, cvai_face_t *face,
                                        cvai_tracker_t *tracker, bool use_reid);

DLL_EXPORT CVI_S32 CVI_AI_DeepSORT_DebugInfo_1(const cviai_handle_t handle, char *debug_info);

DLL_EXPORT CVI_S32 CVI_AI_DeepSORT_GetTracker_Inactive(const cviai_handle_t handle,
                                                       cvai_tracker_t *tracker);

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
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_DeeplabV3(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                    VIDEO_FRAME_INFO_S *out_frame, cvai_class_filter_t *filter);
/**@}*/

/**
 * @brief LicensePlateRecognition(TW/CN).
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param vehicle License plate object info
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_LicensePlateRecognition_TW(const cviai_handle_t handle,
                                                     VIDEO_FRAME_INFO_S *frame,
                                                     cvai_object_t *vehicle);
DLL_EXPORT CVI_S32 CVI_AI_LicensePlateRecognition_CN(const cviai_handle_t handle,
                                                     VIDEO_FRAME_INFO_S *frame,
                                                     cvai_object_t *vehicle);
/**@}*/

/**
 * @brief LicensePlateDetection.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param vehicle_meta Vehicle object info
 * @return int Return CVIAI_SUCCESS on success.
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
 * @return int Return CVIAI_SUCCESS on success.
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
 * @return int Return CVIAI_SUCCESS on success.
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
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_TamperDetection(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                          float *moving_score);

/**
 * @brief Set background frame for motion detection.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame, should be YUV420 format.
 * be returned.
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_Set_MotionDetection_Background(const cviai_handle_t handle,
                                                         VIDEO_FRAME_INFO_S *frame);

/**
 * @brief Do Motion Detection with background subtraction method.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame, should be YUV420 format.
 * @param objects Detected object info
 * @param threshold Threshold of motion detection, the range between 0 and 255.
 * @param min_area Minimal pixel area. The bounding box whose area is larger than this value would
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_MotionDetection(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                          cvai_object_t *objects, uint8_t threshold,
                                          double min_area);

/**@}*/

/**
 * \addtogroup core_dms Driving Monitor System
 * \ingroup core_ai
 */
/**@{*/

/**
 * @brief Do eye classification.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param cvai_face_t structure. Calculate the eye_score in cvai_dms_t.
 * @return int Return CVIAI_SUCCESS on success.
 */

DLL_EXPORT CVI_S32 CVI_AI_EyeClassification(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                            cvai_face_t *face);

/**
 * @brief Do yawn classification.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param cvai_face_t structure. Calculate the yawn_score in cvai_dms_t.
 * @return int Return CVIAI_SUCCESS on success.
 */

DLL_EXPORT CVI_S32 CVI_AI_YawnClassification(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                             cvai_face_t *face);
/**
 * @brief Do face landmark.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param cvai_face_t structure. Calculate the landmarks in cvai_dms_t.
 * @return int Return CVIAI_SUCCESS on success.
 */

DLL_EXPORT CVI_S32 CVI_AI_IncarObjectDetection(const cviai_handle_t handle,
                                               VIDEO_FRAME_INFO_S *frame, cvai_face_t *face);

/**@}*/

/**
 * \addtogroup core_face_landmark Face Landmark
 * \ingroup core_ai
 */
/**@{*/

DLL_EXPORT CVI_S32 CVI_AI_FaceLandmarker(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                         cvai_face_t *face);

/**
 * @brief Do smoke classification.
 *
 * @param handle An AI SDK handle.
 * @param frame Input video frame.
 * @param cvai_face_t structure. Calculate the smoke_score in cvai_dms_t.
 * @return int Return CVIAI_SUCCESS on success.
 */

DLL_EXPORT CVI_S32 CVI_AI_SmokeClassification(const cviai_handle_t handle,
                                              VIDEO_FRAME_INFO_S *frame, cvai_face_t *face);

/**
 * @brief Dump model input frame to npz.
 *
 * @param handle An AI SDK handle.
 * @param model Model id.
 * @param dump_path Output path.
 * @param enable Whether enable or not.
 * @return int Return CVIAI_SUCCESS on success.
 */
/**@}*/
DLL_EXPORT CVI_S32 CVI_AI_EnalbeDumpInput(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model,
                                          const char *dump_path, bool enable);

#ifdef __cplusplus
}
#endif

#endif  // End of _CVIAI_CORE_H_
