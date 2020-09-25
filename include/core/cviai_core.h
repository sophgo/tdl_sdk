#ifndef _CVIAI_CORE_H_
#define _CVIAI_CORE_H_
#include "core/core/cvai_core_types.h"
#include "core/cviai_rescale_bbox.h"
#include "core/cviai_types_mem.h"
#include "core/face/cvai_face_helper.h"
#include "core/face/cvai_face_types.h"
#include "core/object/cvai_object_types.h"

#include <cvi_comm_vb.h>
#include <cvi_sys.h>
typedef void *cviai_handle_t;

/**
 * @brief Supported NN model list. Can be used to config function behavior.
 *
 */
typedef enum {
  CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE,
  CVI_AI_SUPPORTED_MODEL_RETINAFACE,
  CVI_AI_SUPPORTED_MODEL_YOLOV3,
  CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D0,
  CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D1,
  CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D2,
  CVI_AI_SUPPORTED_MODEL_OSNET,
  CVI_AI_SUPPORTED_MODEL_LIVENESS,
  CVI_AI_SUPPORTED_MODEL_FACEQUALITY,
  CVI_AI_SUPPORTED_MODEL_MASKCLASSIFICATION,
  CVI_AI_SUPPORTED_MODEL_THERMALFACE,
  CVI_AI_SUPPORTED_MODEL_MASKFACERECOGNITION,
  CVI_AI_SUPPORTED_MODEL_FACERECOGNITION,
  CVI_AI_SUPPORTED_MODEL_END
} CVI_AI_SUPPORTED_MODEL_E;

/**
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

/**
 * @brief Rescale the output coordinate to original image. Only supports the following modes,
 *        CVI_AI_RescaleBBoxCenter: Padding in four directions.
 *        CVI_AI_RescaleBBoxRB: Padding in right, bottom directions.
 *        Support the following structure types written in _Generic.
 *
 * @param videoFrame Original input image.
 * @param X Input data structure.
 */
#ifdef __cplusplus
#define CVI_AI_RescaleBBoxCenter(videoFrame, X) CVI_AI_RescaleBBoxCenterCpp(videoFrame, X)
#define CVI_AI_RescaleBBoxRB(videoFrame, X) CVI_AI_RescaleBBoxRBCpp(videoFrame, X)
#else
// clang-format off
#define CVI_AI_RescaleBBoxCenter(videoFrame, X) _Generic((X), \
           cvai_face_t*: CVI_AI_RescaleBBoxCenterFace,        \
           cvai_object_t*: CVI_AI_RescaleBBoxCenterObj)(videoFrame, X)
#define CVI_AI_RescaleBBoxRB(videoFrame, X) _Generic((X),     \
           cvai_face_t*: CVI_AI_RescaleBBoxRBFace,            \
           cvai_object_t*: CVI_AI_RescaleBBoxRBObj)(videoFrame, X)
// clang-format on
#endif

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief Create a cviai_handle_t, will automatically find a vpss group id.
 *
 * @param handle An AI SDK handle.
 * @return int Return CVI_SUCCESS if succeed.
 */
int CVI_AI_CreateHandle(cviai_handle_t *handle);

/**
 * @brief Create a cviai_handle_t, need to manually assign a vpss group id.
 *
 * @param handle An AI SDK handle.
 * @param vpssGroupdId Assign a group id to cviai_handle_t.
 * @return int Return CVI_SUCCESS if succeed.
 */
int CVI_AI_CreateHandle2(cviai_handle_t *handle, const VPSS_GRP vpssGroupId);

/**
 * @brief Destroy a cviai_handle_t.
 *
 * @param handle An AI SDK handle.
 * @return int Return CVI_SUCCESS if success to destroy handle.
 */
int CVI_AI_DestroyHandle(cviai_handle_t handle);

/**
 * @brief Set the model path for supported networks.
 *
 * @param handle An AI SDK handle.
 * @param config Supported model type config.
 * @param filepath File path to the cvimodel file.
 * @return int Return CVI_SUCCESS if load model succeed.
 */
int CVI_AI_SetModelPath(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                        const char *filepath);

/**
 * @brief Get set model path from supported models.
 *
 * @param handle An AI SDK handle.
 * @param config Supported model type config.
 * @param filepath Output model path.
 * @return int Return CVI_SUCCESS.
 */
int CVI_AI_GetModelPath(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config, char **filepath);

/**
 * @brief Set skip vpss preprocess for supported networks.
 *
 * @param handle An AI SDK handle.
 * @param config Supported model type config.
 * @param skip To skip preprocess or not.
 * @return int Return CVI_SUCCESS if load model succeed.
 */
int CVI_AI_SetSkipVpssPreprocess(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config, bool skip);

/**
 * @brief Get skip preprocess value for given supported model.
 *
 * @param handle An AI SDK handle.
 * @param config Supported model type config.
 * @param skip Output setting value.
 * @return int Return CVI_SUCCESS.
 */
int CVI_AI_GetSkipVpssPreprocess(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                 bool *skip);

int CVI_AI_SetModelThreshold(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                             float threshold);
int CVI_AI_GetModelThreshold(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
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
int CVI_AI_SetVpssThread(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
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
int CVI_AI_SetVpssThread2(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                          const uint32_t thread, const VPSS_GRP vpssGroupId);

/**
 * @brief Get the set thread index for given supported model.
 *
 * @param handle An AI SDK handle.
 * @param config Supported model type config.
 * @param thread Output thread index.
 * @return int Return CVI_SUCCESS.
 */
int CVI_AI_GetVpssThread(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config, uint32_t *thread);

/**
 * @brief Get the vpss group ids used by the handle.
 *
 * @param handle An AI SDK handle.
 * @param groups Return the list of used vpss group id.
 * @param num Return the length of the list.
 * @return int Return CVI_SUCCESS.
 */
int CVI_AI_GetVpssGrpIds(cviai_handle_t handle, VPSS_GRP **groups, uint32_t *num);

/**
 * @brief Close all opened models and delete the model instances.
 *
 * @param handle An AI SDK handle.
 * @return int Return CVI_SUCCESS if succeed.
 */
int CVI_AI_CloseAllModel(cviai_handle_t handle);

/**
 * @brief Close the chosen model and delete its model instance.
 *
 * @param handle An AI SDK handle.
 * @param config Supported model type config.
 * @return int Return CVI_SUCCESS if close succeed.
 */
int CVI_AI_CloseModel(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config);

/**
 * @brief Convert given image buffer to VB frame.
 *
 * @param buffer The input image buffer.
 * @param width Input image width.
 * @param height Input image height.
 * @param stride Input image stride.
 * @param inFormat Input image buffer format.
 * @param blk VB block id.
 * @param frame Output read image.
 * @param outFormat Set output format, only supports RGB, BGR, planar.
 * @return int Return CVI_SUCCESS if read succeed.
 */
int CVI_AI_Buffer2VBFrame(const uint8_t *buffer, uint32_t width, uint32_t height, uint32_t stride,
                          const PIXEL_FORMAT_E inFormat, VB_BLK *blk, VIDEO_FRAME_INFO_S *frame,
                          const PIXEL_FORMAT_E outFormat);

/**
 * @brief Read image from given path and return a VIDEO_FRAME_INFO_S allocated from VB block.
 *
 * @param filepath GIven image path.
 * @param blk VB block id.
 * @param frame Output read image.
 * @param format Set output format, only supports RGB, BGR, planar.
 * @return int Return CVI_SUCCESS if read succeed.
 */
int CVI_AI_ReadImage(const char *filepath, VB_BLK *blk, VIDEO_FRAME_INFO_S *frame,
                     const PIXEL_FORMAT_E format);

int CVI_AI_FaceAttribute(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                         cvai_face_t *faces);
int CVI_AI_FaceAttributeOne(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                            cvai_face_t *faces, int face_idx);
int CVI_AI_FaceRecognition(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                           cvai_face_t *faces);
int CVI_AI_FaceRecognitionOne(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                              cvai_face_t *faces, int face_idx);
int CVI_AI_Yolov3(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj,
                  cvai_obj_det_type_t det_type);
int CVI_AI_RetinaFace(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_face_t *faces,
                      int *face_count);
int CVI_AI_Liveness(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *rgbFrame,
                    VIDEO_FRAME_INFO_S *irFrame, cvai_face_t *face,
                    cvai_liveness_ir_position_e ir_position);
int CVI_AI_FaceQuality(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_face_t *face);
int CVI_AI_MaskClassification(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                              cvai_face_t *face);
int CVI_AI_ThermalFace(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_face_t *faces);
int CVI_AI_MobileDetV2_D0(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj,
                          cvai_obj_det_type_t det_type);
int CVI_AI_MobileDetV2_D1(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj,
                          cvai_obj_det_type_t det_type);
int CVI_AI_MobileDetV2_D2(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj,
                          cvai_obj_det_type_t det_type);
int CVI_AI_OSNetOne(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj,
                    int obj_idx);
int CVI_AI_OSNet(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj);
int CVI_AI_MaskFaceRecognition(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                               cvai_face_t *faces);
int CVI_AI_TamperDetection(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                           float *moving_score);
int CVI_AI_Deepsort(const cviai_handle_t handle, cvai_object_t *obj, cvai_tracker_t *tracker_t);

#ifdef __cplusplus
}
#endif

#endif  // End of _CVIAI_CORE_H_
