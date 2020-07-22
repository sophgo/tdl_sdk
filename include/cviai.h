#ifndef _CVIAI_H_
#define _CVIAI_H_
#include "core/cvai_core_types.h"
#include "cviai_types_free.h"
#include "face/cvai_face_helper.h"
#include "face/cvai_face_types.h"
#include "object/cvai_object_types.h"

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
  CVI_AI_SUPPORTED_MODEL_LIVENESS,
  CVI_AI_SUPPORTED_MODEL_FACEQUALITY,
  CVI_AI_SUPPORTED_MODEL_MASKCLASSIFICATION,
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
           cvai_face_info_t*: CVI_AI_FreeFaceInfo,     \
           cvai_face_t*: CVI_AI_FreeFace,              \
           cvai_object_info_t*: CVI_AI_FreeObjectInfo, \
           cvai_object_t*: CVI_AI_FreeObject)(X)
// clang-format on
#endif

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief Create a cviai_handle_t.
 *
 * @param handle An AI SDK handle.
 * @return int Return CVI_RC_SUCCESS if succeed.
 */
int CVI_AI_CreateHandle(cviai_handle_t *handle);

/**
 * @brief Destroy a cviai_handle_t.
 *
 * @param handle An AI SDK handle.
 * @return int Return CVI_RC_SUCCESS if success to destroy handle.
 */
int CVI_AI_DestroyHandle(cviai_handle_t handle);

/**
 * @brief Set the model path for supported networks.
 *
 * @param handle An AI SDK handle.
 * @param config Supported model type config.
 * @param filepath File path to the cvimodel file.
 * @return int Return CVI_RC_SUCCESS if load model succeed.
 */
int CVI_AI_SetModelPath(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                        const char *filepath);

/**
 * @brief Close all opened models and delete the model instances.
 *
 * @param handle An AI SDK handle.
 * @return int Return CVI_RC_SUCCESS if succeed.
 */
int CVI_AI_CloseAllModel(cviai_handle_t handle);

/**
 * @brief Close the chosen model and delete its model instance.
 *
 * @param handle An AI SDK handle.
 * @param config Supported model type config.
 * @return int Return CVI_RC_SUCCESS if close succeed.
 */
int CVI_AI_CloseModel(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config);

/**
 * @brief Read image from given path and return a VIDEO_FRAME_INFO_S allocated from VB block.
 *
 * @param filepath GIven image path.
 * @param blk VB block id.
 * @param frame Output read image.
 * @return int Return CVI_RC_SUCCESS if read succeed.
 */
int CVI_AI_ReadImage(const char *filepath, VB_BLK *blk, VIDEO_FRAME_INFO_S *frame);

int CVI_AI_FaceAttribute(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                         cvai_face_t *faces);
int CVI_AI_Yolov3(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *stObjDetFrame,
                  cvai_object_t *obj, cvai_obj_det_type_t det_type);
int CVI_AI_RetinaFace(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_face_t *faces,
                      int *face_count);
int CVI_AI_Liveness(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *rgbFrame,
                    VIDEO_FRAME_INFO_S *irFrame, cvai_face_t *face,
                    cvai_liveness_ir_position_e ir_position);
int CVI_AI_FaceQuality(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_face_t *face);
int CVI_AI_MaskClassification(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                              cvai_face_t *face);

#ifdef __cplusplus
}
#endif

#endif
