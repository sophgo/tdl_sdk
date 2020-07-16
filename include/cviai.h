#ifndef _CVIAI_H_
#define _CVIAI_H_
#include "core/cvi_core_types.h"
#include "face/cvi_face_helper.h"
#include "face/cvi_face_types.h"
#include "object/cvi_object_types.h"

#include "cvi_sys.h"
#include "cviai_types_free.h"
typedef void *cviai_handle_t;

typedef enum {
  CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE,
  CVI_AI_SUPPORTED_MODEL_RETINAFACE,
  CVI_AI_SUPPORTED_MODEL_YOLOV3,
  CVI_AI_SUPPORTED_MODEL_END
} CVI_AI_SUPPORTED_MODEL_E;

#ifdef __cplusplus
#define CVI_AI_Free(X) CVI_AI_FreeCpp(X)
#else
// clang-format off
#define CVI_AI_Free(X) _Generic((X),                 \
           cvi_feature_t*: CVI_AI_FreeFeature,        \
           cvi_pts_t*: CVI_AI_FreePts,                \
           cvi_face_info_t*: CVI_AI_FreeFaceInfo,     \
           cvi_face_t*: CVI_AI_FreeFace,              \
           cvi_object_info_t*: CVI_AI_FreeObjectInfo, \
           cvi_object_t*: CVI_AI_FreeObject)(X)
// clang-format on
#endif

#ifdef __cplusplus
extern "C" {
#endif
int CVI_AI_CreateHandle(cviai_handle_t *handle);
int CVI_AI_DestroyHandle(cviai_handle_t handle);
int CVI_AI_SetModelPath(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                        const char *filepath);
int CVI_AI_CloseAllModel(cviai_handle_t handle);
int CVI_AI_CloseModel(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config);
int CVI_AI_FaceAttribute(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvi_face_t *faces);
int CVI_AI_Yolov3(cviai_handle_t handle, VIDEO_FRAME_INFO_S *stObjDetFrame, cvi_object_t *obj,
                  cvi_obj_det_type_t det_type);
int CVI_AI_RetinaFace(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvi_face_t *faces,
                      int *face_count);

#ifdef __cplusplus
}
#endif

#endif
