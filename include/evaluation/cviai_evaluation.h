#ifndef _CVIAI_EVALUATION_H_
#define _CVIAI_EVALUATION_H_

#include "core/face/cvai_face_types.h"
#include "core/object/cvai_object_types.h"

#include <cvi_sys.h>

typedef void *cviai_eval_handle_t;

#ifdef __cplusplus
extern "C" {
#endif

CVI_S32 CVI_AI_Eval_CreateHandle(cviai_eval_handle_t *handle);

CVI_S32 CVI_AI_Eval_DestroyHandle(cviai_eval_handle_t handle);

CVI_S32 CVI_AI_Eval_CocoInit(cviai_eval_handle_t handle, const char *pathPrefix,
                             const char *jsonPath, uint32_t *imageNum);

CVI_S32 CVI_AI_Eval_CocoGetImageIdPair(cviai_eval_handle_t handle, const uint32_t index,
                                       char **filepath, int *id);

CVI_S32 CVI_AI_Eval_CocoInsertObject(cviai_eval_handle_t handle, const int id, cvai_object_t *obj);

CVI_S32 CVI_AI_Eval_CocoSave2Json(cviai_eval_handle_t handle, const char *filepath);

CVI_S32 CVI_AI_Eval_CocoClearInput(cviai_eval_handle_t handle);

CVI_S32 CVI_AI_Eval_CocoClearObject(cviai_eval_handle_t handle);

CVI_S32 CVI_AI_Eval_LfwInit(cviai_eval_handle_t handle, const char *filepath, uint32_t *imageNum);

CVI_S32 CVI_AI_Eval_LfwGetImageLabelPair(cviai_eval_handle_t handle, const uint32_t index,
                                         char **filepath, char **filepath2, int *label);

CVI_S32 CVI_AI_Eval_LfwInsertFace(cviai_eval_handle_t handle, const int index, const int label,
                                  const cvai_face_t *face1, const cvai_face_t *face2);

CVI_S32 CVI_AI_Eval_LfwSave2File(cviai_eval_handle_t handle, const char *filepath);

CVI_S32 CVI_AI_Eval_LfwClearInput(cviai_eval_handle_t handle);

CVI_S32 CVI_AI_Eval_LfwClearEvalData(cviai_eval_handle_t handle);

CVI_S32 CVI_AI_Eval_WiderFaceInit(cviai_eval_handle_t handle, const char *datasetDir,
                                  const char *resultDir, uint32_t *imageNum);

CVI_S32 CVI_AI_Eval_WiderFaceGetImagePath(cviai_eval_handle_t handle, const uint32_t index,
                                          char **filepath);

CVI_S32 CVI_AI_Eval_WiderFaceResultSave2File(cviai_eval_handle_t handle, const int index,
                                             const VIDEO_FRAME_INFO_S *frame,
                                             const cvai_face_t *face);

CVI_S32 CVI_AI_Eval_WiderFaceClearInput(cviai_eval_handle_t handle);

#ifdef __cplusplus
}
#endif

#endif  // End of _CVIAI_EVALUATION_H_