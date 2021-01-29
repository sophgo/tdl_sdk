#ifndef _CVIAI_EVALUATION_H_
#define _CVIAI_EVALUATION_H_

#include "core/face/cvai_face_types.h"
#include "core/object/cvai_object_types.h"

#include <cvi_sys.h>

typedef void *cviai_eval_handle_t;

#ifdef __cplusplus
extern "C" {
#endif

DLL_EXPORT CVI_S32 CVI_AI_Eval_CreateHandle(cviai_eval_handle_t *handle);

DLL_EXPORT CVI_S32 CVI_AI_Eval_DestroyHandle(cviai_eval_handle_t handle);

/****************************************************************
 * Cityscapes evaluation functions
 **/
DLL_EXPORT CVI_S32 CVI_AI_Eval_CityscapesInit(cviai_eval_handle_t handle, const char *image_dir,
                                              const char *output_dir);

DLL_EXPORT CVI_S32 CVI_AI_Eval_CityscapesGetImage(cviai_eval_handle_t handle, const uint32_t index,
                                                  char **fileName);

DLL_EXPORT CVI_S32 CVI_AI_Eval_CityscapesGetImageNum(cviai_eval_handle_t handle, uint32_t *num);

DLL_EXPORT CVI_S32 CVI_AI_Eval_CityscapesWriteResult(cviai_eval_handle_t handle,
                                                     VIDEO_FRAME_INFO_S *label_frame,
                                                     const int index);

/****************************************************************
 * Coco evaluation functions
 **/
DLL_EXPORT CVI_S32 CVI_AI_Eval_CocoInit(cviai_eval_handle_t handle, const char *pathPrefix,
                                        const char *jsonPath, uint32_t *imageNum);

DLL_EXPORT CVI_S32 CVI_AI_Eval_CocoGetImageIdPair(cviai_eval_handle_t handle, const uint32_t index,
                                                  char **filepath, int *id);

DLL_EXPORT CVI_S32 CVI_AI_Eval_CocoInsertObject(cviai_eval_handle_t handle, const int id,
                                                cvai_object_t *obj);

DLL_EXPORT CVI_S32 CVI_AI_Eval_CocoStartEval(cviai_eval_handle_t handle, const char *filepath);

DLL_EXPORT CVI_S32 CVI_AI_Eval_CocoEndEval(cviai_eval_handle_t handle);

/****************************************************************
 * LFW evaluation functions
 **/
DLL_EXPORT CVI_S32 CVI_AI_Eval_LfwInit(cviai_eval_handle_t handle, const char *filepath,
                                       bool label_pos_first, uint32_t *imageNum);

DLL_EXPORT CVI_S32 CVI_AI_Eval_LfwGetImageLabelPair(cviai_eval_handle_t handle,
                                                    const uint32_t index, char **filepath,
                                                    char **filepath2, int *label);

DLL_EXPORT CVI_S32 CVI_AI_Eval_LfwInsertFace(cviai_eval_handle_t handle, const int index,
                                             const int label, const cvai_face_t *face1,
                                             const cvai_face_t *face2);

DLL_EXPORT CVI_S32 CVI_AI_Eval_LfwInsertLabelScore(cviai_eval_handle_t handle, const int index,
                                                   const int label, const float score);

DLL_EXPORT CVI_S32 CVI_AI_Eval_LfwSave2File(cviai_eval_handle_t handle, const char *filepath);

DLL_EXPORT CVI_S32 CVI_AI_Eval_LfwClearInput(cviai_eval_handle_t handle);

DLL_EXPORT CVI_S32 CVI_AI_Eval_LfwClearEvalData(cviai_eval_handle_t handle);

/****************************************************************
 * Wider Face evaluation functions
 **/
DLL_EXPORT CVI_S32 CVI_AI_Eval_WiderFaceInit(cviai_eval_handle_t handle, const char *datasetDir,
                                             const char *resultDir, uint32_t *imageNum);

DLL_EXPORT CVI_S32 CVI_AI_Eval_WiderFaceGetImagePath(cviai_eval_handle_t handle,
                                                     const uint32_t index, char **filepath);

DLL_EXPORT CVI_S32 CVI_AI_Eval_WiderFaceResultSave2File(cviai_eval_handle_t handle, const int index,
                                                        const VIDEO_FRAME_INFO_S *frame,
                                                        const cvai_face_t *face);

DLL_EXPORT CVI_S32 CVI_AI_Eval_WiderFaceClearInput(cviai_eval_handle_t handle);

/****************************************************************
 * Market1501 evaluation functions
 **/
DLL_EXPORT CVI_S32 CVI_AI_Eval_Market1501Init(cviai_eval_handle_t handle, const char *filepath);
DLL_EXPORT CVI_S32 CVI_AI_Eval_Market1501GetImageNum(cviai_eval_handle_t handle, bool is_query,
                                                     uint32_t *num);
DLL_EXPORT CVI_S32 CVI_AI_Eval_Market1501GetPathIdPair(cviai_eval_handle_t handle,
                                                       const uint32_t index, bool is_query,
                                                       char **filepath, int *cam_id, int *pid);
DLL_EXPORT CVI_S32 CVI_AI_Eval_Market1501InsertFeature(cviai_eval_handle_t handle, const int index,
                                                       bool is_query,
                                                       const cvai_feature_t *feature);
DLL_EXPORT CVI_S32 CVI_AI_Eval_Market1501EvalCMC(cviai_eval_handle_t handle);

/****************************************************************
 * WLFW evaluation functions
 **/
DLL_EXPORT CVI_S32 CVI_AI_Eval_WflwInit(cviai_eval_handle_t handle, const char *filepath,
                                        uint32_t *imageNum);
DLL_EXPORT CVI_S32 CVI_AI_Eval_WflwGetImage(cviai_eval_handle_t handle, const uint32_t index,
                                            char **fileName);
DLL_EXPORT CVI_S32 CVI_AI_Eval_WflwInsertPoints(cviai_eval_handle_t handle, const int index,
                                                const cvai_pts_t points, const int width,
                                                const int height);
DLL_EXPORT CVI_S32 CVI_AI_Eval_WflwDistance(cviai_eval_handle_t handle);

/****************************************************************
 * LPDR evaluation functions
 **/
DLL_EXPORT CVI_S32 CVI_AI_Eval_LPDRInit(cviai_eval_handle_t handle, const char *pathPrefix,
                                        const char *jsonPath, uint32_t *imageNum);

DLL_EXPORT CVI_S32 CVI_AI_Eval_LPDRGetImageIdPair(cviai_eval_handle_t handle, const uint32_t index,
                                                  char **filepath, int *id);

#ifdef __cplusplus
}
#endif

#endif  // End of _CVIAI_EVALUATION_H_