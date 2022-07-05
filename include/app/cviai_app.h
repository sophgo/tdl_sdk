#ifndef _CVIAI_APP_H_
#define _CVIAI_APP_H_
#include "capture/face_capture_type.h"
#include "capture/person_capture_type.h"
#include "core/core/cvai_core_types.h"
#include "core/cviai_core.h"

#include <cvi_comm_vb.h>
#include <cvi_sys.h>
#include "cvi_comm.h"

typedef struct {
  cviai_handle_t ai_handle;
  face_capture_t *face_cpt_info;
  person_capture_t *person_cpt_info;
} cviai_app_context_t;

#ifdef __cplusplus
extern "C" {
#endif

/** @typedef cviai_app_handle_t
 *  @ingroup core_cviaiapp
 *  @brief A cviai application handle.
 */
typedef cviai_app_context_t *cviai_app_handle_t;

/**
 * @brief Create a cviai_app_handle_t.
 * @ingroup core_cviaiapp
 *
 * @param handle A app handle.
 * @param ai_handle A cviai handle.
 * @return CVI_S32 Return CVIAI_SUCCESS if succeed.
 */
DLL_EXPORT CVI_S32 CVI_AI_APP_CreateHandle(cviai_app_handle_t *handle, cviai_handle_t ai_handle);

/**
 * @brief Destroy a cviai_app_handle_t.
 * @ingroup core_cviaiapp
 *
 * @param handle A app handle.
 * @return CVI_S32 Return CVIAI_SUCCESS if success to destroy handle.
 */
DLL_EXPORT CVI_S32 CVI_AI_APP_DestroyHandle(cviai_app_handle_t handle);

/* Face Capture */
DLL_EXPORT CVI_S32 CVI_AI_APP_FaceCapture_Init(const cviai_app_handle_t handle,
                                               uint32_t buffer_size);

DLL_EXPORT CVI_S32 CVI_AI_APP_FaceCapture_QuickSetUp(const cviai_app_handle_t handle,
                                                     int fd_model_id, int fr_model_id,
                                                     const char *fd_model_path,
                                                     const char *fr_model_path,
                                                     const char *fq_model_path);

DLL_EXPORT CVI_S32 CVI_AI_APP_FaceCapture_GetDefaultConfig(face_capture_config_t *cfg);

DLL_EXPORT CVI_S32 CVI_AI_APP_FaceCapture_SetConfig(const cviai_app_handle_t handle,
                                                    face_capture_config_t *cfg);

DLL_EXPORT CVI_S32 CVI_AI_APP_FaceCapture_Run(const cviai_app_handle_t handle,
                                              VIDEO_FRAME_INFO_S *frame);

DLL_EXPORT CVI_S32 CVI_AI_APP_FaceCapture_SetMode(const cviai_app_handle_t handle,
                                                  capture_mode_e mode);
DLL_EXPORT CVI_S32 CVI_AI_APP_FaceCapture_CleanAll(const cviai_app_handle_t handle);

/* Person Capture */
DLL_EXPORT CVI_S32 CVI_AI_APP_PersonCapture_Init(const cviai_app_handle_t handle,
                                                 uint32_t buffer_size);

DLL_EXPORT CVI_S32 CVI_AI_APP_PersonCapture_QuickSetUp(const cviai_app_handle_t handle,
                                                       const char *od_model_name,
                                                       const char *od_model_path,
                                                       const char *reid_model_path);

DLL_EXPORT CVI_S32 CVI_AI_APP_PersonCapture_GetDefaultConfig(person_capture_config_t *cfg);

DLL_EXPORT CVI_S32 CVI_AI_APP_PersonCapture_SetConfig(const cviai_app_handle_t handle,
                                                      person_capture_config_t *cfg);

DLL_EXPORT CVI_S32 CVI_AI_APP_PersonCapture_Run(const cviai_app_handle_t handle,
                                                VIDEO_FRAME_INFO_S *frame);

DLL_EXPORT CVI_S32 CVI_AI_APP_PersonCapture_SetMode(const cviai_app_handle_t handle,
                                                    capture_mode_e mode);
DLL_EXPORT CVI_S32 CVI_AI_APP_PersonCapture_CleanAll(const cviai_app_handle_t handle);

#ifdef __cplusplus
}
#endif

#endif  // End of _CVIAI_APP_H_
