#ifndef _CVIAI_APP_PERSON_CAPTURE_H_
#define _CVIAI_APP_PERSON_CAPTURE_H_

#include "app/capture/person_capture_type.h"
#include "core/cviai_core.h"

CVI_S32 _PersonCapture_Free(person_capture_t *person_cpt_info);

CVI_S32 _PersonCapture_Init(person_capture_t **person_cpt_info, uint32_t buffer_size);

CVI_S32 _PersonCapture_QuickSetUp(cviai_handle_t ai_handle, person_capture_t *person_cpt_info,
                                  const char *od_model_name, const char *od_model_path,
                                  const char *reid_model_path);

CVI_S32 _PersonCapture_GetDefaultConfig(person_capture_config_t *cfg);

CVI_S32 _PersonCapture_SetConfig(person_capture_t *person_cpt_info, person_capture_config_t *cfg,
                                 cviai_handle_t ai_handle);

CVI_S32 _PersonCapture_Run(person_capture_t *person_cpt_info, const cviai_handle_t ai_handle,
                           VIDEO_FRAME_INFO_S *frame);

CVI_S32 _PersonCapture_SetMode(person_capture_t *person_cpt_info, capture_mode_e mode);

CVI_S32 _PersonCapture_CleanAll(person_capture_t *person_cpt_info);

#endif  // End of _CVIAI_APP_PERSON_CAPTURE_H_
