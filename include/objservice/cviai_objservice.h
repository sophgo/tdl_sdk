#ifndef _CVIAI_OBJSERVICE_H_
#define _CVIAI_OBJSERVICE_H_

#include "core/cviai_core.h"
#include "core/object/cvai_object_types.h"

#include <cvi_sys.h>

typedef void *cviai_objservice_handle_t;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Create a cviai_objservice_handle_t.
 *
 * @param handle An OBJ Service handle.
 * @return CVI_S32 Return CVI_SUCCESS if succeed.
 */
CVI_S32 CVI_AI_OBJService_CreateHandle(cviai_objservice_handle_t *handle, cviai_handle_t ai_handle);

/**
 * @brief Destroy a cviai_objservice_handle_t.
 *
 * @param handle An OBJ Service handle.
 * @return CVI_S32 Return CVI_SUCCESS if success to destroy handle.
 */
CVI_S32 CVI_AI_OBJService_DestroyHandle(cviai_objservice_handle_t handle);

/**
 * @brief Zoom in to the largest face from the output of object detection results.
 *
 * @param handle An OBJ Service handle.
 * @param inFrame Input frame.
 * @param meta THe result from face detection.
 * @param object_skip_ratio Skip the objects that are too small comparing to the area of the image.
 * Default is 0.05.
 * @param trans_ratio Change to zoom in ratio. Default is 0.1.
 * @param outFrame Output result image, will keep aspect ratio.
 * @return CVI_S32 Return CVI_SUCCESS if succeed.
 */
CVI_S32 CVI_AI_OBJService_DigitalZoom(cviai_objservice_handle_t handle,
                                      const VIDEO_FRAME_INFO_S *inFrame, const cvai_object_t *meta,
                                      const float obj_skip_ratio, const float trans_ratio,
                                      VIDEO_FRAME_INFO_S *outFrame);

/**
 * @brief Draw rect to YUV frame with given object meta.
 *
 * @param meta meta structure.
 * @param frame In/ out YUV frame.
 * @return CVI_S32 Return CVI_SUCCESS if succeed.
 */
CVI_S32 CVI_AI_OBJService_DrawRect(const cvai_object_t *meta, VIDEO_FRAME_INFO_S *frame);
#ifdef __cplusplus
}
#endif

#endif  // End of _CVIAI_OBJSERVICE_H_