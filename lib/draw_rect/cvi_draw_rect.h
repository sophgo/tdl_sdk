#ifndef _CVI_DARW_RECT_HEAD_
#define _CVI_DARW_RECT_HEAD_

#if defined(_MIDDLEWARE_V2_)
#include <linux/cvi_type.h>
#else
#include <cvi_type.h>
#endif
#define DLL_EXPORT __attribute__((visibility("default")))
#include "cviai.h"
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Draw rect to frame with given face meta with a global brush.
 * @ingroup core_cviaiservice
 *
 * @param meta meta structure.
 * @param frame In/ out YUV frame.
 * @param drawText Choose to draw name of the face.
 * @param brush A brush for drawing
 * @return CVI_S32 Return CVIAI_SUCCESS if succeed.
 */
DLL_EXPORT CVI_S32 CVI_AI_FaceDrawRect(const cvai_face_t *meta, VIDEO_FRAME_INFO_S *frame,
                                               const bool drawText, cvai_service_brush_t brush);

/**
 * @brief Draw rect to frame with given face meta with individual brushes.
 * @ingroup core_cviaiservice
 *
 * @param meta meta structure.
 * @param frame In/ out YUV frame.
 * @param drawText Choose to draw name of the face.
 * @param brushes brushes for drawing. The count of brushes must be same as meta->size.
 * @return CVI_S32 Return CVIAI_SUCCESS if succeed.
 */
DLL_EXPORT CVI_S32 CVI_AI_FaceDrawRect2(const cvai_face_t *meta, VIDEO_FRAME_INFO_S *frame,
                                                const bool drawText, cvai_service_brush_t *brushes);

/**
 * @brief Draw rect to frame with given object meta with a global brush.
 * @ingroup core_cviaiservice
 *
 * @param meta meta structure.
 * @param frame In/ out YUV frame.
 * @param drawText Choose to draw name of the object.
 * @param brush A brush for drawing
 * @return CVI_S32 Return CVIAI_SUCCESS if succeed.
 */
DLL_EXPORT CVI_S32 CVI_AI_ObjectDrawRect(const cvai_object_t *meta,
                                                 VIDEO_FRAME_INFO_S *frame, const bool drawText,
                                                 cvai_service_brush_t brush);

/**
 * @brief Draw rect to frame with given object meta with individual brushes.
 * @ingroup core_cviaiservice
 *
 * @param meta meta structure.
 * @param frame In/ out YUV frame.
 * @param drawText Choose to draw name of the object.
 * @param brushes brushes for drawing. The count of brushes must be same as meta->size.
 * @return CVI_S32 Return CVIAI_SUCCESS if succeed.
 */
DLL_EXPORT CVI_S32 CVI_AI_ObjectDrawRect2(const cvai_object_t *meta,
                                                  VIDEO_FRAME_INFO_S *frame, const bool drawText,
                                                  cvai_service_brush_t *brushes);


#ifdef __cplusplus
}
#endif
#endif