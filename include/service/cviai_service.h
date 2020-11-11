#ifndef _CVIAI_OBJSERVICE_H_
#define _CVIAI_OBJSERVICE_H_
#include "service/cviai_service_types.h"

#include "core/cviai_core.h"
#include "core/face/cvai_face_types.h"
#include "core/object/cvai_object_types.h"

#include <cvi_sys.h>

/** @typedef cviai_service_handle_t
 *  @ingroup core_cviaiservice
 *  @brief A cviai objservice handle.
 */
typedef void *cviai_service_handle_t;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Create a cviai_service_handle_t.
 * @ingroup core_cviaiservice
 *
 * @param handle A service handle.
 * @param ai_handle A cviai handle.
 * @return CVI_S32 Return CVI_SUCCESS if succeed.
 */
DLL_EXPORT CVI_S32 CVI_AI_Service_CreateHandle(cviai_service_handle_t *handle,
                                               cviai_handle_t ai_handle);

/**
 * @brief Destroy a cviai_service_handle_t.
 * @ingroup core_cviaiservice
 *
 * @param handle A service handle.
 * @return CVI_S32 Return CVI_SUCCESS if success to destroy handle.
 */
DLL_EXPORT CVI_S32 CVI_AI_Service_DestroyHandle(cviai_service_handle_t handle);

/**
 * @brief Register a feature array to OBJ Service.
 * @ingroup core_cviaiservice
 *
 * @param handle A service handle.
 * @param featureArray Input registered feature array.
 * @param method Set feature matching method.
 * @return CVI_S32 Return CVI_SUCCESS if succeed.
 */
DLL_EXPORT CVI_S32 CVI_AI_Service_RegisterFeatureArray(
    cviai_service_handle_t handle, const cvai_service_feature_array_t featureArray,
    const cvai_service_feature_matching_e method);

/**
 * @brief Do a single cviai_face_t feature matching with registed feature array.
 * @ingroup core_cviaiservice
 *
 * @param handle A service handle.
 * @param face_info The cvai_face_info_t from NN output with feature data.
 * @param k Output top k results.
 * @param index Output top k index.
 * @return CVI_S32 Return CVI_SUCCESS if succeed.
 */
DLL_EXPORT CVI_S32 CVI_AI_Service_FaceInfoMatching(cviai_service_handle_t handle,
                                                   cvai_face_info_t *face_info, const uint32_t k,
                                                   uint32_t **index);

/**
 * @brief Do a single cvai_object_info_t feature matching with registed feature array.
 * @ingroup core_cviaiservice
 *
 * @param handle A service handle.
 * @param object_info The cvai_object_info_t from NN output with feature data.
 * @param k Output top k results.
 * @param index Output top k index.
 * @return CVI_S32 Return CVI_SUCCESS if succeed.
 */
DLL_EXPORT CVI_S32 CVI_AI_Service_ObjectInfoMatching(cviai_service_handle_t handle,
                                                     const cvai_object_info_t *object_info,
                                                     const uint32_t k, uint32_t **index);

/**
 * @brief Do a single raw data with registed feature array.
 * @ingroup core_cviaiservice
 *
 * @param handle A service handle.
 * @param feature Raw feature vector.
 * @param type The data type of the feature vector.
 * @param k Output top k results.
 * @param index Output top k index.
 * @return CVI_S32 Return CVI_SUCCESS if succeed.
 */
DLL_EXPORT CVI_S32 CVI_AI_Service_RawMatching(cviai_service_handle_t handle, const uint8_t *feature,
                                              const feature_type_e type, const uint32_t k,
                                              uint32_t **index);

/**
 * @brief Zoom in to the largest face from the output of face detection results.
 * @ingroup core_cviaiservice
 *
 * @param handle A service handle.
 * @param inFrame Input frame.
 * @param meta THe result from face detection.
 * @param face_skip_ratio Skip the faces that are too small comparing to the area of the image.
 * Default is 0.05.
 * @param trans_ratio Change to zoom in ratio. Default is 0.1.
 * @param padding_ratio Bounding box padding ratio. Default is 0.3.
 * @param outFrame Output result image, will keep aspect ratio.
 * @return CVI_S32 Return CVI_SUCCESS if succeed.
 */
DLL_EXPORT CVI_S32 CVI_AI_Service_FaceDigitalZoom(
    cviai_service_handle_t handle, const VIDEO_FRAME_INFO_S *inFrame, const cvai_face_t *meta,
    const float face_skip_ratio, const float trans_ratio, const float padding_ratio,
    VIDEO_FRAME_INFO_S *outFrame);

/**
 * @brief Zoom in to the largest face from the output of object detection results.
 * @ingroup core_cviaiservice
 *
 * @param handle A service handle.
 * @param inFrame Input frame.
 * @param meta THe result from face detection.
 * @param obj_skip_ratio Skip the objects that are too small comparing to the area of the image.
 * Default is 0.05.
 * @param trans_ratio Change to zoom in ratio. Default is 0.1.
 * @param padding_ratio Bounding box padding ratio. Default is 0.3.
 * @param outFrame Output result image, will keep aspect ratio.
 * @return CVI_S32 Return CVI_SUCCESS if succeed.
 */
DLL_EXPORT CVI_S32 CVI_AI_Service_ObjectDigitalZoom(
    cviai_service_handle_t handle, const VIDEO_FRAME_INFO_S *inFrame, const cvai_object_t *meta,
    const float obj_skip_ratio, const float trans_ratio, const float padding_ratio,
    VIDEO_FRAME_INFO_S *outFrame);

/**
 * @brief Draw rect to YUV frame with given face meta.
 * @ingroup core_cviaiservice
 *
 * @param meta meta structure.
 * @param frame In/ out YUV frame.
 * @param drawText Choose to draw name of the face.
 * @return CVI_S32 Return CVI_SUCCESS if succeed.
 */
DLL_EXPORT CVI_S32 CVI_AI_Service_FaceDrawRect(const cvai_face_t *meta, VIDEO_FRAME_INFO_S *frame,
                                               const bool drawText);

/**
 * @brief Draw rect to YUV frame with given object meta.
 * @ingroup core_cviaiservice
 *
 * @param meta meta structure.
 * @param frame In/ out YUV frame.
 * @param drawText Choose to draw name of the object.
 * @return CVI_S32 Return CVI_SUCCESS if succeed.
 */
DLL_EXPORT CVI_S32 CVI_AI_Service_ObjectDrawRect(const cvai_object_t *meta,
                                                 VIDEO_FRAME_INFO_S *frame, const bool drawText);

/**
 * @brief Set intersect area for detection.
 * @ingroup core_cviaiservice
 *
 * @param handle A service handle.
 * @param pts Intersect area or line. (pts must larger than 2 or more.)
 * @return CVI_S32 Return CVI_SUCCESS if succeed.
 */
DLL_EXPORT CVI_S32 CVI_AI_Service_SetIntersect(cviai_service_handle_t handle,
                                               const cvai_pts_t *pts);

/**
 * @brief Check if the object intersected with the set area or line.
 * @ingroup core_cviaiservice
 *
 * @param handle A service handle.
 * @param frame Input frame.
 * @param obj_meta Object meta structure.
 * @param status Output status of each object.
 * @return CVI_S32 Return CVI_SUCCESS if succeed.
 */
DLL_EXPORT CVI_S32 CVI_AI_Service_ObjectDetectIntersect(cviai_service_handle_t handle,
                                                        const VIDEO_FRAME_INFO_S *frame,
                                                        const cvai_object_t *obj_meta,
                                                        cvai_area_detect_e **status);
#ifdef __cplusplus
}
#endif

#endif  // End of _CVIAI_OBJSERVICE_H_