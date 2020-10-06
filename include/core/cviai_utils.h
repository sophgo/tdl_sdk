#ifndef _CVIAI_UTILS_H_
#define _CVIAI_UTILS_H_
#include "cviai_core.h"
#include "face/cvai_face_types.h"
#include "object/cvai_object_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Do quantize scale for a given VIDEO_FRAME_INFO_S, but the quantized_factor and
 *        quantized_mean have to calculate manually.
 *
 * @param handle An AI SDK handle.
 * @param frame Input frame.
 * @param output Output frame.
 * @param factor Quantized factor.
 * @param mean Quantized mean, must be positive.
 * @param thread Working thread id of vpss thread. A new thread will be created if thread doesn't
 * exist.
 * @return int Return CVI_SUCCESS on success.
 */
int CVI_AI_SQPreprocessRaw(cviai_handle_t handle, const VIDEO_FRAME_INFO_S *frame,
                           VIDEO_FRAME_INFO_S *output, const float quantized_factor,
                           const float quantized_mean, const uint32_t thread);

/**
 * @brief Do Quantize scale for a given VIDEO_FRAME_INFO_S.
 *        The formula of scale is (factor * x - mean).
 *
 * @param handle An AI SDK handle.
 * @param frame Input frame.
 * @param output Output frame.
 * @param factor Factor.
 * @param mean Mean.
 * @param quantize_threshold Threshold for quantization.
 * @param thread Working thread id of vpss thread. A new thread will be created if thread doesn't
 * exist.
 * @return int Return CVI_SUCCESS on success.
 */
int CVI_AI_SQPreprocess(cviai_handle_t handle, const VIDEO_FRAME_INFO_S *frame,
                        VIDEO_FRAME_INFO_S *output, const float factor, const float mean,
                        const float quantize_threshold, const uint32_t thread);

/**
 * @brief Do non maximum suppression on cvai_face_t.
 *
 * @param face Input cvai_face_t.
 * @param faceNMS Output result.
 * @param threshold NMS threshold.
 * @param method Support 'u' and 'm'. (intersection over union and intersection over min area)
 * @return int Return CVI_SUCCESS on success.
 */
int CVI_AI_FaceNMS(const cvai_face_t *face, cvai_face_t *faceNMS, const float threshold,
                   const char method);

/**
 * @brief Do non maximum suppression on cvai_object_t.
 *
 * @param obj Input cvai_object_t.
 * @param objNMS Output result.
 * @param threshold NMS threshold.
 * @param method Support 'u' and 'm'. (intersection over union and intersection over min area)
 * @return int Return CVI_SUCCESS on success.
 */
int CVI_AI_ObjectNMS(const cvai_object_t *obj, cvai_object_t *objNMS, const float threshold,
                     const char method);

#ifdef __cplusplus
}
#endif

#endif  // End of _CVIAI_UTILS_H_
