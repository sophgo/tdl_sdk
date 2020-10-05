#ifndef _CVIAI_UTILS_H_
#define _CVIAI_UTILS_H_
#include "face/cvai_face_types.h"
#include "object/cvai_object_types.h"

#ifdef __cplusplus
extern "C" {
#endif

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
