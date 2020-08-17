#ifndef _CVIAI_FRSERVICE_H_
#define _CVIAI_FRSERVICE_H_
#include "frservice/cviai_frservice_types.h"

#include "core/cviai_core.h"
#include "core/face/cvai_face_types.h"

#include <cvi_sys.h>

typedef void *cviai_frservice_handle_t;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Create a cviai_frservice_handle_t.
 *
 * @param handle An FR Service handle.
 * @return CVI_S32 Return CVI_SUCCESS if succeed.
 */
CVI_S32 CVI_AI_FRService_CreateHandle(cviai_frservice_handle_t *handle, cviai_handle_t *ai_handle);

/**
 * @brief Destroy a cviai_frservice_handle_t.
 *
 * @param handle An FR Service handle.
 * @return CVI_S32 Return CVI_SUCCESS if success to destroy handle.
 */
CVI_S32 CVI_AI_FRService_DestroyHandle(cviai_frservice_handle_t handle);

/**
 * @brief Register a feature array to FR Service.
 *
 * @param handle An FR Service handle.
 * @param featureArray Input registered feature array.
 * @return CVI_S32 Return CVI_SUCCESS if succeed.
 */
CVI_S32 CVI_AI_FRService_RegisterFeatureArray(cviai_frservice_handle_t handle,
                                              const cvai_frservice_feature_array_t featureArray);

/**
 * @brief Do a single cviai_face_t feature matching with registed feature array.
 *
 * @param handle An FR Service handle.
 * @param face The cvai_face_t from NN output with feature data.
 * @param k Output top k results.
 * @param index Output top k index.
 * @return CVI_S32 Return CVI_SUCCESS if succeed.
 */
CVI_S32 CVI_AI_FRService_FaceInfoMatching(cviai_frservice_handle_t handle, const cvai_face_t *face,
                                          const uint32_t k, uint32_t **index);

/**
 * @brief Do a single raw data with registed feature array.
 *
 * @param handle An FR Service handle.
 * @param face The cvai_face_t from NN output with feature data.
 * @param k Output top k results.
 * @param index Output top k index.
 * @return CVI_S32 Return CVI_SUCCESS if succeed.
 */
CVI_S32 CVI_AI_FRService_RawMatching(cviai_frservice_handle_t handle, const uint8_t *feature,
                                     const feature_type_e type, const uint32_t k, uint32_t **index);

#ifdef __cplusplus
}
#endif

#endif  // End of _CVIAI_FRSERVICE_H_