#ifndef TOOL_UTILS_OD_H
#define TOOL_UTILS_OD_H
#include "cvi_comm.h"
#include "cviai.h"

#define RETURN_IF_FAILED(func)   \
  do {                           \
    CVI_S32 ai_ret = (func);     \
    if (ai_ret != CVI_SUCCESS) { \
      goto ai_failed;            \
    }                            \
  } while (0)

#define GOTO_IF_FAILED(func, result, label)                              \
  do {                                                                   \
    result = (func);                                                     \
    if (result != CVI_SUCCESS) {                                         \
      printf("failed! ret=%#x, at %s:%d\n", result, __FILE__, __LINE__); \
      goto label;                                                        \
    }                                                                    \
  } while (0)

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @typedef ODInferenceFunc
 * @brief Inference function pointer
 */
typedef int (*ODInferenceFunc)(cviai_handle_t, VIDEO_FRAME_INFO_S *, cvai_object_t *);

/**
 * @brief Get object detection model's ID and inference function according to model_name.
 * @param model_name [in] model name
 * @param model_id [out] model id
 * @param inference_func [out] inference function in AI SDK
 * @return int Return CVIAI_SUCCESS if corresponding model information is found. Otherwise, return
 * CVIAI_FAILURE
 */
CVI_S32 get_od_model_info(const char *model_name, CVI_AI_SUPPORTED_MODEL_E *model_id,
                          ODInferenceFunc *inference_func);

/**
 * @brief Get person detection model's ID and inference function according to model_name.
 * @param model_name [in] model name
 * @param model_id [out] model id
 * @param inference_func [out] inference function in AI SDK
 * @return int Return CVIAI_SUCCESS if corresponding model information is found. Otherwise, return
 * CVIAI_FAILURE
 */
CVI_S32 get_pd_model_info(const char *model_name, CVI_AI_SUPPORTED_MODEL_E *model_id,
                          ODInferenceFunc *inference_func);

#ifdef __cplusplus
}
#endif

#endif
