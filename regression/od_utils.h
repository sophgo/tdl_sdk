#ifndef SAMPLE_UTILS_H_
#define SAMPLE_UTILS_H_
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

typedef int (*ODInferenceFunc)(cviai_handle_t, VIDEO_FRAME_INFO_S *, cvai_object_t *);
CVI_S32 get_od_model_info(const char *model_name, CVI_AI_SUPPORTED_MODEL_E *model_index,
                          ODInferenceFunc *inference_func);

#endif
