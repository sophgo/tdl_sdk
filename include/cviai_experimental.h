#ifndef _CVIAI_EXPERIMENTAL_H_
#define _CVIAI_EXPERIMENTAL_H_

#include "cviai.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Enable GDC hardware acceleration.
 * @ingroup core_ai_settings
 *
 * @param handle An AI SDK handle.
 * @param use_gdc Set true to use hardware.
 */
DLL_EXPORT void CVI_AI_EnableGDC(cviai_handle_t handle, bool use_gdc);

#ifdef __cplusplus
}
#endif

#endif  // End of _CVIAI_EXPERIMENTAL_H_
