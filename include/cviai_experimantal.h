#ifndef _CVIAI_H_
#define _CVIAI_H_

#include "cviai.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Enable GDC hardware acceleration.
 *
 * @param handle An AI SDK handle.
 * @param use_gdc Set true to use hardware.
 */
void CVI_AI_EnableGDC(cviai_handle_t handle, bool use_gdc);

#ifdef __cplusplus
}
#endif

#endif
