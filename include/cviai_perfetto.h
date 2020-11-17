#ifndef _CVIAI_TRACE_H_
#define _CVIAI_TRACE_H_

#include "cviai.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Init Perfetto tracer.
 *
 */
DLL_EXPORT void CVI_AI_PerfettoInit();

DLL_EXPORT void CVI_AI_TraceBegin(const char *name);

DLL_EXPORT void CVI_AI_TraceEnd();

#ifdef __cplusplus
}
#endif

#endif  // End of _CVIAI_TRACE_H_
