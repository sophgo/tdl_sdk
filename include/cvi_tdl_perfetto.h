#ifndef _CVI_TDL_TRACE_H_
#define _CVI_TDL_TRACE_H_

#include "cvi_tdl.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Init Perfetto tracer.
 *
 */
DLL_EXPORT void CVI_TDL_PerfettoInit();

DLL_EXPORT void CVI_TDL_TraceBegin(const char *name);

DLL_EXPORT void CVI_TDL_TraceEnd();

#ifdef __cplusplus
}
#endif

#endif  // End of _CVI_TDL_TRACE_H_
