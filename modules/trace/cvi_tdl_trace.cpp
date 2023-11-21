#include "cvi_tdl_trace.hpp"
// clang-format off
#ifdef ENABLE_TRACE
  #ifndef SYSTRACE_FALLBACK
    #if __GNUC__ >= 7
    PERFETTO_TRACK_EVENT_STATIC_STORAGE();
    #endif
  #endif
#endif
// clang-format on