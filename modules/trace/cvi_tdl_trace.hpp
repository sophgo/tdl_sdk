#pragma once
// clang-format off
#ifdef ENABLE_TRACE
  #ifdef SYSTRACE_FALLBACK
    #include "tracer.h"
    #define TRACE_EVENT(X, Y) ScopedTrace scopedtrace(Y)
    #define TRACE_EVENT_BEGIN(X, Y) Tracer::TraceBegin(Y)
    #define TRACE_EVENT_END(X) Tracer::TraceEnd()
  #else
    #if __GNUC__ >= 7
      #include "perfetto.h"
      PERFETTO_DEFINE_CATEGORIES(
          perfetto::Category("cvi_tdl_api").SetDescription("Events from cvi_tdl_api"),
          perfetto::Category("cvi_tdl_core").SetDescription("Events from cvi_tdl_core"),
          perfetto::Category("cvi_tdl_service").SetDescription("Events from cvi_tdl_service"));
    #else
      #error "Perfetto only supports GCC version >= 7."
    #endif
  #endif
#else
  #define TRACE_EVENT(X, Y)
  #define TRACE_EVENT_BEGIN(X, Y)
  #define TRACE_EVENT_END(X)
#endif

inline void __attribute__((always_inline)) prefettoInit() {
#ifndef SYSTRACE_FALLBACK
  #ifdef ENABLE_TRACE
    #if __GNUC__ >= 7
      perfetto::TracingInitArgs args;
      args.backends = perfetto::kSystemBackend;

      perfetto::Tracing::Initialize(args);
      perfetto::TrackEvent::Register();
    #endif
  #endif
#endif
}
// clang-format on
