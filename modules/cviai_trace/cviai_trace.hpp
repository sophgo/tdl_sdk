#pragma once
// clang-format off
#ifdef ENABLE_TRACE
  #if __GNUC__ >= 7
    #include "perfetto.h"
    PERFETTO_DEFINE_CATEGORIES(
        perfetto::Category("cviai_core").SetDescription("Events from cviai_core"),
        perfetto::Category("cviai_frservice").SetDescription("Events from cviai_frservice"),
        perfetto::Category("cviai_objservice").SetDescription("Events from cviai_objservice"));
  #else
    #error "Perfetto only supports GCC version >= 7."
  #endif
#else
  #define TRACE_EVENT(X, Y)
  #define TRACE_EVENT_BEGIN(X, Y)
  #define TRACE_EVENT_END(X)
#endif

inline void __attribute__((always_inline)) prefettoInit() {
#ifdef ENABLE_TRACE
  #if __GNUC__ >= 7
    perfetto::TracingInitArgs args;
    args.backends |= perfetto::kInProcessBackend;
    args.backends |= perfetto::kSystemBackend;

    perfetto::Tracing::Initialize(args);
    perfetto::TrackEvent::Register();
  #endif
#endif
}
// clang-format on
