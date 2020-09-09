#pragma once
#if __GNUC__ >= 7
#include "perfetto/perfetto.h"
PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("cviai_core").SetDescription("Events from cviai_core"),
    perfetto::Category("cviai_frservice").SetDescription("Events from cviai_frservice"),
    perfetto::Category("cviai_objservice").SetDescription("Events from cviai_objservice"));
#else
#define TRACE_EVENT(X, Y)
#define TRACE_EVENT_BEGIN(X, Y)
#define TRACE_EVENT_END(X)
#endif