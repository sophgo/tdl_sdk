#include "tracer.h"

#if ENABLE_TRACE
static Tracer gTracer;
int Tracer::sTraceFD = -1;
#endif
