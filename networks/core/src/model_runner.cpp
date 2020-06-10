// Copyright 2018 Bitmain Inc.
// License
// Author Yangwen Hunag <yangwen.huang@bitmain.com>

#include "model_runner.hpp"
#include <config.h>

#if USE_LEGACY_BMTAP2 == 0
#else
// For backward compatibility
struct bmnet_model_info_s {
    // Empty
};
#endif