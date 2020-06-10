/**
 * @file log_common.h
 * @ingroup NETWORKS_UTILS
 */
#pragma once

#include "glog/logging.h"  //google-glog

#define LOGD DLOG(INFO)
#define LOGI LOG(INFO)
#define LOGW LOG(WARNING)
#define LOGE LOG(ERROR)
#define LOGF LOG(FATAL)
