#pragma once
#include <syslog.h>

#define CVIAI_LOG_CHN LOG_LOCAL7
#define LOGI(...) syslog(CVIAI_LOG_CHN | LOG_INFO, __VA_ARGS__)
#define LOGN(...) syslog(CVIAI_LOG_CHN | LOG_NOTICE, __VA_ARGS__)
#define LOGW(...) syslog(CVIAI_LOG_CHN | LOG_WARNING, __VA_ARGS__)
#define LOGE(...) syslog(CVIAI_LOG_CHN | LOG_ERR, __VA_ARGS__)
#define LOGC(...) syslog(CVIAI_LOG_CHN | LOG_CRIT, __VA_ARGS__)