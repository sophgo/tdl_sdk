#pragma once
#include <inttypes.h>
#include <syslog.h>

#define MODULE_NAME "AISDK"
#define CVIAI_LOG_CHN LOG_LOCAL7
#define LOGD(fmt, ...) \
  syslog(CVIAI_LOG_CHN | LOG_DEBUG, "[%s] [D] " fmt, MODULE_NAME, ##__VA_ARGS__)
#define LOGI(fmt, ...) syslog(CVIAI_LOG_CHN | LOG_INFO, "[%s] [I] " fmt, MODULE_NAME, ##__VA_ARGS__)
#define LOGN(fmt, ...) \
  syslog(CVIAI_LOG_CHN | LOG_NOTICE, "[%s] [N] " fmt, MODULE_NAME, ##__VA_ARGS__)
#define LOGW(fmt, ...) \
  syslog(CVIAI_LOG_CHN | LOG_WARNING, "[%s] [W] " fmt, MODULE_NAME, ##__VA_ARGS__)
#define LOGE(fmt, ...) syslog(CVIAI_LOG_CHN | LOG_ERR, "[%s] [E] " fmt, MODULE_NAME, ##__VA_ARGS__)
#define LOGC(fmt, ...) syslog(CVIAI_LOG_CHN | LOG_CRIT, "[%s] [C] " fmt, MODULE_NAME, ##__VA_ARGS__)