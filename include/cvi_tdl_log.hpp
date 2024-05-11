#pragma once
#include <inttypes.h>
#include <syslog.h>

#define MODULE_NAME "TDLSDK"
#define CVI_TDL_LOG_CHN LOG_LOCAL7
#define LOGD(fmt, ...) \
  syslog(CVI_TDL_LOG_CHN | LOG_DEBUG, "[%s] [D] " fmt, MODULE_NAME, ##__VA_ARGS__)
#define LOGI(fmt, ...) \
  syslog(CVI_TDL_LOG_CHN | LOG_INFO, "[%s] [I] " fmt, MODULE_NAME, ##__VA_ARGS__)
#define LOGN(fmt, ...) \
  syslog(CVI_TDL_LOG_CHN | LOG_NOTICE, "[%s] [N] " fmt, MODULE_NAME, ##__VA_ARGS__)
#define LOGW(fmt, ...) \
  syslog(CVI_TDL_LOG_CHN | LOG_WARNING, "[%s] [W] " fmt, MODULE_NAME, ##__VA_ARGS__)
#define LOGE(fmt, ...) \
  syslog(CVI_TDL_LOG_CHN | LOG_ERR, "[%s] [E] " fmt, MODULE_NAME, ##__VA_ARGS__)
#define LOGC(fmt, ...) \
  syslog(CVI_TDL_LOG_CHN | LOG_CRIT, "[%s] [C] " fmt, MODULE_NAME, ##__VA_ARGS__)
