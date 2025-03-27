#ifndef TDL_LOG_HPP
#define TDL_LOG_HPP
#include <inttypes.h>
#ifdef CONFIG_ALIOS
#include <ulog/ulog.h>
#ifdef LOGD
#undef LOGD
#endif
#define LOGD(fmt, ...) \
  ulog(LOG_DEBUG, "TDLSDK", ULOG_TAG, "[TDLSDK] [D] " fmt, ##__VA_ARGS__)
#ifdef LOGI
#undef LOGI
#endif
#define LOGI(fmt, ...) \
  ulog(LOG_INFO, "TDLSDK", ULOG_TAG, "[TDLSDK] [I] " fmt, ##__VA_ARGS__)
#ifdef LOGN
#undef LOGN
#endif
#define LOGN(fmt, ...) \
  ulog(LOG_NOTICE, "TDLSDK", ULOG_TAG, "[TDLSDK] [N] " fmt, ##__VA_ARGS__)
#ifdef LOGW
#undef LOGW
#endif
#define LOGW(fmt, ...) \
  ulog(LOG_WARNING, "TDLSDK", ULOG_TAG, "[TDLSDK] [W] " fmt, ##__VA_ARGS__)
#ifdef LOGE
#undef LOGE
#endif
#define LOGE(fmt, ...) \
  ulog(LOG_ERR, "TDLSDK", ULOG_TAG, "[TDLSDK] [E] " fmt, ##__VA_ARGS__)
#ifdef LOGC
#undef LOGC
#endif
#define LOGC(fmt, ...) \
  ulog(LOG_CRIT, "TDLSDK", ULOG_TAG, "[TDLSDK] [C] " fmt, ##__VA_ARGS__)
#ifndef syslog
#define syslog(level, fmt, ...) printf(fmt, ##__VA_ARGS__)
#endif
#else
#include <syslog.h>

#define MODULE_NAME "TDLSDK"
#define TDL_LOG_CHN LOG_LOCAL7
#define LOGD(fmt, ...)                                              \
  syslog(TDL_LOG_CHN | LOG_DEBUG, "[%s] [D] " fmt, MODULE_NAME, \
         ##__VA_ARGS__)
#define LOGI(fmt, ...)                                             \
  syslog(TDL_LOG_CHN | LOG_INFO, "[%s] [I] " fmt, MODULE_NAME, \
         ##__VA_ARGS__)
#define LOGN(fmt, ...)                                               \
  syslog(TDL_LOG_CHN | LOG_NOTICE, "[%s] [N] " fmt, MODULE_NAME, \
         ##__VA_ARGS__)
#define LOGW(fmt, ...)                                                \
  syslog(TDL_LOG_CHN | LOG_WARNING, "[%s] [W] " fmt, MODULE_NAME, \
         ##__VA_ARGS__)
#define LOGE(fmt, ...) \
  syslog(TDL_LOG_CHN | LOG_ERR, "[%s] [E] " fmt, MODULE_NAME, ##__VA_ARGS__)
#define LOGC(fmt, ...)                                             \
  syslog(TDL_LOG_CHN | LOG_CRIT, "[%s] [C] " fmt, MODULE_NAME, \
         ##__VA_ARGS__)

#include <stdio.h>

// #define LOGI(fmt, ...) printf("[%s] [I] " fmt "\n", MODULE_NAME,
// ##__VA_ARGS__) #define LOGE(fmt, ...) printf("[%s] [E] " fmt "\n",
// MODULE_NAME, ##__VA_ARGS__) #define LOGW(fmt, ...) printf("[%s] [W] " fmt
// "\n", MODULE_NAME, ##__VA_ARGS__)
#endif
#endif