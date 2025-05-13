#ifndef TDL_LOG_HPP
#define TDL_LOG_HPP
#include <inttypes.h>
#include <cstring>

// 辅助宏，用于提取文件名（不含路径）
#define __FILENAME__ \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#ifdef CONFIG_ALIOS
#include <ulog/ulog.h>
#ifdef LOGD
#undef LOGD
#endif
#define LOGD(fmt, ...)                                                  \
  ulog(LOG_DEBUG, "TDLSDK", ULOG_TAG, "[%s:%d] [D] " fmt, __FILENAME__, \
       __LINE__, ##__VA_ARGS__)
#ifdef LOGI
#undef LOGI
#endif
#define LOGI(fmt, ...)                                                 \
  ulog(LOG_INFO, "TDLSDK", ULOG_TAG, "[%s:%d] [I] " fmt, __FILENAME__, \
       __LINE__, ##__VA_ARGS__)
#ifdef LOGN
#undef LOGN
#endif
#define LOGN(fmt, ...)                                                   \
  ulog(LOG_NOTICE, "TDLSDK", ULOG_TAG, "[%s:%d] [N] " fmt, __FILENAME__, \
       __LINE__, ##__VA_ARGS__)
#ifdef LOGW
#undef LOGW
#endif
#define LOGW(fmt, ...)                                                    \
  ulog(LOG_WARNING, "TDLSDK", ULOG_TAG, "[%s:%d] [W] " fmt, __FILENAME__, \
       __LINE__, ##__VA_ARGS__)
#ifdef LOGE
#undef LOGE
#endif
#define LOGE(fmt, ...)                                                \
  ulog(LOG_ERR, "TDLSDK", ULOG_TAG, "[%s:%d] [E] " fmt, __FILENAME__, \
       __LINE__, ##__VA_ARGS__)
#ifdef LOGC
#undef LOGC
#endif
#define LOGC(fmt, ...)                                                 \
  ulog(LOG_CRIT, "TDLSDK", ULOG_TAG, "[%s:%d] [C] " fmt, __FILENAME__, \
       __LINE__, ##__VA_ARGS__)
#ifndef syslog
#define syslog(level, fmt, ...) printf(fmt, ##__VA_ARGS__)
#endif
#else
#include <syslog.h>

#define MODULE_NAME "TDLSDK"
#define TDL_LOG_CHN LOG_LOCAL7
#define LOGD(fmt, ...)                                                        \
  syslog(TDL_LOG_CHN | LOG_DEBUG, "[%s:%d] [D] " fmt, __FILENAME__, __LINE__, \
         ##__VA_ARGS__)
#define LOGI(fmt, ...)                                                       \
  syslog(TDL_LOG_CHN | LOG_INFO, "[%s:%d] [I] " fmt, __FILENAME__, __LINE__, \
         ##__VA_ARGS__)
#define LOGN(fmt, ...)                                                         \
  syslog(TDL_LOG_CHN | LOG_NOTICE, "[%s:%d] [N] " fmt, __FILENAME__, __LINE__, \
         ##__VA_ARGS__)
#define LOGW(fmt, ...)                                                \
  syslog(TDL_LOG_CHN | LOG_WARNING, "[%s:%d] [W] " fmt, __FILENAME__, \
         __LINE__, ##__VA_ARGS__)
#define LOGE(fmt, ...)                                                      \
  syslog(TDL_LOG_CHN | LOG_ERR, "[%s:%d] [E] " fmt, __FILENAME__, __LINE__, \
         ##__VA_ARGS__)
#define LOGC(fmt, ...)                                                       \
  syslog(TDL_LOG_CHN | LOG_CRIT, "[%s:%d] [C] " fmt, __FILENAME__, __LINE__, \
         ##__VA_ARGS__)

#include <stdio.h>

// #define LOGI(fmt, ...) \
//   printf("[%s:%d] [I] " fmt "\n", __FILENAME__, __LINE__, ##__VA_ARGS__)
// #define LOGE(fmt, ...) \
//   printf("[%s:%d] [E] " fmt "\n", __FILENAME__, __LINE__, ##__VA_ARGS__)
// #define LOGW(fmt, ...) \
//   printf("[%s:%d] [W] " fmt "\n", __FILENAME__, __LINE__, ##__VA_ARGS__)
#endif
#endif