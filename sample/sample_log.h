#ifndef _SAMPLE_LOG_H_H
#define _SAMPLE_LOG_H_H

#ifndef LOG_TAG
#define LOG_TAG "Unknown"
#endif

#define LOG_LEVEL_VERBOSE 4
#define LOG_LEVEL_DEBUG 3
#define LOG_LEVEL_INFO 2
#define LOG_LEVEL_WARNING 1
#define LOG_LEVEL_ERROR 0

#ifndef LOG_LEVEL
#define LOG_LEVEL LOG_LEVEL_INFO
#endif

#define LOG_PRINT(level, level_prefix, fmt, ...)                                   \
  do {                                                                             \
    if (level <= LOG_LEVEL) {                                                      \
      printf("[%s:%d][%s]: " fmt, LOG_TAG, __LINE__, level_prefix, ##__VA_ARGS__); \
    }                                                                              \
  } while (0)

#define AI_LOGV(fmt, ...) LOG_PRINT(LOG_LEVEL_VERBOSE, "V", fmt, ##__VA_ARGS__)
#define AI_LOGD(fmt, ...) LOG_PRINT(LOG_LEVEL_DEBUG, "D", fmt, ##__VA_ARGS__)
#define AI_LOGI(fmt, ...) LOG_PRINT(LOG_LEVEL_INFO, "I", fmt, ##__VA_ARGS__)
#define AI_LOGW(fmt, ...) LOG_PRINT(LOG_LEVEL_WARNING, "W", fmt, ##__VA_ARGS__)
#define AI_LOGE(fmt, ...) LOG_PRINT(LOG_LEVEL_ERROR, "E", fmt, ##__VA_ARGS__)

#endif