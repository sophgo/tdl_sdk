#include <arpa/inet.h>
#include <curl/curl.h>
#include <locale.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>
#include <wchar.h>
#include <wctype.h>
#include "cjson/cJSON.h"


#define DEBUG 0

typedef struct {
  char role[16];
  char *content;
} Message;

// 定义响应缓冲区结构
struct MemoryChunk {
  char *data;   // 完整数据存储
  size_t size;  // 总数据长度

  // 新增流式处理状态
  struct {
    char *buffer;  // 流式处理缓冲区
    size_t len;    // 缓冲区当前长度
  } stream;
};

#define INPUT_MAX 1000

typedef struct {
  char input_buf[INPUT_MAX];  // 主输入缓冲区
  int pos;                    // 当前写入位置

  // UTF-8字符收集器
  struct {
    char bytes[4];  // UTF-8字节缓存
    int len;        // 当前收集的字节数
    wchar_t wc;     // 转换后的宽字符
  } utf8;
} InputState;

struct termios oldt;

// 新增判断是否中文字符的函数（UTF-8范围判断）
int is_chinese_char(const char *bytes) {
  // 中文字符的UTF-8首字节范围：0xE0-0xEF（3字节字符）
  return (bytes[0] & 0xF0) == 0xE0;
}

// 处理退格键
void handle_backspace(InputState *state) {
  if (state->pos == 0) return;

  // 1. 计算需要删除的字节数（UTF-8字符可能跨多字节）
  int back_bytes = 1;
  while (state->pos - back_bytes > 0 &&
         (state->input_buf[state->pos - back_bytes] & 0xC0) == 0x80) {
    back_bytes++;
  }

  // 2. 判断是否为中文字符
  const char *deleted_char = &state->input_buf[state->pos - back_bytes];
  int is_chinese = is_chinese_char(deleted_char);

  // 3. 更新缓冲区状态
  state->pos -= back_bytes;
  state->input_buf[state->pos] = '\0';

  // 4. 根据字符类型回退光标
  if (is_chinese) {
    // 中文字符：回退2列并清除
    printf("\b\b  \b\b");  // 回退两列，覆盖空格，再回退
  } else {
    // 英文字符：回退1列
    printf("\b \b");
  }
  fflush(stdout);
}

// UTF-8收集器需修改状态
int collect_utf8(InputState *state, char c) {
  if ((unsigned long)state->utf8.len >= sizeof(state->utf8.bytes) - 1) return 0;

  state->utf8.bytes[state->utf8.len++] = c;
  state->utf8.bytes[state->utf8.len] = '\0';

  mbstate_t ps = {0};
  size_t ret = mbrtowc(&state->utf8.wc, state->utf8.bytes, state->utf8.len, &ps);

  if (ret == (size_t)-1 || ret == (size_t)-2) {
    return 0;  // 需要更多字节
  } else if (ret > 0) {
    return 1;  // 完整字符
  }
  return 0;
}

size_t write_callback(char *ptr, size_t size, size_t nmemb, void *userdata) {
  size_t realsize = size * nmemb;
  struct MemoryChunk *mem = (struct MemoryChunk *)userdata;

  // 调试输出原始数据块
  // printf("\n[Raw Chunk %zu] %.*s\n", realsize, (int)realsize > 200 ? 200 : (int)realsize, ptr);

  // 1. 保存原始数据（保持原有逻辑）
  char *new_data = realloc(mem->data, mem->size + realsize + 1);
  if (!new_data) return 0;
  mem->data = new_data;
  memcpy(mem->data + mem->size, ptr, realsize);
  mem->size += realsize;
  mem->data[mem->size] = '\0';

  // 2. 合并流式缓冲区（优化内存分配）
  mem->stream.buffer = realloc(mem->stream.buffer, mem->stream.len + realsize + 1);
  memcpy(mem->stream.buffer + mem->stream.len, ptr, realsize);
  mem->stream.len += realsize;
  mem->stream.buffer[mem->stream.len] = '\0';

  // 3. 事件处理循环（增强日志）
  char *current_ptr = mem->stream.buffer;
  while (1) {
#if DEBUG
    printf("\n[Processing Cycle] Current Buffer: %.*s...\n",
           (int)(mem->stream.len - (current_ptr - mem->stream.buffer)) > 100
               ? 100
               : (int)(mem->stream.len - (current_ptr - mem->stream.buffer)),
           current_ptr);
#endif

    // 改进的事件分割逻辑
    char *event_start = strstr(current_ptr, "data:");
    if (!event_start) {
#if DEBUG
      printf("[Event Check] No more 'data:' markers found\n");
#endif
      break;
    }

    // 查找事件结束位置（支持多种结束符）
    char *event_end = NULL;
    for (char *p = event_start; p < mem->stream.buffer + mem->stream.len - 1; p++) {
      if ((*p == '\n' && *(p + 1) == '\n') ||
          (*p == '\r' && *(p + 1) == '\n' && *(p + 2) == '\r' && *(p + 3) == '\n')) {
        event_end = p;
        break;
      }
    }

    if (!event_end) {
#if DEBUG
      printf("[Event Check] Found 'data:' but no end marker, remaining data: %zu bytes\n",
             (size_t)(mem->stream.buffer + mem->stream.len - event_start));
#endif
      break;
    }

    // 提取事件内容
    size_t event_length = event_end - event_start;
    char *json_start = event_start + 5;  // Skip "data:"
    size_t json_length = event_length - 5;
#if DEBUG
    printf("[Event Found] Start:%ld End:%ld Length:%zu\nContent: %.*s\n",
           event_start - mem->stream.buffer, event_end - mem->stream.buffer, json_length,
           (int)json_length > 200 ? 200 : (int)json_length, json_start);
#endif

    // JSON结构完整性检查（增强版）
    if (json_length > 0) {
      int brace_level = 0;
      int is_valid = 1;
      for (size_t i = 0; i < json_length; i++) {
        if (json_start[i] == '{') brace_level++;
        if (json_start[i] == '}') brace_level--;
        // 如果遇到换行符提前终止
        if (json_start[i] == '\n' || json_start[i] == '\r') {
          is_valid = 0;
          break;
        }
      }

      if (brace_level == 0 && is_valid) {
        cJSON *chunk = cJSON_ParseWithLength(json_start, json_length);
        if (chunk) {
          // 内容提取（增加null检查）
          cJSON *choices = cJSON_GetObjectItem(chunk, "choices");
          if (choices && cJSON_IsArray(choices)) {
            cJSON *first_choice = cJSON_GetArrayItem(choices, 0);
            if (first_choice) {
              cJSON *delta = cJSON_GetObjectItem(first_choice, "delta");
              if (delta) {
                cJSON *content = cJSON_GetObjectItem(delta, "content");
                if (content && cJSON_IsString(content) && content->valuestring) {
#if DEBUG
                  printf("[Content Output] %s\n", content->valuestring);  // 调试输出
#endif
                  printf("%s", content->valuestring);
                  fflush(stdout);
                }
              }
            }
          }
          cJSON_Delete(chunk);
        } else {
#if DEBUG
          printf("[JSON Error] At position %ld-%ld: %.*s\n", json_start - mem->stream.buffer,
                 json_start - mem->stream.buffer + json_length,
                 (int)json_length > 100 ? 100 : (int)json_length, json_start);
#endif
        }
      } else {
#if DEBUG
        printf("[JSON Structure Invalid] Brace level:%d Valid:%d\n", brace_level, is_valid);
#endif
        break;  // 保留数据等待后续分片
      }
    }

    current_ptr = event_end + (strncmp(event_end, "\n\n", 2) ? 4 : 2);
  }

  // 4. 缓冲区滑动（增加边界检查）
  size_t processed = current_ptr - mem->stream.buffer;
  if (processed > 0) {
#if DEBUG
    printf("[Buffer Sliding] Processed:%zu Remaining:%zu\n", processed,
           mem->stream.len - processed);
#endif

    memmove(mem->stream.buffer, current_ptr, mem->stream.len - processed);
    mem->stream.len -= processed;
    mem->stream.buffer[mem->stream.len] = '\0';
  } else {
#if DEBUG
    printf("[Buffer Sliding] No data processed, keeping all\n");
#endif
  }

  return realsize;
}

CURLcode fetch_direct(CURL *curl, const char *api_key, const char *post_data) {
  CURLcode res;
  struct curl_slist *headers = NULL;
  struct curl_slist *resolve = NULL;

  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);

  // 构建强制解析列表
  resolve = curl_slist_append(resolve, "www.sophnet.com:443:39.103.239.196");

  // 设置请求头
  headers = curl_slist_append(headers, "Host: www.sophnet.com");
  headers = curl_slist_append(headers, "Content-Type: application/json");
  char auth_header[128];
  snprintf(auth_header, sizeof(auth_header), "Authorization: Bearer %s", api_key);
  headers = curl_slist_append(headers, auth_header);

  // 核心配置
  curl_easy_setopt(curl, CURLOPT_URL, "https://www.sophnet.com/api/open-apis/chat/completions");
  curl_easy_setopt(curl, CURLOPT_RESOLVE, resolve);
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, post_data);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)strlen(post_data));

  // 增强SSL配置
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
  curl_easy_setopt(curl, CURLOPT_SSLVERSION, CURL_SSLVERSION_TLSv1_2);
  // curl_easy_setopt(curl, CURLOPT_SSL_SNI, "www.sophnet.com");

  // 网络配置
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 15L);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
  curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);

  // 每次请求后主动关闭连接
  curl_easy_setopt(curl, CURLOPT_FORBID_REUSE, 1L);

// 调试配置
#if DEBUG
  curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);  // 启用详细日志
#else
  curl_easy_setopt(curl, CURLOPT_VERBOSE, 0L);  // 关闭详细日志
#endif

#if DEBUG
  printf("start to curl_easy_perform\n");
#endif

  res = curl_easy_perform(curl);

#if DEBUG
  printf("end curl_easy_perform\n");
#endif

  // 后置清理
  curl_slist_free_all(headers);
  curl_slist_free_all(resolve);
  return res;
}

int main(void) {
  CURL *curl;
  CURLcode res;

  // 消息历史记录
  Message *message_history = NULL;
  size_t message_count = 0;

  printf("开始使用deepseek\n");

  const char *api_key =
      "_jorSEWcc1_s0-veCFy7przZXbenJQssoe_Z4XqZiaTaVbDl7BGuvZy13qRi6Q8b4vOlbG2LCrOnsoUuwPjHhA";
  if (!api_key) {
    fprintf(stderr, "Error: DEEPSEEK_API_KEY environment variable not set\n");
    return 1;
  }

  curl_global_init(CURL_GLOBAL_ALL);
  curl = curl_easy_init();
  if (!curl) {
    fprintf(stderr, "Error initializing CURL\n");
    goto cleanup;
  }

  // 配置终端为非规范模式
  struct termios oldt, newt;
  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);  // 禁用规范模式和回显
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);

  setlocale(LC_ALL, "en_US.UTF-8");

  // 添加循环控制变量
  int keep_running = 1;

  while (keep_running) {
    // 获取用户输入
    printf("\n请输入你的问题: \n");

    InputState state = {.input_buf = {0}, .pos = 0, .utf8 = {.bytes = {0}, .len = 0, .wc = 0}};

    memset(state.input_buf, 0, INPUT_MAX);

    while (1) {
      char c = getchar();
      // printf("%c", (unsigned char)c);
      // fflush(stdout);

      if (c == 127 || c == 8) {
        handle_backspace(&state);
        state.utf8.len = 0;  // 明确重置状态
      } else if (c == '\n') {
        printf("\n");
        // printf("\nFinal: %s\n", state.input_buf);
        break;
      } else {
        if (collect_utf8(&state, c)) {
          if (state.pos + state.utf8.len < INPUT_MAX) {
            memcpy(state.input_buf + state.pos, state.utf8.bytes, state.utf8.len);
            state.pos += state.utf8.len;
            printf("%s", state.utf8.bytes);
            fflush(stdout);
          }
          state.utf8.len = 0;  // 显式清理
        }
      }
    }

    // 检查退出条件
    if (strcmp(state.input_buf, "exit") == 0) {
      keep_running = 0;
      continue;
    }

#if DEBUG
    printf("user_input data: %.*s\n", 500, state.input_buf);
#endif

    // 添加用户消息到历史记录
    Message user_msg;
    strncpy(user_msg.role, "user", sizeof(user_msg.role));
    user_msg.content = strdup(state.input_buf);
    message_history = realloc(message_history, (message_count + 1) * sizeof(Message));
    if (!message_history) {
      fprintf(stderr, "内存分配失败\n");
      exit(1);
    }
    message_history[message_count++] = user_msg;

    // 构建请求体
    cJSON *root = cJSON_CreateObject();

    // 在构建请求JSON时添加stream字段
    cJSON_AddTrueToObject(root, "stream");  // 添加此行
    cJSON_AddStringToObject(root, "model", "spn3/DeepSeek-chat");

    cJSON *messages = cJSON_CreateArray();
    for (size_t i = 0; i < message_count; i++) {
      cJSON *msg = cJSON_CreateObject();
      cJSON_AddStringToObject(msg, "role", message_history[i].role);
      cJSON_AddStringToObject(msg, "content", message_history[i].content);
      cJSON_AddItemToArray(messages, msg);
    }
    cJSON_AddItemToObject(root, "messages", messages);

    char *post_data = cJSON_PrintUnformatted(root);

#if DEBUG
    printf("post data: %.*s\n", 500, post_data);  // 打印前500字节
#endif

    // 每次请求前重置句柄状态
    curl_easy_reset(curl);  // 关键！清除上次请求的配置

    // 初始化缓冲区（重要！）
    struct MemoryChunk chunk = {.data = NULL, .size = 0, .stream = {.buffer = NULL, .len = 0}};
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &chunk);

    res = fetch_direct(curl, api_key, post_data);

    // 释放请求相关内存
    free(post_data);
    cJSON_Delete(root);

    if (res != CURLE_OK) {
      fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
    } else {
      printf("Received %zu bytes\n", chunk.size);

      if (chunk.data) {
#if DEBUG
// printf("Response: %.*s\n", 500, chunk.data); // 打印前500字节
#endif

        free(chunk.data);
        free(chunk.stream.buffer);
        chunk.size = 0;
      }
    }
  }

  // 恢复终端设置
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);

cleanup:
  // 释放消息历史
  for (size_t i = 0; i < message_count; i++) {
    free(message_history[i].content);
  }
  free(message_history);

  if (curl) curl_easy_cleanup(curl);
  curl_global_cleanup();
  return 0;
}