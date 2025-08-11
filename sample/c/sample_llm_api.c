#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tdl_ex.h"
#include "tdl_sdk.h"

// 缓冲区大小定义
#define BUF_SIZE 4096

// 从文件中读取所有内容，返回 malloc 出来的字符串（末尾 '\0'），
// 成功时 *out_size 为文件长度，失败返回 NULL。
static char* read_file(const char* path, size_t* out_size) {
  FILE* fp = fopen(path, "rb");
  if (!fp) {
    return NULL;
  }
  if (fseek(fp, 0, SEEK_END) != 0) {
    fclose(fp);
    return NULL;
  }
  long len = ftell(fp);
  if (len < 0) {
    fclose(fp);
    return NULL;
  }
  rewind(fp);

  char* buf = (char*)malloc((size_t)len + 1);
  if (!buf) {
    fclose(fp);
    return NULL;
  }
  size_t read_bytes = fread(buf, 1, (size_t)len, fp);
  fclose(fp);

  if (read_bytes != (size_t)len) {
    free(buf);
    return NULL;
  }
  buf[len] = '\0';
  if (out_size) {
    *out_size = (size_t)len;
  }
  return buf;
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    fprintf(stderr,
            "用法: %s <client_type> <method_name> <params_json|json_file>\n",
            argv[0]);
    fprintf(stderr,
            "示例1: %s sophnet chat "
            "'{\"api_key\":\"your_key\",\"text\":\"Hello\"}'\n",
            argv[0]);
    fprintf(stderr, "示例2: %s sophnet chat ./params.json\n", argv[0]);
    return 1;
  }

  // 1. 创建主上下文
  TDLHandle api_handle = TDL_CreateHandleEx(0);  // 传入合适的 device id
  if (!api_handle) {
    fprintf(stderr, "创建API上下文失败\n");
    return 1;
  }

  // 2. 准备参数
  const char* client_type = argv[1];
  const char* method_name = argv[2];
  const char* params_arg = argv[3];
  char* params_json = NULL;
  int ret;
  char result_buf[BUF_SIZE];

  // 如果第三个参数是一个可读文件，就读取文件内容
  size_t json_len = 0;
  params_json = read_file(params_arg, &json_len);
  if (params_json) {
    // printf("检测到文件，已读取 %zu 字节 JSON 数据\n", json_len);
  } else {
    // 不是文件或读取失败，就当作 JSON 字符串
    params_json = (char*)params_arg;
  }

  // 3. 调用API
  ret = TDL_LLMApiCall(api_handle, client_type, method_name, params_json,
                       result_buf, BUF_SIZE);

  // 如果我们 malloc 了 params_json，需要释放
  if (params_json != params_arg) {
    free(params_json);
  }

  // 4. 处理结果
  if (ret == 0) {
    printf("调用成功:\n%s\n", result_buf);
  } else if (ret == -1) {
    fprintf(stderr, "参数错误\n");
  } else if (ret == -2) {
    fprintf(stderr, "缓冲区大小不足\n");
  } else if (ret == -3) {
    fprintf(stderr, "调用失败: %s\n", result_buf);
  } else {
    fprintf(stderr, "未知返回码: %d\n", ret);
  }

  // 5. 释放资源
  TDL_DestroyHandleEx(api_handle);
  return ret;
}
