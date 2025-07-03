// dashscope_oss.h
#pragma once

#include <stdexcept>
#include <string>

namespace OSS {
struct PolicyData {
  std::string upload_host;
  std::string oss_access_key_id;
  std::string signature;
  std::string policy;
  std::string x_oss_object_acl;
  std::string x_oss_forbid_overwrite;
  std::string upload_dir;
};
class OSSClient {
 public:
  /**
   * @brief 获取文件上传凭证
   * @param apiKey       DashScope API Key
   * @param modelName    模型名称，如 "qwen-vl-plus"
   * @return PolicyData  包含 OSS 上传所需的各项字段
   * @throws std::runtime_error on HTTP or JSON error
   */
  static PolicyData getUploadPolicy(const std::string &apiKey,
                                    const std::string &modelName);

  /**
   * @brief 将文件上传到 OSS
   * @param policy     从 getUploadPolicy 返回的 PolicyData
   * @param filePath   本地文件路径
   * @return std::string OSS 内部路径，例如 "oss://upload_dir/filename"
   * @throws std::runtime_error on HTTP 上传失败
   */
  static std::string uploadFileToOss(const PolicyData &policy,
                                     const std::string &filePath);

  /**
   * @brief 组合接口：获取上传凭证 + 上传文件 + 返回 oss:// URL
   * @param apiKey     DashScope API Key
   * @param modelName  模型名称
   * @param filePath   本地文件路径
   * @return std::string oss:// URL
   */
  static std::string uploadFileAndGetUrl(const std::string &apiKey,
                                         const std::string &modelName,
                                         const std::string &filePath);

  static size_t WriteCallback(void *ptr, size_t size, size_t nmemb,
                              void *userdata);
};
}  // namespace OSS