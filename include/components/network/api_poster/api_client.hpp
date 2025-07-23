#ifndef API_CLIENT_API_CLIENT_HPP
#define API_CLIENT_API_CLIENT_HPP

#include <curl/curl.h>
#include <json.hpp>
#include <string>
#include <vector>

namespace APIClient {
// chatresponse结构体封装API调用结果
struct ChatResponse {
  bool success;               // 操作是否成功的标志
  std::string error_message;  // 失败时的错误信息
  std::string content;  // 成功时返回内容(chat时为文本，voice时为文件路径)
  ChatResponse() : success(false) {}  // 默认构造函数将success初始化为false
};

class CommonFunctions {
 public:
  static size_t WriteCallback(void *contents, size_t size, size_t nmemb,
                              std::string *response);
  static std::string escapeJson(const std::string &str);
  static std::string extractContent(const std::string &json);
  static std::string encodeBase64(const std::string &data);
  static std::string decodeBase64(const std::string &data);
  // 图片加载
  static std::string loadImageAsBase64(const std::string &file_path);
  bool loadBase64AsImage(const std::string &base64_str,
                         const std::string &output_path);
  static std::string base64ToDataURI(
      const std::string &base64_str,
      const std::string
          &image_path);  // 用于将base64数据改为url格式，主要用于阿里云的模型
  // 获取当前 UTC ISO8601 时间戳，用于火山模型的签名生成
  static std::string getISO8601Time();
  bool downloadImage(const std::string &url, const std::string &outputPath);
  static size_t WriteFileCallback(void *ptr, size_t size, size_t nmemb,
                                  void *stream);
};

// openaiclient类负责与ai服务通信
class VolcengineClient {
 private:
  std::string buildStylePayload(
      const std::string &req_key, const std::string &sub_req_key,
      const std::string &img_base64) const;  // AIGC-图像风格化
  std::string buildbackgroundPayload(
      const std::string &ref_prompt,
      const std::string &img_base64) const;  // AIGC-主体保持
  std::string buildpicturetopicturePayload(
      const std::string &ref_prompt,
      const std::string &img_base64) const;  // 实时图生图
  std::string buildhumansegmentPayload(
      CURL *curl, const std::string &img_base64) const;  // 人像抠图
  std::string buildhumanagetransPayload(
      const int &target_age,
      const std::string &img_base64) const;  // 人像年龄转换
  static std::string UrlEncode(CURL *curl, const std::string &s);

 public:
  ChatResponse stylizeImage(const std::string &ak, const std::string &sk,
                            const std::string &req_key,
                            const std::string &sub_req_key,
                            const std::string &image_path,
                            const std::string &output_path);  // AIGC-图像风格化
  ChatResponse backgroundchange(
      const std::string &ak, const std::string &sk,
      const std::string &ref_prompt, const std::string &image_path,
      const std::string &output_path);  // AIGC-主体保持
  ChatResponse picturetopicture(const std::string &ak, const std::string &sk,
                                const std::string &ref_prompt,
                                const std::string &image_path,
                                const std::string &output_path);  // 实时图生图
  ChatResponse humansegment(const std::string &ak, const std::string &sk,
                            const std::string &image_path,
                            const std::string &output_path);  // 人像抠图
  ChatResponse humanagetrans(const std::string &ak, const std::string &sk,
                             const int &target_age,
                             const std::string &image_path,
                             const std::string &output_path);  // 人像年龄转换
};

class SophnetClient {
 private:
  // 文本、图片payload构建
  std::string buildTextPayload(const std::string &message) const;
  std::string buildImagePayload(const std::string &text,
                                const std::string &image_data) const;

 public:
  ChatResponse chat(const std::string &api_key,
                    const std::string &textdomain);  // 文本交流-科普内容生成
  ChatResponse analyzeImage(
      const std::string &api_key, const std::string &text,
      const std::string &image_path);  // 分析图片-万物识别
};

class AliyunClient {
 private:
  //     std::string buildpictureeditorPayload(const std::string &api_key, const
  //     std::string &model, const std::string &img_base64) const; //
  //     通用图像编辑
  ChatResponse createTask(const std::string &api_key,
                          const std::string &function,
                          const std::string &image_path,
                          const std::string &ref_prompt);
  ChatResponse pollTask(const std::string &api_key, const std::string &task_id,
                        const std::string &output_path, int max_attempts = 10,
                        int interval_secs = 2);

 public:
  ChatResponse imgeditor(
      const std::string &api_key, const std::string &function,
      const std::string &image_path, const std::string &output_path,
      const std::string
          &ref_prompt);  // 通用图像编辑，选择不同的function对应不同的功能
};
}  // namespace APIClient

#endif