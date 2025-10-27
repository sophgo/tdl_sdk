#include "api_client.hpp"
#include <curl/curl.h>
#include <chrono>
#include <cstring>  // for std::memset
#include <fstream>
#include <iomanip>
#include <iostream>
#include <json.hpp>
#include <regex>
#include <sstream>
#include <thread>
#include "utils/dashscope_oss.hpp"
#include "utils/volcengine_signer.hpp"

namespace APIClient {
size_t CommonFunctions::WriteCallback(void *contents, size_t size, size_t nmemb,
                                      std::string *response) {
  size_t total_size = size * nmemb;
  response->append((char *)contents, total_size);
  return total_size;
}

std::string CommonFunctions::escapeJson(const std::string &str) {
  std::string escaped;
  escaped.reserve(str.size() * 2);

  for (char c : str) {
    switch (c) {
      case '"':
        escaped += "\\\"";
        break;
      case '\\':
        escaped += "\\\\";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped += c;
        break;
    }
  }
  return escaped;
}

std::string CommonFunctions::extractContent(const std::string &json) {
  size_t choices_pos = json.find("\"choices\"");
  if (choices_pos == std::string::npos) return "";

  size_t message_pos = json.find("\"message\"", choices_pos);
  if (message_pos == std::string::npos) return "";

  size_t content_pos = json.find("\"content\"", message_pos);
  if (content_pos == std::string::npos) return "";

  size_t start = json.find("\"", content_pos + 9);
  if (start == std::string::npos) return "";
  start++;

  size_t end = start;
  while (end < json.length() &&
         !(json[end] == '"' && (end == 0 || json[end - 1] != '\\'))) {
    end++;
  }

  if (end >= json.length()) return "";

  std::string content = json.substr(start, end - start);

  // 简单反转义
  std::string result;
  for (size_t i = 0; i < content.length(); i++) {
    if (content[i] == '\\' && i + 1 < content.length()) {
      switch (content[i + 1]) {
        case 'n':
          result += '\n';
          i++;
          break;
        case 't':
          result += '\t';
          i++;
          break;
        case 'r':
          result += '\r';
          i++;
          break;
        case '"':
          result += '"';
          i++;
          break;
        case '\\':
          result += '\\';
          i++;
          break;
        default:
          result += content[i];
          break;
      }
    } else {
      result += content[i];
    }
  }

  return result;
}

std::string CommonFunctions::encodeBase64(const std::string &data) {
  const std::string chars =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string result;

  for (size_t i = 0; i < data.length(); i += 3) {
    unsigned char a = data[i];
    unsigned char b = (i + 1 < data.length()) ? data[i + 1] : 0;
    unsigned char c = (i + 2 < data.length()) ? data[i + 2] : 0;

    unsigned int combined = (a << 16) | (b << 8) | c;

    result += chars[(combined >> 18) & 0x3F];
    result += chars[(combined >> 12) & 0x3F];
    result += (i + 1 < data.length()) ? chars[(combined >> 6) & 0x3F] : '=';
    result += (i + 2 < data.length()) ? chars[combined & 0x3F] : '=';
  }

  return result;
}

std::string CommonFunctions::decodeBase64(const std::string &input) {
  static int T[256];
  static bool initialized = false;
  if (!initialized) {
    std::memset(T, -1, sizeof(T));
    for (int i = 0; i < 64; i++) {
      T["ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[i]] =
          i;
    }
    T['='] = 0;
    initialized = true;
  }
  std::string out;
  int val = 0, valb = -8;
  for (unsigned char c : input) {
    if (T[c] < 0) continue;
    val = (val << 6) + T[c];
    valb += 6;
    if (valb >= 0) {
      out.push_back(char((val >> valb) & 0xFF));
      valb -= 8;
    }
  }
  return out;
}

std::string CommonFunctions::loadImageAsBase64(const std::string &file_path) {
  std::ifstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    return "";
  }

  std::string data((std::istreambuf_iterator<char>(file)),
                   std::istreambuf_iterator<char>());
  file.close();

  return encodeBase64(data);
}
bool CommonFunctions::loadBase64AsImage(const std::string &base64_str,
                                        const std::string &output_path) {
  // 处理单个 Base64 字符串（无需遍历数组）
  // 1. 移除 data URI 前缀（如有）
  auto comma_pos = base64_str.find(',');
  std::string pure_base64;
  if (comma_pos != std::string::npos &&
      base64_str.substr(0, comma_pos).find("base64") != std::string::npos) {
    pure_base64 = base64_str.substr(comma_pos + 1);
  } else {
    pure_base64 = base64_str;
  }

  // 2. 解码 Base64
  std::string decoded_data = decodeBase64(pure_base64);
  if (decoded_data.empty()) {
    std::cerr << "Base64 解码失败\n";
    return false;
  }

  // 3. 写入文件
  std::ofstream ofs(output_path, std::ios::binary);
  if (!ofs) {
    std::cerr << "无法打开文件: " << output_path << std::endl;
    return false;
  }
  ofs.write(reinterpret_cast<const char *>(decoded_data.data()),
            decoded_data.size());
  return ofs.good();
}
std::string CommonFunctions::base64ToDataURI(const std::string &base64_str,
                                             const std::string &image_path) {
  // 检查输入是否为空
  if (base64_str.empty()) {
    std::cerr << "输入 Base64 字符串为空\n";
    return "";
  }

  // 从文件路径推断 MIME 类型
  std::string mime_type = "image/jpeg";  // 默认类型

  if (!image_path.empty()) {
    // 查找文件扩展名
    size_t ext_pos = image_path.find_last_of('.');
    if (ext_pos != std::string::npos) {
      std::string extension = image_path.substr(ext_pos + 1);

      // 转换为小写以进行不区分大小写的匹配
      std::transform(extension.begin(), extension.end(), extension.begin(),
                     ::tolower);

      // 根据扩展名设置 MIME 类型
      if (extension == "jpg" || extension == "jpeg") {
        mime_type = "image/jpeg";
      } else if (extension == "png") {
        mime_type = "image/png";
      } else if (extension == "gif") {
        mime_type = "image/gif";
      } else if (extension == "bmp") {
        mime_type = "image/bmp";
      } else if (extension == "webp") {
        mime_type = "image/webp";
      } else if (extension == "svg") {
        mime_type = "image/svg+xml";
      }
      // 可以根据需要添加更多类型
    }
  }
  // 构建完整的 data URI
  return "data:" + mime_type + ";base64," + base64_str;
}

std::string CommonFunctions::getISO8601Time() {
  // 返回 20250528T074152Z 这种格式
  auto t = std::time(nullptr);
  std::tm gm{};
  gmtime_r(&t, &gm);
  std::ostringstream ss;
  ss << std::put_time(&gm, "%Y%m%dT%H%M%SZ");
  return ss.str();
}

size_t CommonFunctions::WriteFileCallback(void *ptr, size_t size, size_t nmemb,
                                          void *stream) {
  std::ofstream *out = static_cast<std::ofstream *>(stream);
  size_t written = size * nmemb;
  out->write(reinterpret_cast<char *>(ptr), written);
  return written;
}

bool CommonFunctions::downloadImage(const std::string &url,
                                    const std::string &outputPath) {
  // 1. 打开文件
  std::ofstream outFile(outputPath, std::ios::binary);
  if (!outFile.is_open()) {
    std::cerr << "无法创建文件: " << outputPath << std::endl;
    return false;
  }

  // 2. init curl
  CURL *curl = curl_easy_init();
  if (!curl) {
    std::cerr << "CURL 初始化失败" << std::endl;
    return false;
  }

  // 3. 自定义写回调，把数据直接写入 ofstream
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteFileCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &outFile);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);  // 可选：超时
  // curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);  // 可选：跟随重定向

  // 4. 执行下载
  CURLcode res = curl_easy_perform(curl);
  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

  // 5. 清理
  curl_easy_cleanup(curl);
  outFile.close();

  // 6. 检查结果
  if (res != CURLE_OK) {
    std::cerr << "下载失败: " << curl_easy_strerror(res) << " (HTTP "
              << http_code << ")\n";
    return false;
  }
  if (http_code < 200 || http_code >= 300) {
    std::cerr << "HTTP 错误: " << http_code << std::endl;
    return false;
  }

  // std::cout << "图片已成功下载到: " << outputPath << std::endl;
  return true;
}

std::string VolcengineClient::UrlEncode(CURL *curl, const std::string &s) {
  char *out = curl_easy_escape(curl, s.c_str(), (int)s.size());
  std::string ret(out);
  curl_free(out);
  return ret;
}

std::string VolcengineClient::buildhumansegmentPayload(
    CURL *curl, const std::string &img_base64) const {
  std::ostringstream o;
  o << "image_base64="
    << UrlEncode(curl, img_base64)  // 把 +/= 等字符转 %2B/%3D/%2F……
    << "&refine=0"
    << "&return_foreground_image=1";
  return o.str();
}

std::string VolcengineClient::buildStylePayload(
    const std::string &req_key, const std::string &sub_req_key,
    const std::string &img_base64) const {
  std::ostringstream o;
  o << "{"
    // 1. req_key
    << "\"req_key\":\"" << CommonFunctions::escapeJson(req_key) << "\","
    << "\"sub_req_key\":\"" << CommonFunctions::escapeJson(sub_req_key)
    << "\","
    // 2. return_url
    << "\"return_url\":false,"
    // 3. binary_data_base64 数组
    << "\"binary_data_base64\":[\"" << CommonFunctions::escapeJson(img_base64)
    << "\"],"
    // 4. logo_info 对象
    << "\"logo_info\":{"
    << "\"add_logo\":true,"
    << "\"position\":0,"
    << "\"language\":0,"
    // 如果不需要 opacity 可以删掉下一行
    // << "\"opacity\":0.3,"
    << "\"logo_text_content\":\""
    << CommonFunctions::escapeJson("这里是明水印内容") << "\""
    << "}"
    << "}";

  return o.str();
}

std::string VolcengineClient::buildbackgroundPayload(
    const std::string &ref_prompt, const std::string &img_base64) const {
  std::ostringstream o;
  o << "{"
    // 1. req_key
    << "\"req_key\":\""
    << "img2img_bgpaint_light"
    << "\","
    // 2. binary_data_base64 为空数组
    << "\"binary_data_base64\":[\"" << CommonFunctions::escapeJson(img_base64)
    << "\"],"
    // 3. seed
    << "\"seed\":-1,"
    // 4. blend
    << "\"blend\":0.5,"
    // 5. prompt
    << "\"prompt\":\"" << CommonFunctions::escapeJson(ref_prompt)
    << "\","
    // 6. num_steps
    << "\"num_steps\":25,"
    // 7. return_url
    << "\"return_url\":false,"
    // 8. logo_info 对象
    << "\"logo_info\":{"
    << "\"add_logo\":false,"  // 不添加水印
    << "\"position\":0,"
    << "\"language\":0,"
    << "\"logo_text_content\":\""
    << CommonFunctions::escapeJson("这里是明水印内容") << "\""
    << "}"
    << "}";

  return o.str();
}

std::string VolcengineClient::buildpicturetopicturePayload(
    const std::string &ref_prompt, const std::string &img_base64) const {
  std::ostringstream o;
  o << "{"
    // 1. req_key
    << "\"req_key\":\""
    << "img2img_ai_doodle_dreamina"
    << "\","
    // 2. binary_data_base64 为空数组
    << "\"binary_data_base64\":[\"" << CommonFunctions::escapeJson(img_base64)
    << "\"],"
    // 5. prompt
    << "\"prompt\":\"" << CommonFunctions::escapeJson(ref_prompt)
    << "\","
    //   << "\"cfg\":8.0,"
    //   << "\"strength\":0.3,"
    // 7. return_url
    << "\"return_url\":false,"
    // 8. logo_info 对象
    << "\"logo_info\":{"
    << "\"add_logo\":false,"  // 不添加水印
    << "\"position\":0,"
    << "\"language\":0,"
    << "\"logo_text_content\":\""
    << CommonFunctions::escapeJson("这里是明水印内容") << "\""
    << "}"
    << "}";

  return o.str();
}

std::string VolcengineClient::buildhumanagetransPayload(
    const int &target_age, const std::string &img_base64) const {
  std::ostringstream o;
  o << "{"
    // 1. req_key
    << "\"req_key\":\""
    << "all_age_generation"
    << "\","
    // 2. binary_data_base64 为空数组
    << "\"binary_data_base64\":[\"" << CommonFunctions::escapeJson(img_base64)
    << "\"],"
    // 5. prompt
    << "\"target_age\":" << target_age
    << ","
    //   << "\"cfg\":8.0,"
    //   << "\"strength\":0.3,"
    // 7. return_url
    << "\"return_url\":false,"
    // 8. logo_info 对象
    << "\"logo_info\":{"
    << "\"add_logo\":false,"  // 不添加水印
    << "\"position\":0,"
    << "\"language\":0,"
    << "\"logo_text_content\":\""
    << CommonFunctions::escapeJson("这里是明水印内容") << "\""
    << "}"
    << "}";

  return o.str();
}

ChatResponse VolcengineClient::stylizeImage(const std::string &ak,
                                            const std::string &sk,
                                            const std::string &req_key,
                                            const std::string &sub_req_key,
                                            const std::string &image_path,
                                            const std::string &output_path) {
  ChatResponse result;

  // 0. 获取 AK/SK
  if (ak.empty() || sk.empty()) {
    throw std::runtime_error("Environment variables VOLC_AK/VOLC_SK not set");
  }

  // 1. 加载并 Base64 编码
  std::string img_base64 = CommonFunctions::loadImageAsBase64(image_path);
  if (img_base64.empty()) {
    result.error_message = "Failed to load image: " + image_path;
    return result;
  }

  // 2. 构建请求体
  std::string body = buildStylePayload(req_key, sub_req_key, img_base64);

  // 3. 准备签名器（注意：VolcengineSigner 构造函数只接受 ak, sk, region）
  const char *host = "visual.volcengineapi.com";
  std::string region = "cn-beijing";
  VolcengineSigner signer(ak, sk, region);

  // 4. 计算 body 的 SHA256 十六进制摘要
  std::string content_sha256 =
      VolcengineSigner::HexEncode(VolcengineSigner::Hash(body));

  // 5. 构造 QueryString 和 Headers
  std::map<std::string, std::string> qs = {{"Action", "AIGCStylizeImage"},
                                           {"Version", "2024-06-06"}};

  // 注意 Header 的 key 要小写
  std::string date =
      CommonFunctions::getISO8601Time();  // e.g. 20250529T151700Z
  std::map<std::string, std::string> sign_headers = {
      {"host", host},
      {"content-type", "application/json"},
      {"x-content-sha256", content_sha256},
      {"x-date", date}};

  // 6. 生成 Authorization header
  std::string auth_header = signer.Sign("POST", qs, sign_headers, body);

  // // DEBUG 输出
  // std::cerr << "------------ DEBUG START ------------\n"
  //           << "[Auth Header]\n"
  //           << auth_header << "\n\n"
  //           << "[Request URL]\n"
  //           << "https://" << host <<
  //           "/?Action=AIGCStylizeImage&Version=2024-06-06\n\n"
  //           << "[Request Headers]\n"
  //           << "Authorization: " << auth_header << "\n"
  //           << "Host: " << host << "\n"
  //           << "Content-Type: application/json\n"
  //           << "X-Content-Sha256: " << content_sha256 << "\n"
  //           << "X-Date: " << date << "\n\n"
  //           << "[Request Body]\n"
  //           << body << "\n"
  //           << "------------- DEBUG END -------------\n\n";

  // 7. 发起 HTTP 请求
  CURL *curl = curl_easy_init();
  if (!curl) {
    result.error_message = "curl init failed";
    return result;
  }

  std::string url = "https://" + std::string(host) +
                    "/?Action=AIGCStylizeImage&Version=2024-06-06";
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

  struct curl_slist *hdrs = nullptr;
  hdrs = curl_slist_append(hdrs, ("Authorization: " + auth_header).c_str());
  hdrs = curl_slist_append(hdrs, ("Host: " + std::string(host)).c_str());
  hdrs = curl_slist_append(hdrs, "Content-Type: application/json");
  hdrs =
      curl_slist_append(hdrs, ("X-Content-Sha256: " + content_sha256).c_str());
  hdrs = curl_slist_append(hdrs, ("X-Date: " + date).c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdrs);

  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, CommonFunctions::WriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &result.content);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

  CURLcode res = curl_easy_perform(curl);
  long http_code = 0;
  if (res == CURLE_OK) {
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
  } else {
    result.success = false;
    result.error_message = curl_easy_strerror(res);
  }

  curl_slist_free_all(hdrs);
  curl_easy_cleanup(curl);

  if (http_code == 200) {
    result.success = true;
    auto j = nlohmann::json::parse(result.content);

    // 2. 定位到 binary_data_base64 数组
    auto &arr = j["Result"]["data"]["binary_data_base64"];
    // std::cout << "提取的数据为: " << arr << std::endl;
    if (!arr.is_array() || arr.empty()) {
      // std::cerr << "no binary_data_base64 found or array is empty\n";
      result.success = false;
      result.error_message = "no binary_data_base64 found or array is empty";
      return result;
    }
    std::string base64Str = arr[0].get<std::string>();
    CommonFunctions commonFuncs;
    if (commonFuncs.loadBase64AsImage(base64Str, output_path)) {
      // std::cout << "风格化图片已保存至: " << output_path << std::endl;
      result.success = true;
      result.content = output_path;
      return result;
    } else {
      // std::cerr << "保存图片失败\n";
      result.success = false;
      result.error_message = "Failed to save image from base64 data";
      return result;
    }
  } else {
    result.error_message =
        "HTTP " + std::to_string(http_code) + ": " + result.content;
  }
  return result;
}

ChatResponse VolcengineClient::backgroundchange(
    const std::string &ak, const std::string &sk, const std::string &ref_prompt,
    const std::string &image_path, const std::string &output_path) {
  ChatResponse result;

  // 0. 获取 AK/SK
  if (ak.empty() || sk.empty()) {
    throw std::runtime_error("Environment variables VOLC_AK/VOLC_SK not set");
  }

  // 1. 加载并 Base64 编码
  std::string img_base64 = CommonFunctions::loadImageAsBase64(image_path);
  if (img_base64.empty()) {
    result.error_message = "Failed to load image: " + image_path;
    return result;
  }

  // 2. 构建请求体
  std::string body = buildbackgroundPayload(ref_prompt, img_base64);

  // 3. 准备签名器（注意：VolcengineSigner 构造函数只接受 ak, sk, region）
  const char *host = "visual.volcengineapi.com";
  std::string region = "cn-north-1";
  VolcengineSigner signer(ak, sk, region);

  // 4. 计算 body 的 SHA256 十六进制摘要
  std::string content_sha256 =
      VolcengineSigner::HexEncode(VolcengineSigner::Hash(body));

  // 5. 构造 QueryString 和 Headers
  std::map<std::string, std::string> qs = {{"Action", "CVProcess"},
                                           {"Version", "2022-08-31"}};

  // 注意 Header 的 key 要小写
  std::string date =
      CommonFunctions::getISO8601Time();  // e.g. 20250529T151700Z
  std::map<std::string, std::string> sign_headers = {
      {"host", host},
      {"content-type", "application/json"},
      {"x-content-sha256", content_sha256},
      {"x-date", date}};

  // 6. 生成 Authorization header
  std::string auth_header = signer.Sign("POST", qs, sign_headers, body);

  // 7. 发起 HTTP 请求
  CURL *curl = curl_easy_init();
  if (!curl) {
    result.error_message = "curl init failed";
    return result;
  }

  std::string url =
      "https://" + std::string(host) + "/?Action=CVProcess&Version=2022-08-31";
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

  struct curl_slist *hdrs = nullptr;
  hdrs = curl_slist_append(hdrs, ("Authorization: " + auth_header).c_str());
  hdrs = curl_slist_append(hdrs, ("Host: " + std::string(host)).c_str());
  hdrs = curl_slist_append(hdrs, "Content-Type: application/json");
  hdrs =
      curl_slist_append(hdrs, ("X-Content-Sha256: " + content_sha256).c_str());
  hdrs = curl_slist_append(hdrs, ("X-Date: " + date).c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdrs);

  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, CommonFunctions::WriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &result.content);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

  CURLcode res = curl_easy_perform(curl);
  long http_code = 0;
  if (res == CURLE_OK) {
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
  } else {
    result.error_message = curl_easy_strerror(res);
  }

  curl_slist_free_all(hdrs);
  curl_easy_cleanup(curl);

  if (http_code == 200) {
    result.success = true;
    auto j = nlohmann::json::parse(result.content);

    // 2. 定位到 binary_data_base64 数组
    auto &arr = j["data"]["binary_data_base64"];
    // std::cout << "提取的数据为: " << arr << std::endl;
    if (!arr.is_array() || arr.empty()) {
      // std::cerr << "no binary_data_base64 found or array is empty\n";
      result.success = false;
      result.error_message = "no binary_data_base64 found or array is empty";
      return result;
    }
    std::string base64Str = arr[0].get<std::string>();
    CommonFunctions commonFuncs;
    if (commonFuncs.loadBase64AsImage(base64Str, output_path)) {
      // std::cout << "背景生成图片已保存至: " << output_path << std::endl;
      result.success = true;
      result.content = output_path;
      return result;
    } else {
      // std::cerr << "保存图片失败\n";
      result.success = false;
      result.error_message = "Failed to save image from base64 data";
      return result;
    }
  } else {
    result.error_message =
        "HTTP " + std::to_string(http_code) + ": " + result.content;
  }
  return result;
}

ChatResponse VolcengineClient::picturetopicture(
    const std::string &ak, const std::string &sk, const std::string &ref_prompt,
    const std::string &image_path, const std::string &output_path) {
  ChatResponse result;

  // 0. 获取 AK/SK
  if (ak.empty() || sk.empty()) {
    throw std::runtime_error("Environment variables VOLC_AK/VOLC_SK not set");
  }

  // 1. 加载并 Base64 编码
  std::string img_base64 = CommonFunctions::loadImageAsBase64(image_path);
  if (img_base64.empty()) {
    result.error_message = "Failed to load image: " + image_path;
    return result;
  }

  // 2. 构建请求体
  std::string body = buildpicturetopicturePayload(ref_prompt, img_base64);

  // 3. 准备签名器（注意：VolcengineSigner 构造函数只接受 ak, sk, region）
  const char *host = "visual.volcengineapi.com";
  std::string region = "cn-north-1";
  VolcengineSigner signer(ak, sk, region);

  // 4. 计算 body 的 SHA256 十六进制摘要
  std::string content_sha256 =
      VolcengineSigner::HexEncode(VolcengineSigner::Hash(body));

  // 5. 构造 QueryString 和 Headers
  std::map<std::string, std::string> qs = {{"Action", "CVProcess"},
                                           {"Version", "2022-08-31"}};

  // 注意 Header 的 key 要小写
  std::string date =
      CommonFunctions::getISO8601Time();  // e.g. 20250529T151700Z
  std::map<std::string, std::string> sign_headers = {
      {"host", host},
      {"content-type", "application/json"},
      {"x-content-sha256", content_sha256},
      {"x-date", date}};

  // 6. 生成 Authorization header
  std::string auth_header = signer.Sign("POST", qs, sign_headers, body);

  // 7. 发起 HTTP 请求
  CURL *curl = curl_easy_init();
  if (!curl) {
    result.error_message = "curl init failed";
    return result;
  }

  std::string url =
      "https://" + std::string(host) + "/?Action=CVProcess&Version=2022-08-31";
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

  struct curl_slist *hdrs = nullptr;
  hdrs = curl_slist_append(hdrs, ("Authorization: " + auth_header).c_str());
  hdrs = curl_slist_append(hdrs, ("Host: " + std::string(host)).c_str());
  hdrs = curl_slist_append(hdrs, "Content-Type: application/json");
  hdrs =
      curl_slist_append(hdrs, ("X-Content-Sha256: " + content_sha256).c_str());
  hdrs = curl_slist_append(hdrs, ("X-Date: " + date).c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdrs);

  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, CommonFunctions::WriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &result.content);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);
  curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  curl_easy_setopt(curl, CURLOPT_IPRESOLVE, CURL_IPRESOLVE_V4);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

  CURLcode res = curl_easy_perform(curl);
  long http_code = 0;
  if (res == CURLE_OK) {
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
  } else {
    result.error_message = curl_easy_strerror(res);
  }

  curl_slist_free_all(hdrs);
  curl_easy_cleanup(curl);

  if (http_code == 200) {
    result.success = true;
    auto j = nlohmann::json::parse(result.content);

    // 2. 定位到 binary_data_base64 数组
    auto &arr = j["data"]["binary_data_base64"];
    // std::cout << "提取的数据为: " << arr << std::endl;
    if (!arr.is_array() || arr.empty()) {
      // std::cerr << "no binary_data_base64 found or array is empty\n";
      result.success = false;
      result.error_message = "no binary_data_base64 found or array is empty";
      return result;
    }
    std::string base64Str = arr[0].get<std::string>();
    CommonFunctions commonFuncs;
    if (commonFuncs.loadBase64AsImage(base64Str, output_path)) {
      // std::cout << "图片已保存至: " << output_path << std::endl;
      result.success = true;
      result.content = output_path;
      return result;
    } else {
      // std::cerr << "保存图片失败\n";
      result.success = false;
      result.error_message = "Failed to save image from base64 data";
      return result;
    }
  } else {
    result.error_message =
        "HTTP " + std::to_string(http_code) + ": " + result.content;
  }
  return result;
}

ChatResponse VolcengineClient::humansegment(const std::string &ak,
                                            const std::string &sk,
                                            const std::string &image_path,
                                            const std::string &output_path) {
  ChatResponse result;

  // 1. 加载并 Base64 编码
  std::string img_base64 = CommonFunctions::loadImageAsBase64(image_path);
  if (img_base64.empty()) {
    result.error_message = "Failed to load image: " + image_path;
    return result;
  }
  std::cerr << "Base64编码长度: " << img_base64.size() << std::endl;

  CURL *curl = curl_easy_init();
  if (!curl) {
    result.error_message = "curl init failed";
    return result;
  }
  std::string body = buildhumansegmentPayload(curl, img_base64);

  // 3. 准备签名器（注意：VolcengineSigner 构造函数只接受 ak, sk, region）
  const char *host = "visual.volcengineapi.com";
  std::string region = "cn-beijing";
  VolcengineSigner signer(ak, sk, region);

  // 4. 计算 body 的 SHA256 十六进制摘要
  std::string content_sha256 =
      VolcengineSigner::HexEncode(VolcengineSigner::Hash(body));

  // 5. 构造 QueryString 和 Headers
  std::map<std::string, std::string> qs = {{"Action", "HumanSegment"},
                                           {"Version", "2020-08-26"}};

  // 注意 Header 的 key 要小写
  std::string date =
      CommonFunctions::getISO8601Time();  // e.g. 20250529T151700Z
  std::map<std::string, std::string> sign_headers = {
      {"host", host},
      {"content-type", "application/x-www-form-urlencoded"},
      {"x-content-sha256", content_sha256},
      {"x-date", date}};

  // 6. 生成 Authorization header
  std::string auth_header = signer.Sign("POST", qs, sign_headers, body);

  std::string url = "https://" + std::string(host) +
                    "/?Action=HumanSegment&Version=2020-08-26";
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

  struct curl_slist *hdrs = nullptr;
  hdrs = curl_slist_append(hdrs, ("Authorization: " + auth_header).c_str());
  hdrs = curl_slist_append(hdrs, ("Host: " + std::string(host)).c_str());
  hdrs = curl_slist_append(hdrs,
                           "Content-Type: application/x-www-form-urlencoded");
  hdrs =
      curl_slist_append(hdrs, ("X-Content-Sha256: " + content_sha256).c_str());
  hdrs = curl_slist_append(hdrs, ("X-Date: " + date).c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdrs);

  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, CommonFunctions::WriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &result.content);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

  CURLcode res = curl_easy_perform(curl);
  long http_code = 0;
  if (res == CURLE_OK) {
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
  } else {
    result.error_message = curl_easy_strerror(res);
  }

  curl_slist_free_all(hdrs);
  curl_easy_cleanup(curl);

  if (http_code == 200) {
    result.success = true;
    auto j = nlohmann::json::parse(result.content);

    // 2. 定位到 binary_data_base64 数组
    auto &arr = j["data"]["foreground_image"];
    // std::cout << "提取的数据为: " << arr << std::endl;
    if (!arr.is_string() || arr.empty()) {
      // std::cerr << "no binary_data_base64 found or array is empty\n";
      result.success = false;
      result.error_message = "no binary_data_base64 found or array is empty";
      return result;
    }
    CommonFunctions commonFuncs;
    if (commonFuncs.loadBase64AsImage(arr, output_path)) {
      // std::cout << "图片已保存至: " << output_path << std::endl;
      result.success = true;
      result.content = output_path;
      return result;
    } else {
      // std::cerr << "保存图片失败\n";
      result.success = false;
      result.error_message = "Failed to save image from base64 data";
      return result;
    }
  } else {
    result.error_message =
        "HTTP " + std::to_string(http_code) + ": " + result.content;
  }
  return result;
}

ChatResponse VolcengineClient::humanagetrans(const std::string &ak,
                                             const std::string &sk,
                                             const int &target_age,
                                             const std::string &image_path,
                                             const std::string &output_path) {
  ChatResponse result;

  // 0. 获取 AK/SK
  if (ak.empty() || sk.empty()) {
    throw std::runtime_error("Environment variables VOLC_AK/VOLC_SK not set");
  }

  // 1. 加载并 Base64 编码
  std::string img_base64 = CommonFunctions::loadImageAsBase64(image_path);
  if (img_base64.empty()) {
    result.error_message = "Failed to load image: " + image_path;
    return result;
  }

  // 2. 构建请求体
  std::string body = buildhumanagetransPayload(target_age, img_base64);

  // 3. 准备签名器（注意：VolcengineSigner 构造函数只接受 ak, sk, region）
  const char *host = "visual.volcengineapi.com";
  std::string region = "cn-north-1";
  VolcengineSigner signer(ak, sk, region);

  // 4. 计算 body 的 SHA256 十六进制摘要
  std::string content_sha256 =
      VolcengineSigner::HexEncode(VolcengineSigner::Hash(body));

  // 5. 构造 QueryString 和 Headers
  std::map<std::string, std::string> qs = {{"Action", "CVProcess"},
                                           {"Version", "2022-08-31"}};

  // 注意 Header 的 key 要小写
  std::string date =
      CommonFunctions::getISO8601Time();  // e.g. 20250529T151700Z
  std::map<std::string, std::string> sign_headers = {
      {"host", host},
      {"content-type", "application/json"},
      {"x-content-sha256", content_sha256},
      {"x-date", date}};

  // 6. 生成 Authorization header
  std::string auth_header = signer.Sign("POST", qs, sign_headers, body);

  // 7. 发起 HTTP 请求
  CURL *curl = curl_easy_init();
  if (!curl) {
    result.error_message = "curl init failed";
    return result;
  }

  std::string url =
      "https://" + std::string(host) + "/?Action=CVProcess&Version=2022-08-31";
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

  struct curl_slist *hdrs = nullptr;
  hdrs = curl_slist_append(hdrs, ("Authorization: " + auth_header).c_str());
  hdrs = curl_slist_append(hdrs, ("Host: " + std::string(host)).c_str());
  hdrs = curl_slist_append(hdrs, "Content-Type: application/json");
  hdrs =
      curl_slist_append(hdrs, ("X-Content-Sha256: " + content_sha256).c_str());
  hdrs = curl_slist_append(hdrs, ("X-Date: " + date).c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdrs);

  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, CommonFunctions::WriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &result.content);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

  CURLcode res = curl_easy_perform(curl);
  long http_code = 0;
  if (res == CURLE_OK) {
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
  } else {
    result.error_message = curl_easy_strerror(res);
  }

  curl_slist_free_all(hdrs);
  curl_easy_cleanup(curl);

  if (http_code == 200) {
    result.success = true;
    auto j = nlohmann::json::parse(result.content);

    // 2. 定位到 binary_data_base64 数组
    auto &arr = j["data"]["binary_data_base64"];
    // std::cout << "提取的数据为: " << arr << std::endl;
    if (!arr.is_array() || arr.empty()) {
      // std::cerr << "no binary_data_base64 found or array is empty\n";
      result.success = false;
      result.error_message = "no binary_data_base64 found or array is empty";
      return result;
    }
    std::string base64Str = arr[0].get<std::string>();
    CommonFunctions commonFuncs;
    if (commonFuncs.loadBase64AsImage(base64Str, output_path)) {
      // std::cout << "图片已保存至: " << output_path << std::endl;
      result.success = true;
      result.content = output_path;
      return result;
    } else {
      // std::cerr << "保存图片失败\n";
      result.success = false;
      result.error_message = "Failed to save image from base64 data";
      return result;
    }
  } else {
    result.error_message =
        "HTTP " + std::to_string(http_code) + ": " + result.content;
  }
  return result;
}

std::string SophnetClient::buildTextPayload(const std::string &message) const {
  std::ostringstream json;
  json << "{"
       << "\"messages\":["
       // 如果需要 system，可以在这里插一条
       << "{\"role\":\"user\","
       << "\"content\":\"" << CommonFunctions::escapeJson(message) << "\"}"
       << "],"
       << "\"model\":\""
       << "DeepSeek-V3-Fast"
       << "\""
       << "}";
  return json.str();
}

std::string SophnetClient::buildImagePayload(
    const std::string &text, const std::string &image_data) const {
  std::ostringstream json;
  json << "{\n";
  json << "  \"messages\": [\n";
  json << "    {\n";
  json << "      \"role\": \"user\",\n";
  json << "      \"content\": [\n";
  json << "        {\"type\": \"text\", \"text\": \""
       << CommonFunctions::escapeJson(text) << "\"},\n";
  json << "        {\"type\": \"image_url\", \"image_url\": {\"url\": "
          "\"data:image/jpeg;base64,"
       << image_data << "\"}}\n";
  json << "      ]\n";
  json << "    }\n";
  json << "  ],\n";
  json << "  \"model\": \""
       << "Qwen2.5-VL-72B-Instruct"
       << "\"\n";
  json << "}";
  return json.str();
}

ChatResponse SophnetClient::chat(const std::string &api_key,
                                 const std::string &text) {
  ChatResponse result;

  CURL *curl = curl_easy_init();
  if (!curl) {
    result.error_message = "Failed to initialize CURL";
    return result;
  }
  std::string url = "https://www.sophnet.com/api/open-apis/v1/chat/completions";
  std::string response_data;
  std::string json_payload = buildTextPayload(text);

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_payload.c_str());

  struct curl_slist *headers = nullptr;
  std::string auth_header = "Authorization: Bearer " + api_key;
  headers = curl_slist_append(headers, auth_header.c_str());
  headers = curl_slist_append(headers, "Content-Type: application/json");
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, CommonFunctions::WriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

  CURLcode res = curl_easy_perform(curl);

  if (res != CURLE_OK) {
    result.error_message = curl_easy_strerror(res);
  } else {
    long response_code;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);

    if (response_code == 200) {
      result.content = CommonFunctions::extractContent(response_data);
      result.success = !result.content.empty();
      if (!result.success) {
        result.error_message = "Failed to parse response: " + response_data;
      }
    } else {
      result.error_message =
          "HTTP Error " + std::to_string(response_code) + ": " + response_data;
    }
  }

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);

  return result;
}

ChatResponse SophnetClient::analyzeImage(const std::string &api_key,
                                         const std::string &text,
                                         const std::string &image_path) {
  ChatResponse result;

  // 加载图片并转换为base64
  std::string image_data = CommonFunctions::loadImageAsBase64(image_path);
  if (image_data.empty()) {
    result.error_message = "Failed to load image file: " + image_path;
    return result;
  }
  // std::string image_uri =
  // CommonFunctions::base64ToDataURI(image_data,image_path);

  CURL *curl = curl_easy_init();
  if (!curl) {
    result.error_message = "Failed to initialize CURL";
    return result;
  }
  std::string response_data;
  std::string json_payload = buildImagePayload(text, image_data);
  std::string url = "https://www.sophnet.com/api/open-apis/v1/chat/completions";

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_payload.c_str());

  struct curl_slist *headers = nullptr;
  std::string auth_header = "Authorization: Bearer " + api_key;
  headers = curl_slist_append(headers, auth_header.c_str());
  headers = curl_slist_append(headers, "Content-Type: application/json");
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, CommonFunctions::WriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);  // 图片分析可能需要更长时间
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

  CURLcode res = curl_easy_perform(curl);

  if (res != CURLE_OK) {
    result.error_message = curl_easy_strerror(res);
  } else {
    long response_code;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);

    if (response_code == 200) {
      result.content = CommonFunctions::extractContent(response_data);
      result.success = !result.content.empty();
      if (!result.success) {
        result.error_message = "Failed to parse response: " + response_data;
      }
    } else {
      result.error_message =
          "HTTP Error " + std::to_string(response_code) + ": " + response_data;
    }
  }

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);

  return result;
}

ChatResponse AliyunClient::createTask(const std::string &api_key,
                                      const std::string &function,
                                      const std::string &image_path,
                                      const std::string &ref_prompt) {
  ChatResponse resp;
  std::string image_uri = OSS::OSSClient::uploadFileAndGetUrl(
      api_key, "wanx2.1-imageedit", image_path);
  CURL *curl = curl_easy_init();
  if (!curl) {
    resp.error_message = "createTask: curl_easy_init() failed";
    return resp;
  }
  // curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
  // URL + 请求体
  const std::string url =
      "https://dashscope.aliyuncs.com/api/v1/services/aigc/image2image/"
      "image-synthesis";
  nlohmann::json j = {{"model", "wanx2.1-imageedit"},
                      {"input",
                       {{"function", function},
                        {"prompt", ref_prompt},
                        {"base_image_url", image_uri}}},
                      {"parameters", {{"n", 1}}}};
  std::string post_data = j.dump();
  // std::cerr << "请求体: " << post_data << std::endl;
  // headers
  struct curl_slist *hdrs = nullptr;
  hdrs = curl_slist_append(hdrs, "X-DashScope-Async: enable");
  hdrs = curl_slist_append(hdrs, "X-DashScope-OssResourceResolve: enable");
  hdrs = curl_slist_append(hdrs, ("Authorization: Bearer " + api_key).c_str());
  hdrs = curl_slist_append(hdrs, "Content-Type: application/json");

  std::string body;
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdrs);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, post_data.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, CommonFunctions::WriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);

  CURLcode code = curl_easy_perform(curl);
  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

  curl_slist_free_all(hdrs);
  curl_easy_cleanup(curl);

  if (code != CURLE_OK) {
    resp.error_message =
        "createTask curl error=" + std::string(curl_easy_strerror(code)) +
        " HTTP=" + std::to_string(http_code) + " RESP=" + body;
    return resp;
  }
  if (http_code < 200 || http_code >= 300) {
    resp.error_message =
        "createTask non-2xx HTTP=" + std::to_string(http_code) +
        " RESP=" + body;
    return resp;
  }

  try {
    auto pj = nlohmann::json::parse(body);
    resp.content = pj.at("output").at("task_id").get<std::string>();
    resp.success = true;
  } catch (std::exception &e) {
    resp.error_message =
        "createTask parse JSON failed: " + std::string(e.what()) +
        " RESP=" + body;
  }
  return resp;
}
ChatResponse AliyunClient::pollTask(const std::string &api_key,
                                    const std::string &task_id,
                                    const std::string &output_path,
                                    int max_attempts, int interval_secs) {
  ChatResponse resp;
  CURL *curl = curl_easy_init();
  if (!curl) {
    resp.error_message = "pollTask: curl_easy_init() failed";
    return resp;
  }
  // curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
  const std::string base = "https://dashscope.aliyuncs.com/api/v1/tasks/";
  std::string url = base + task_id;

  for (int i = 1; i <= max_attempts; ++i) {
    std::this_thread::sleep_for(std::chrono::seconds(interval_secs));

    struct curl_slist *hdrs = nullptr;
    hdrs =
        curl_slist_append(hdrs, ("Authorization: Bearer " + api_key).c_str());

    std::string body;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdrs);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,
                     CommonFunctions::WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);

    CURLcode code = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    curl_slist_free_all(hdrs);

    if (code != CURLE_OK) {
      resp.error_message = "pollTask curl error #" + std::to_string(i) + "=" +
                           curl_easy_strerror(code) +
                           " HTTP=" + std::to_string(http_code) +
                           " RESP=" + body;
      break;
    }
    if (http_code < 200 || http_code >= 300) {
      resp.error_message = "pollTask non-2xx #" + std::to_string(i) +
                           " HTTP=" + std::to_string(http_code) +
                           " RESP=" + body;
      break;
    }

    try {
      auto pj = nlohmann::json::parse(body);
      std::string status = pj.at("output").at("task_status").get<std::string>();
      if (status == "SUCCEEDED") {
        auto &arr = pj.at("output").at("results");
        if (!arr.empty() && arr[0].contains("url")) {
          resp.content = arr[0].at("url").get<std::string>();
          resp.success = true;
          CommonFunctions commonFuncs;
          if (commonFuncs.loadBase64AsImage(resp.content, output_path)) {
            resp.content = output_path;
            return resp;
          } else {
            // std::cerr << "保存图片失败\n";
            resp.success = false;
            resp.error_message = "Failed to save image from base64 data";
            return resp;
          }
        } else {
          resp.error_message = "SUCCEEDED but no URL: " + body;
        }
        break;
      } else if (status == "FAILED" || status == "CANCELED") {
        resp.error_message = "Task " + status + ": " + body;
        break;
      }
      // 否则继续下一轮
    } catch (std::exception &e) {
      resp.error_message =
          "pollTask parse JSON failed: " + std::string(e.what()) +
          " RESP=" + body;
      break;
    }
  }

  curl_easy_cleanup(curl);

  if (!resp.success && resp.error_message.empty()) {
    resp.error_message = "Exceeded max polling attempts";
  }
  return resp;
}

// ==== 3. 主流程：按步骤调用 ====
ChatResponse AliyunClient::imgeditor(const std::string &api_key,
                                     const std::string &function,
                                     const std::string &image_path,
                                     const std::string &output_path,
                                     const std::string &ref_prompt) {
  // 全局 init（进程只需调用一次）
  curl_global_init(CURL_GLOBAL_ALL);

  // 1) 创建任务
  auto ct = createTask(api_key, function, image_path, ref_prompt);
  if (!ct.success) {
    curl_global_cleanup();
    return ct;  // error_message 已填，success=false
  }

  // 2) 轮询任务
  ChatResponse final_resp = pollTask(api_key, ct.content, output_path);

  // 全局 cleanup（进程结束前）
  curl_global_cleanup();
  return final_resp;
}
}  // namespace APIClient