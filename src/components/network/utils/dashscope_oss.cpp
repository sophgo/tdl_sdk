// dashscope_oss.cpp
#include "dashscope_oss.hpp"
#include <curl/curl.h>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <json.hpp>
#include <sstream>

using json = nlohmann::json;
namespace OSS {
// Callback for libcurl to write response data into a std::string
size_t OSSClient::WriteCallback(void *ptr, size_t size, size_t nmemb,
                                void *userdata) {
  auto realSize = size * nmemb;
  std::string *resp = static_cast<std::string *>(userdata);
  resp->append(static_cast<char *>(ptr), realSize);
  return realSize;
}

PolicyData OSSClient::getUploadPolicy(const std::string &apiKey,
                                      const std::string &modelName) {
  CURL *curl = curl_easy_init();
  if (!curl) {
    throw std::runtime_error("Failed to init CURL");
  }

  std::string baseUrl = "https://dashscope.aliyuncs.com/api/v1/uploads";
  // url escape modelName
  char *esc = curl_easy_escape(curl, modelName.c_str(), (int)modelName.size());
  std::string fullUrl = baseUrl + "?action=getPolicy&model=" + esc;
  curl_free(esc);

  struct curl_slist *headers = nullptr;
  headers =
      curl_slist_append(headers, ("Authorization: Bearer " + apiKey).c_str());
  headers = curl_slist_append(headers, "Content-Type: application/json");

  std::string resp;
  curl_easy_setopt(curl, CURLOPT_URL, fullUrl.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

  CURLcode code = curl_easy_perform(curl);
  long status = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);

  if (code != CURLE_OK || status != 200) {
    std::ostringstream oss;
    oss << "getUploadPolicy failed: "
        << (code == CURLE_OK ? resp : curl_easy_strerror(code));
    throw std::runtime_error(oss.str());
  }

  auto j = json::parse(resp);
  if (!j.contains("data")) {
    throw std::runtime_error("Invalid JSON response, missing data");
  }
  auto d = j["data"];
  PolicyData policy{};
  policy.upload_host = d.at("upload_host").get<std::string>();
  policy.oss_access_key_id = d.at("oss_access_key_id").get<std::string>();
  policy.signature = d.at("signature").get<std::string>();
  policy.policy = d.at("policy").get<std::string>();
  policy.x_oss_object_acl = d.at("x_oss_object_acl").get<std::string>();
  policy.x_oss_forbid_overwrite =
      d.at("x_oss_forbid_overwrite").get<std::string>();
  policy.upload_dir = d.at("upload_dir").get<std::string>();
  return policy;
}

std::string OSSClient::uploadFileToOss(const PolicyData &policy,
                                       const std::string &filePath) {
  CURL *curl = curl_easy_init();
  if (!curl) {
    throw std::runtime_error("Failed to init CURL for upload");
  }

  // extract filename
  size_t pos = filePath.find_last_of("/\\");
  std::string filename =
      (pos == std::string::npos) ? filePath : filePath.substr(pos + 1);
  std::string key = policy.upload_dir + "/" + filename;

  curl_mime *mime = curl_mime_init(curl);

  auto addPart = [&](const char *name, const std::string &value) {
    curl_mimepart *part = curl_mime_addpart(mime);
    curl_mime_name(part, name);
    curl_mime_data(part, value.c_str(), CURL_ZERO_TERMINATED);
  };

  addPart("OSSAccessKeyId", policy.oss_access_key_id);
  addPart("Signature", policy.signature);
  addPart("policy", policy.policy);
  addPart("x-oss-object-acl", policy.x_oss_object_acl);
  addPart("x-oss-forbid-overwrite", policy.x_oss_forbid_overwrite);
  addPart("key", key);
  addPart("success_action_status", "200");

  // file part
  {
    curl_mimepart *part = curl_mime_addpart(mime);
    curl_mime_name(part, "file");
    curl_mime_filedata(part, filePath.c_str());
    curl_mime_filename(part, filename.c_str());
  }

  std::string resp;
  curl_easy_setopt(curl, CURLOPT_URL, policy.upload_host.c_str());
  curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

  CURLcode code = curl_easy_perform(curl);
  long status = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);

  curl_mime_free(mime);
  curl_easy_cleanup(curl);

  if (code != CURLE_OK || status != 200) {
    std::ostringstream oss;
    oss << "upload failed: "
        << (code == CURLE_OK ? resp : curl_easy_strerror(code));
    throw std::runtime_error(oss.str());
  }

  return "oss://" + key;
}

std::string OSSClient::uploadFileAndGetUrl(const std::string &apiKey,
                                           const std::string &modelName,
                                           const std::string &filePath) {
  auto policy = getUploadPolicy(apiKey, modelName);
  return uploadFileToOss(policy, filePath);
}
}  // namespace OSS
