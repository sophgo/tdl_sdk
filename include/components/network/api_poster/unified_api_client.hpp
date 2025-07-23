#ifndef UNIFIED_API_CLIENT_HPP
#define UNIFIED_API_CLIENT_HPP

#include <functional>
#include <json.hpp>
#include <memory>
#include <string>
#include <unordered_map>

#include "api_client.hpp"  // 新的集中式 api_client
#include "asr_client.hpp"
#include "tts_client.hpp"

using MethodFunc = std::function<nlohmann::json(const nlohmann::json &)>;

class UnifiedApiClient {
 public:
  UnifiedApiClient();
  ~UnifiedApiClient();

  // 统一调用接口：clientType 如 "sophnet"/"volcengine"/"aliyun"
  nlohmann::json call(const std::string &clientType,
                      const std::string &methodName,
                      const nlohmann::json &params);

  // 判断是否注册了某个 clientType
  bool isClientInitialized(const std::string &clientType) const;

 private:
  // 实例化 client
  void initClients();
  // 注册所有方法到 methodMap
  void registerMethods();
  // 统一错误格式
  nlohmann::json createErrorResponse(const std::string &errorMsg) const;

  // 存放各个 Client 实例
  std::unique_ptr<APIClient::SophnetClient> sophnetClient;
  std::unique_ptr<APIClient::VolcengineClient> volcengineClient;
  std::unique_ptr<APIClient::AliyunClient> aliyunClient;
  // std::unique_ptr<Asr::AsrClient> asrClient;
  // std::unique_ptr<Tts::TtsClient> ttsClient;

  // 映射：clientType -> (methodName -> lambda 调用)
  std::unordered_map<std::string, std::unordered_map<std::string, MethodFunc>>
      methodMap;
};

#endif  // UNIFIED_API_CLIENT_HPP