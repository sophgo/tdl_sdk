#include "tdl_ex.h"
#include "tdl_sdk.h"

#include <opencv2/opencv.hpp>
#include "app/app_data_types.hpp"
#include "common/common_types.hpp"
#include "tdl_type_internal_ex.hpp"
#include "tdl_utils.h"
#include "utils/common_utils.hpp"
#include "utils/tdl_log.hpp"

#if !defined(__CMODEL_CV181X__) && !defined(__CMODEL_CV184X__)

TDLHandleEx TDL_CreateHandleEx(const int32_t tpu_device_id) {
  TDLContextEx *context = new TDLContextEx();
  return (TDLHandleEx)context;
}

int32_t TDL_LLMApiCall(TDLHandleEx handle, const char *client_type,
                       const char *method_name, const char *params_json,
                       char *result_buf, size_t buf_size) {
  if (!handle || !client_type || !method_name || !params_json || !result_buf ||
      buf_size == 0) {
    return -1;
  }

  try {
    TDLContextEx *context = (TDLContextEx *)handle;
    if (context->api_client == nullptr) {
      context->api_client = std::make_unique<UnifiedApiClient>();
      if (context->api_client == nullptr) {
        LOGE("Failed to create api client");
        return -1;
      }
    }
    nlohmann::json params = nlohmann::json::parse(params_json);
    nlohmann::json resp =
        context->api_client->call(client_type, method_name, params);
    std::string resp_str = resp.dump();

    if (resp_str.size() >= buf_size) {
      return -2;  // 缓冲区不足
    }
    strncpy(result_buf, resp_str.c_str(), buf_size - 1);
    result_buf[buf_size - 1] = '\0';
    return 0;
  } catch (const std::exception &e) {
    const char *err_msg = e.what();
    size_t err_len = strlen(err_msg);
    if (err_len >= buf_size) {
      err_len = buf_size - 1;
    }
    strncpy(result_buf, err_msg, err_len);
    result_buf[err_len] = '\0';
    return -3;  // 解析或调用出错
  }
}

int32_t TDL_DestroyHandleEx(TDLHandleEx handle) {
  TDLContextEx *context = (TDLContextEx *)handle;
  if (context == nullptr) {
    return -1;
  }
  for (auto &model : context->core_context.models) {
    TDL_CloseModel(handle, model.first);
  }
  if (context->core_context.app_task) {
    context->core_context.app_task->release();
  }
#if !defined(__CMODEL_CV181X__) && !defined(__CMODEL_CV184X__)
  if (context->api_client) {
    context->api_client.reset();
  }
#endif
  delete context;
  context = nullptr;
  return 0;
}
#endif
