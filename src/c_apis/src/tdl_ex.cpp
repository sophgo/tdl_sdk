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

#include "media_analysis/media_analysis_event_manager.hpp"
#include "media_analysis/media_analysis_server.hpp"
#include "media_analysis/tasks/behavior_analysis/behavior_analysis_task.hpp"
#include "media_analysis/tasks/face/face_matching_task.hpp"

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

int32_t TDL_MediaAnalysisServer_Init(const char *config_path) {
  if (config_path == nullptr) {
    LOGE("config_path is null");
    return -1;
  }
  MediaAnalysisServer::GetInstance()->parse_config(config_path);
  MediaAnalysisServer::GetInstance()->init();
  return 0;
}

int32_t TDL_MediaAnalysisServer_Stop() {
  MediaAnalysisServer::GetInstance()->stop();
  return 0;
}

int32_t TDL_MediaAnalysisServer_SendImage(const uint8_t *image_data,
                                          size_t size, uint64_t timestamp,
                                          int channel_id, uint64_t frame_id,
                                          const char *metadata_json) {
  if (image_data == nullptr || size == 0) {
    LOGE("Invalid image data");
    return -1;
  }
  std::vector<uint8_t> data(image_data, image_data + size);
  std::string metadata_str = (metadata_json != nullptr) ? metadata_json : "";
  MediaAnalysisServer::GetInstance()->send_image_to_web_client(
      data, timestamp, channel_id, frame_id, metadata_str);
  return 0;
}

int32_t TDL_MediaAnalysisServer_AddFaceInfo(int registered_id,
                                            int face_track_id) {
  auto task =
      MediaAnalysisEventManager::GetInstance()->GetTask("face_matching");
  if (!task) {
    LOGE("face_matching task not found");
    return -1;
  }
  auto face_task = std::dynamic_pointer_cast<FaceMatchingTask>(task);
  if (!face_task) {
    LOGE("face_matching task cast failed");
    return -1;
  }
  face_task->add_face_info(registered_id, face_track_id);
  return 0;
}

int32_t TDL_MediaAnalysisServer_SubmitBehaviorVideo(const char *video_path,
                                                    const char *person_name,
                                                    int person_id,
                                                    uint64_t appearance_id,
                                                    uint32_t duration_sec) {
  if (video_path == nullptr) {
    LOGE("video_path is null");
    return -1;
  }
  auto task =
      MediaAnalysisEventManager::GetInstance()->GetTask("behavior_analysis");
  if (!task) {
    LOGE("behavior_analysis task not found");
    return -1;
  }
  auto behavior_task = std::dynamic_pointer_cast<BehaviorAnalysisTask>(task);
  if (!behavior_task) {
    LOGE("behavior_analysis task cast failed");
    return -1;
  }
  std::string name_str = (person_name != nullptr) ? person_name : "";
  std::string person_id_str = (person_id > 0) ? std::to_string(person_id) : "";
  std::string appearance_id_str =
      (appearance_id > 0) ? std::to_string(appearance_id) : "";
  behavior_task->submitBehaviorVideo(video_path, name_str, person_id_str,
                                     appearance_id_str, duration_sec);
  return 0;
}
#endif
