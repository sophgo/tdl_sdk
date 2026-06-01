#ifndef MEDIA_ANALYSIS_SERVER_HPP
#define MEDIA_ANALYSIS_SERVER_HPP

#include <libwebsockets.h>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "media_analysis_task.hpp"

class FaceRegistrationTask;

class MediaAnalysisServer {
 public:
  static MediaAnalysisServer* GetInstance();

  // Initialize server with port
  void init();
  void parse_config(const std::string& config_path);
  void stop();

  // Send image to web client
  void send_image_to_web_client(const std::vector<uint8_t>& image_data,
                                uint64_t timestamp, int channel_id,
                                uint64_t frame_id,
                                const std::string& metadata_json = "");

  std::string get_data_path() const { return data_path_; }
  std::string get_model_dir() const { return model_dir_; }

  // Send message to cloud client (exposed for BehaviorAnalysisTask)
  void send_to_cloud_client(const std::string& msg);

  // Send message to web client (exposed for task push events)
  void send_to_web_client(const std::string& msg);

  // LWS Callback (must be public or static friend)
  static int callback_http(struct lws* wsi, enum lws_callback_reasons reason,
                           void* user, void* in, size_t len);

  static int callback_media_analysis(struct lws* wsi,
                                     enum lws_callback_reasons reason,
                                     void* user, void* in, size_t len);

 private:
  MediaAnalysisServer();
  ~MediaAnalysisServer();

  struct lws_context* context_ = nullptr;
  int port_;
  std::thread service_thread_;
  bool is_running_ = false;
  std::string data_path_;
  std::string model_dir_;

  // Connections
  std::mutex conn_mutex_;
  struct lws* web_client_wsi_ = nullptr;
  struct lws* cloud_client_wsi_ = nullptr;

  // Output queues for thread-safe writing
  std::mutex queue_mutex_;
  std::queue<std::string> web_client_outbox_;
  std::queue<std::string> cloud_client_outbox_;

  // Message buffers for fragmented WebSocket messages
  std::map<struct lws*, std::string> message_buffers_;
  std::mutex buffer_mutex_;

  // HTTP POST body accumulation
  std::map<struct lws*, std::string> http_post_bodies_;
  std::map<struct lws*, std::string> http_post_responses_;
  std::mutex http_mutex_;

  // Face registration task
  std::shared_ptr<FaceRegistrationTask> face_registration_task_;
};

#endif  // MEDIA_ANALYSIS_SERVER_HPP
