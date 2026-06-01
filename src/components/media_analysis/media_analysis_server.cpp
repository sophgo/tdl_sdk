#include "components/media_analysis/media_analysis_server.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <json.hpp>

// Include task headers
#include "tasks/behavior_analysis/behavior_analysis_task.hpp"
#include "tasks/face/face_matching_task.hpp"
#include "tasks/face/face_registration_task.hpp"
#include "tasks/identity/identity_browse_task.hpp"
#include "tasks/image_analysis/image_analysis_task.hpp"
#include "tasks/image_text/image_text_task.hpp"
#include "tasks/reports/reports_task.hpp"
#include "tasks/statistics/statistics_task.hpp"

#include "components/media_analysis/media_analysis_event_manager.hpp"
#include "smart_home_db.hpp"

using json = nlohmann::json;

#define RX_BUFFER_BYTES (2 * 1024 * 1024)  // 2MB缓冲区

static const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

static std::string base64_encode(unsigned char const* bytes_to_encode,
                                 unsigned int in_len) {
  std::string ret;
  int i = 0;
  int j = 0;
  unsigned char char_array_3[3];
  unsigned char char_array_4[4];

  while (in_len--) {
    char_array_3[i++] = *(bytes_to_encode++);
    if (i == 3) {
      char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
      char_array_4[1] =
          ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
      char_array_4[2] =
          ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
      char_array_4[3] = char_array_3[2] & 0x3f;

      for (i = 0; (i < 4); i++) ret += base64_chars[char_array_4[i]];
      i = 0;
    }
  }

  if (i) {
    for (j = i; j < 3; j++) char_array_3[j] = '\0';

    char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
    char_array_4[1] =
        ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
    char_array_4[2] =
        ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
    char_array_4[3] = char_array_3[2] & 0x3f;

    for (j = 0; (j < i + 1); j++) ret += base64_chars[char_array_4[j]];

    while ((i++ < 3)) ret += '=';
  }

  return ret;
}

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <map>
#include <sstream>

// Protocol list
static struct lws_protocols protocols[] = {
    {
        "http",
        MediaAnalysisServer::callback_http,
        0,
        RX_BUFFER_BYTES,
    },
    {
        "media-analysis-protocol",
        MediaAnalysisServer::callback_media_analysis,
        0,  // per session data size
        RX_BUFFER_BYTES,
    },
    {NULL, NULL, 0, 0}  // terminator
};

int MediaAnalysisServer::callback_http(struct lws* wsi,
                                       enum lws_callback_reasons reason,
                                       void* user, void* in, size_t len) {
  MediaAnalysisServer* server = MediaAnalysisServer::GetInstance();

  switch (reason) {
    case LWS_CALLBACK_HTTP: {
      char* uri = (char*)in;
      std::string uri_str(uri);
      std::cout << "HTTP Request: " << uri_str << std::endl;

      if (uri_str.find("/api/image_proxy") == 0) {
        std::string path_param = "";
        char buf[1024];
        const char* val =
            lws_get_urlarg_by_name(wsi, "path=", buf, sizeof(buf));
        if (val) {
          path_param = val;
          // Decode URL encoding (minimal)
          std::string decoded = "";
          for (size_t i = 0; i < path_param.length(); ++i) {
            if (path_param[i] == '%' && i + 2 < path_param.length()) {
              int value = 0;
              std::stringstream ss;
              ss << std::hex << path_param.substr(i + 1, 2);
              ss >> value;
              decoded += (char)value;
              i += 2;
            } else if (path_param[i] == '+') {
              decoded += ' ';
            } else {
              decoded += path_param[i];
            }
          }
          path_param = decoded;
        }

        if (path_param.empty()) {
          lws_return_http_status(wsi, HTTP_STATUS_BAD_REQUEST, NULL);
          return -1;
        }

        // Check file exists
        struct stat st;
        if (stat(path_param.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
          lws_return_http_status(wsi, HTTP_STATUS_NOT_FOUND, NULL);
          return -1;
        }

        // Serve file
        const char* cors_header = "Access-Control-Allow-Origin: *\r\n";
        if (lws_serve_http_file(wsi, path_param.c_str(), "image/jpeg",
                                cors_header, strlen(cors_header)) < 0) {
          return -1;
        }
        return 0;
      }

      if (uri_str.find("/api/video_proxy") == 0) {
        std::string path_param = "";
        char buf[1024];
        const char* val =
            lws_get_urlarg_by_name(wsi, "path=", buf, sizeof(buf));
        if (val) {
          path_param = val;
          std::string decoded = "";
          for (size_t i = 0; i < path_param.length(); ++i) {
            if (path_param[i] == '%' && i + 2 < path_param.length()) {
              int value = 0;
              std::stringstream ss;
              ss << std::hex << path_param.substr(i + 1, 2);
              ss >> value;
              decoded += (char)value;
              i += 2;
            } else if (path_param[i] == '+') {
              decoded += ' ';
            } else {
              decoded += path_param[i];
            }
          }
          path_param = decoded;
        }

        if (path_param.empty()) {
          lws_return_http_status(wsi, HTTP_STATUS_BAD_REQUEST, NULL);
          return -1;
        }

        struct stat st;
        if (stat(path_param.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
          lws_return_http_status(wsi, HTTP_STATUS_NOT_FOUND, NULL);
          return -1;
        }

        const char* cors_header = "Access-Control-Allow-Origin: *\r\n";
        if (lws_serve_http_file(wsi, path_param.c_str(),
                                "application/octet-stream", cors_header,
                                strlen(cors_header)) < 0) {
          return -1;
        }
        return 0;
      }

      if (uri_str.find("/api/register_face") == 0) {
        // Check if CORS preflight (OPTIONS request)
        int options_len = lws_hdr_total_length(wsi, WSI_TOKEN_OPTIONS_URI);
        if (options_len > 0) {
          // Respond with CORS headers
          unsigned char buf[512];
          unsigned char* p = buf;
          p += snprintf((char*)p, sizeof(buf) - (p - buf),
                        "HTTP/1.1 200 OK\r\n"
                        "Access-Control-Allow-Origin: *\r\n"
                        "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n"
                        "Access-Control-Allow-Headers: Content-Type\r\n"
                        "Content-Length: 0\r\n"
                        "\r\n");
          unsigned char* out_buf = (unsigned char*)malloc(LWS_PRE + (p - buf));
          if (out_buf) {
            memcpy(out_buf + LWS_PRE, buf, p - buf);
            lws_write(wsi, out_buf + LWS_PRE, p - buf, LWS_WRITE_HTTP_HEADERS);
            free(out_buf);
          }
          if (lws_http_transaction_completed(wsi)) return -1;
          return 0;
        }

        // POST request: initialize body buffer
        {
          std::lock_guard<std::mutex> lock(server->http_mutex_);
          server->http_post_bodies_[wsi] = "";
        }
        // Return 0 to continue receiving body
        return 0;
      }

      lws_return_http_status(wsi, HTTP_STATUS_NOT_FOUND, NULL);
      return -1;
    }

    case LWS_CALLBACK_HTTP_BODY: {
      if (!in || len == 0) break;
      std::lock_guard<std::mutex> lock(server->http_mutex_);
      auto it = server->http_post_bodies_.find(wsi);
      if (it != server->http_post_bodies_.end()) {
        it->second.append((char*)in, len);
      }
      break;
    }

    case LWS_CALLBACK_HTTP_BODY_COMPLETION: {
      std::string body;
      {
        std::lock_guard<std::mutex> lock(server->http_mutex_);
        auto it = server->http_post_bodies_.find(wsi);
        if (it != server->http_post_bodies_.end()) {
          body = std::move(it->second);
          server->http_post_bodies_.erase(it);
        }
      }

      if (!body.empty()) {
        std::string response_json;
        try {
          json j = json::parse(body);
          std::string image_b64 = j.value("image", "");
          std::string name = j.value("name", "");
          bool force = j.value("force", false);

          if (server->face_registration_task_) {
            json result = server->face_registration_task_->registerFace(
                image_b64, name, force);
            response_json = result.dump();
          } else {
            json err;
            err["success"] = false;
            err["error_code"] = "INTERNAL_ERROR";
            err["error_message"] = "Face registration service not available";
            response_json = err.dump();
          }
        } catch (const std::exception& e) {
          json err;
          err["success"] = false;
          err["error_code"] = "INTERNAL_ERROR";
          err["error_message"] = std::string("Invalid request: ") + e.what();
          response_json = err.dump();
        }

        {
          std::lock_guard<std::mutex> lock(server->http_mutex_);
          server->http_post_responses_[wsi] = response_json;
        }
        lws_callback_on_writable(wsi);
      }
      break;
    }

    case LWS_CALLBACK_HTTP_WRITEABLE: {
      std::string response_json;
      {
        std::lock_guard<std::mutex> lock(server->http_mutex_);
        auto it = server->http_post_responses_.find(wsi);
        if (it != server->http_post_responses_.end()) {
          response_json = std::move(it->second);
          server->http_post_responses_.erase(it);
        }
      }

      if (!response_json.empty()) {
        unsigned char header_buf[512];
        unsigned char* p = header_buf;
        unsigned char* end = header_buf + sizeof(header_buf);
        if (lws_add_http_common_headers(wsi, HTTP_STATUS_OK, "application/json",
                                        response_json.size(), &p, end) < 0) {
          if (lws_http_transaction_completed(wsi)) return -1;
          return 0;
        }
        int ret = lws_add_http_header_by_name(
            wsi,
            (unsigned char*)"Access-Control-Allow-Origin:", (unsigned char*)"*",
            1, &p, end);
        (void)ret;

        int header_len = p - header_buf;
        int total_len = header_len + response_json.size();
        unsigned char* out_buf = (unsigned char*)malloc(LWS_PRE + total_len);
        if (out_buf) {
          memcpy(out_buf + LWS_PRE, header_buf, header_len);
          memcpy(out_buf + LWS_PRE + header_len, response_json.c_str(),
                 response_json.size());
          lws_write(wsi, out_buf + LWS_PRE, total_len,
                    LWS_WRITE_HTTP_HEADERS_CONTINUATION);
          free(out_buf);
        }

        if (lws_http_transaction_completed(wsi)) return -1;
      }
      return 0;
    }

    case LWS_CALLBACK_HTTP_FILE_COMPLETION: {
      if (lws_http_transaction_completed(wsi)) {
        return -1;
      }
      return 0;
    }

    case LWS_CALLBACK_CLOSED_HTTP: {
      std::lock_guard<std::mutex> lock(server->http_mutex_);
      server->http_post_bodies_.erase(wsi);
      server->http_post_responses_.erase(wsi);
      break;
    }

    default:
      break;
  }
  return 0;
}

MediaAnalysisServer* MediaAnalysisServer::GetInstance() {
  static MediaAnalysisServer instance;
  return &instance;
}

MediaAnalysisServer::MediaAnalysisServer() : port_(8000) {}

MediaAnalysisServer::~MediaAnalysisServer() { stop(); }

void MediaAnalysisServer::init() {
  struct lws_context_creation_info info;
  memset(&info, 0, sizeof(info));

  info.port = port_;
  info.protocols = protocols;
  info.gid = -1;
  info.uid = -1;
  // info.options = LWS_SERVER_OPTION_DO_SSL_GLOBAL_INIT; // If SSL is needed

  context_ = lws_create_context(&info);
  if (!context_) {
    std::cerr << "Failed to create libwebsockets context" << std::endl;
    return;
  }

  is_running_ = true;

  // Register identity browse task (always available)
  auto browse_task = std::make_shared<IdentityBrowseTask>();
  MediaAnalysisEventManager::GetInstance()->RegisterTask(browse_task);

  // Start service thread
  service_thread_ = std::thread([this]() {
    std::cout << "MediaAnalysisServer (LWS) started on port " << port_
              << std::endl;
    while (is_running_) {
      lws_service(context_, 50);
    }
  });
}

void MediaAnalysisServer::stop() {
  is_running_ = false;

  if (service_thread_.joinable()) {
    service_thread_.join();
  }

  if (context_) {
    lws_context_destroy(context_);
    context_ = nullptr;
  }
}

void MediaAnalysisServer::parse_config(const std::string& config_path) {
  std::ifstream config_file(config_path);
  if (!config_file.is_open()) {
    std::cerr << "Cannot open config file: " << config_path << std::endl;
    return;
  }

  json config;
  try {
    config_file >> config;

    // Parse port from config
    if (config.contains("port")) {
      port_ = config["port"];
      std::cout << "Port set to: " << port_ << " from config" << std::endl;
    }

    // Parse paths
    std::string data_path = config.value("data_path", "");
    data_path_ = data_path;
    std::string model_dir = config.value("model_dir", "");
    if (model_dir.empty()) model_dir = config.value("model_dir_", "");
    model_dir_ = model_dir;

    std::string txt_dir = config.value("txt_dir", "");
    if (txt_dir.empty()) txt_dir = config.value("txt_dir_", "");

    // Register Face Matching Task
    if (!data_path.empty()) {
      auto face_task = std::make_shared<FaceMatchingTask>(data_path);
      MediaAnalysisEventManager::GetInstance()->RegisterTask(face_task);
      std::cout << "Face Matching Task registered." << std::endl;
    }

    // Register Face Registration Task
    if (!data_path.empty() && !model_dir.empty()) {
      face_registration_task_ =
          std::make_shared<FaceRegistrationTask>(model_dir, data_path);
      MediaAnalysisEventManager::GetInstance()->RegisterTask(
          face_registration_task_);
      std::cout << "Face Registration Task registered." << std::endl;
    }

    // Register Behavior Analysis Task
    auto behavior_task = std::make_shared<BehaviorAnalysisTask>();
    MediaAnalysisEventManager::GetInstance()->RegisterTask(behavior_task);
    std::cout << "Behavior Analysis Task registered." << std::endl;

    // Register Image Text Task
    if (!data_path.empty() && !model_dir.empty() && !txt_dir.empty()) {
      auto image_text_task =
          std::make_shared<ImageTextTask>(data_path, model_dir, txt_dir);
      MediaAnalysisEventManager::GetInstance()->RegisterTask(image_text_task);
      std::cout << "Image Text Matching Task registered." << std::endl;
    }

    // Always register Image Analysis Task
    auto analysis_task = std::make_shared<ImageAnalysisTask>(data_path);
    MediaAnalysisEventManager::GetInstance()->RegisterTask(analysis_task);
    std::cout << "Image Analysis Task registered." << std::endl;

    // Initialize SmartHomeDB
    if (!data_path.empty()) {
      std::string db_path = data_path + "/smart_home.db";
      if (SmartHomeDB::GetInstance()->init(db_path)) {
        std::cout << "SmartHomeDB initialized at " << db_path << std::endl;

        // Register Statistics Task (Feat-1)
        auto stats_task = std::make_shared<StatisticsTask>();
        MediaAnalysisEventManager::GetInstance()->RegisterTask(stats_task);
        std::cout << "Statistics Task registered." << std::endl;

        // Register Reports Task (Feat-11)
        auto reports_task = std::make_shared<ReportsTask>();
        MediaAnalysisEventManager::GetInstance()->RegisterTask(reports_task);
        std::cout << "Reports Task registered." << std::endl;
      } else {
        std::cerr << "Failed to initialize SmartHomeDB" << std::endl;
      }
    }

  } catch (const std::exception& e) {
    std::cerr << "Config file parsing error: " << e.what() << std::endl;
  }
}

void MediaAnalysisServer::send_to_web_client(const std::string& msg) {
  std::lock_guard<std::mutex> lock(queue_mutex_);
  web_client_outbox_.push(msg);
  if (web_client_wsi_) {
    lws_callback_on_writable(web_client_wsi_);
  }
}

void MediaAnalysisServer::send_to_cloud_client(const std::string& msg) {
  std::lock_guard<std::mutex> lock(queue_mutex_);
  cloud_client_outbox_.push(msg);
  if (cloud_client_wsi_) {
    lws_callback_on_writable(cloud_client_wsi_);
  }
}

int MediaAnalysisServer::callback_media_analysis(
    struct lws* wsi, enum lws_callback_reasons reason, void* user, void* in,
    size_t len) {
  MediaAnalysisServer* server = MediaAnalysisServer::GetInstance();

  switch (reason) {
    case LWS_CALLBACK_ESTABLISHED: {
      std::cout << "New connection established" << std::endl;
      // 初始化消息缓冲区
      std::lock_guard<std::mutex> lock(server->buffer_mutex_);
      server->message_buffers_[wsi] = "";
      break;
    }

    case LWS_CALLBACK_RECEIVE: {
      std::string message_to_process;
      {
        std::lock_guard<std::mutex> lock(server->buffer_mutex_);
        std::string& buffer = server->message_buffers_[wsi];

        // 追加接收到的数据
        buffer.append((char*)in, len);

        // Check if message is complete
        if (lws_remaining_packet_payload(wsi) == 0 &&
            lws_is_final_fragment(wsi)) {
          message_to_process = buffer;
          buffer.clear();
        }
      }

      if (message_to_process.empty()) {
        // Message incomplete, wait for more data
        break;
      }

      // 处理完整消息
      try {
        json j = json::parse(message_to_process);

        // Registration
        if (j.contains("task_type") || j.contains("client_type")) {
          std::string type = j.value("client_type", "");
          if (type.empty()) type = j.value("task_type", "");

          std::lock_guard<std::mutex> lock(server->conn_mutex_);
          if (type == "web_client" || type == "image_and_text_matching" ||
              type == "face_matching") {
            server->web_client_wsi_ = wsi;
            std::cout << "Web Client registered (type: " << type
                      << ") wsi: " << wsi << " on server: " << server
                      << std::endl;
            // Handle as request if it's not just a registration handshake
            if (type != "web_client") {
              // It's a request, fall through to process?
            }
          } else if (type == "cloud_client" || type == "image_analysis") {
            server->cloud_client_wsi_ = wsi;
            std::cout << "Cloud Client registered (type: " << type << ")"
                      << std::endl;
            // Image analysis loop disabled — replaced by event-driven
            // BehaviorAnalysisTask
            return 0;
          }
        }

        // Handle Messages
        bool is_web_client = (wsi == server->web_client_wsi_);
        bool is_cloud_client = (wsi == server->cloud_client_wsi_);

        // If not registered yet, try to guess or just register implicitly
        if (!is_web_client && !is_cloud_client) {
          // Logic to auto-register based on content?
        }

        if (is_web_client) {
          // Delegate to Event Manager
          json response =
              MediaAnalysisEventManager::GetInstance()->HandleEvent(j);
          if (!response.empty()) {
            server->send_to_web_client(response.dump());
          }

        } else if (is_cloud_client) {
          // Forward Cloud Client message to Web Client
          server->send_to_web_client(message_to_process);
        }

      } catch (const std::exception& e) {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
      }
      break;
    }

    case LWS_CALLBACK_SERVER_WRITEABLE: {
      std::lock_guard<std::mutex> lock(server->queue_mutex_);

      if (wsi == server->web_client_wsi_ &&
          !server->web_client_outbox_.empty()) {
        std::string msg = server->web_client_outbox_.front();
        server->web_client_outbox_.pop();

        unsigned char* buf = (unsigned char*)malloc(LWS_PRE + msg.size());
        if (buf) {
          memcpy(buf + LWS_PRE, msg.c_str(), msg.size());
          lws_write(wsi, buf + LWS_PRE, msg.size(), LWS_WRITE_TEXT);
          free(buf);
        }

        // If more messages, schedule another write
        if (!server->web_client_outbox_.empty()) {
          lws_callback_on_writable(wsi);
        }
      }

      if (wsi == server->cloud_client_wsi_ &&
          !server->cloud_client_outbox_.empty()) {
        std::string msg = server->cloud_client_outbox_.front();
        server->cloud_client_outbox_.pop();

        unsigned char* buf = (unsigned char*)malloc(LWS_PRE + msg.size());
        if (buf) {
          memcpy(buf + LWS_PRE, msg.c_str(), msg.size());
          lws_write(wsi, buf + LWS_PRE, msg.size(), LWS_WRITE_TEXT);
          free(buf);
        }

        if (!server->cloud_client_outbox_.empty()) {
          lws_callback_on_writable(wsi);
        }
      }
      break;
    }

    case LWS_CALLBACK_CLOSED: {
      std::lock_guard<std::mutex> lock(server->conn_mutex_);
      // 清理消息缓冲区
      {
        std::lock_guard<std::mutex> buffer_lock(server->buffer_mutex_);
        server->message_buffers_.erase(wsi);
      }

      if (wsi == server->web_client_wsi_) {
        server->web_client_wsi_ = nullptr;
        std::cout << "Web Client disconnected" << std::endl;

        // 清理旧的待发送队列
        std::lock_guard<std::mutex> queue_lock(server->queue_mutex_);
        std::queue<std::string> empty_queue;
        std::swap(server->web_client_outbox_, empty_queue);

      } else if (wsi == server->cloud_client_wsi_) {
        server->cloud_client_wsi_ = nullptr;
        std::cout << "Cloud Client disconnected" << std::endl;

        // 清理旧的待发送队列
        std::lock_guard<std::mutex> queue_lock(server->queue_mutex_);
        std::queue<std::string> empty_queue;
        std::swap(server->cloud_client_outbox_, empty_queue);
      }
      break;
    }

    default:
      break;
  }

  return 0;
}

void MediaAnalysisServer::send_image_to_web_client(
    const std::vector<uint8_t>& image_data, uint64_t timestamp, int channel_id,
    uint64_t frame_id, const std::string& metadata_json) {
  // Optimize: if no client, don't encode
  {
    std::lock_guard<std::mutex> lock(conn_mutex_);
    if (!web_client_wsi_) {
      return;
    }
  }

  std::string base64_img = base64_encode(image_data.data(), image_data.size());

  // Format frame_id as 8-digit string
  char frame_id_str[16];
  snprintf(frame_id_str, sizeof(frame_id_str), "%08" PRIu64, frame_id);

  json j;
  j["timestamp"] = timestamp;
  j["camera_channel"] = std::to_string(channel_id);
  j["payload"] = {
      {"event", "frame"}, {"frame_id", frame_id_str}, {"image", base64_img}};

  if (!metadata_json.empty()) {
    try {
      json metadata = json::parse(metadata_json);
      j["payload"]["metadata"] = metadata;

      if (metadata.contains("width") && metadata.contains("height")) {
        j["payload"]["width"] = metadata["width"];
        j["payload"]["height"] = metadata["height"];
      }
    } catch (const std::exception& e) {
      std::cerr << "Failed to parse metadata JSON: " << e.what() << std::endl;
    }
  }

  send_to_web_client(j.dump());
}
