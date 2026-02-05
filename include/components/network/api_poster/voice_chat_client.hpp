#pragma once

#include <libwebsockets.h>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <json.hpp>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include "audio/audio_capture.hpp"
#include "audio/audio_player.hpp"

// Forward declarations for CVI structures to avoid including cvi_audio.h in
// header if possible But cvi_audio.h is likely needed for compilation of the
// cpp file. We will include it in the cpp file.

namespace VoiceChat {

class VoiceChatClient {
 public:
  struct Config {
    std::string app_id;
    std::string access_key;
    std::string app_key = "PlgvMymc7f3tQnJ6";
    std::string resource_id = "volc.speech.dialog";
    std::string base_url =
        "wss://openspeech.bytedance.com/api/v3/realtime/dialogue";
    int sample_rate = 16000;
    std::string device_id = "cvi_device";
  };

  VoiceChatClient(const Config& config);
  ~VoiceChatClient();

  void start();
  void stop();
  void join();  // Wait for stop

  // Internal LWS callback, needs to be public for the C-style callback wrapper
  int onCallback(struct lws* wsi, enum lws_callback_reasons reason, void* user,
                 void* in, size_t len);

 private:
  void connect();
  void lwsServiceLoop();

  // Audio
  void audioCaptureLoop();
  void audioPlaybackLoop();

  // Protocol
  void sendStartConnection();
  void sendStartSession();
  void sendFinishConnection();
  void sendGreeting();
  void sendAudioData(const uint8_t* data, size_t len);
  std::vector<uint8_t> generateHeader(uint8_t msg_type, uint8_t msg_flags,
                                      uint8_t serial_method,
                                      uint8_t compression);
  void handleMessage(const std::vector<uint8_t>& data);

  void sendJson(const nlohmann::json& j, uint8_t msg_type, uint32_t msg_id);

  Config config_;
  std::atomic<bool> running_{false};
  std::string uuid_;
  std::string session_id_;

  // LWS
  struct lws_context* context_ = nullptr;
  struct lws* wsi_ = nullptr;
  std::thread service_thread_;
  std::mutex lws_mutex_;
  std::vector<uint8_t> rx_buffer_;  // For reassembling fragmented messages

  // Tx Queue
  std::deque<std::vector<uint8_t>> tx_queue_;
  size_t tx_offset_ = 0;
  std::mutex tx_mutex_;
  void sendPacket(const std::vector<uint8_t>& packet);

  // Audio
  std::thread capture_thread_;
  std::thread playback_thread_;

  std::queue<std::vector<uint8_t>> playback_queue_;
  std::mutex playback_mutex_;
  std::condition_variable playback_cv_;

  bool greeting_sent_ = false;
  bool session_start_sent_ = false;
  std::atomic<bool> reset_playback_buffer_{false};
  std::atomic<int> tts_payload_format_{-1};
  std::atomic<int64_t> last_play_ts_ms_{0};
  std::atomic<int64_t> last_tx_audio_ts_ms_{0};

  std::shared_ptr<AudioCapture> audio_capture_;
  std::shared_ptr<AudioPlayer> audio_player_;
};

}  // namespace VoiceChat
