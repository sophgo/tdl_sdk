#include "voice_chat_client.hpp"
#include <unistd.h>
#include <zlib.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include "cvi_audio.h"
#include "cvi_sys.h"

// Protocol Constants
#define PROTOCOL_VERSION 0b0001
#define DEFAULT_HEADER_SIZE 0b0001

#define CLIENT_FULL_REQUEST 0b0001
#define CLIENT_AUDIO_ONLY_REQUEST 0b0010
#define SERVER_FULL_RESPONSE 0b1001
#define SERVER_ACK 0b1011
#define SERVER_ERROR_RESPONSE 0b1111

#define MSG_WITH_EVENT 0b0100
#define NO_SEQUENCE 0b0000
#define NEG_SEQUENCE 0b0010

#define SERIALIZATION_JSON 0b0001
#define NO_SERIALIZATION 0b0000

#define COMPRESSION_NONE 0b0000
#define COMPRESSION_GZIP 0b0001

// Audio Constants
#define AUDIO_PERIOD_SIZE 640
#define AUDIO_INPUT_SAMPLE_RATE 16000
#define AUDIO_OUTPUT_SAMPLE_RATE 16000
#define AUDIO_FORMAT_SIZE 2  // 16bit = 2bytes
#define AUDIO_VAD_AVG_THRESHOLD 350
#define AUDIO_ECHO_MUTE_MS 300
#define AUDIO_KEEPALIVE_MS 200

namespace VoiceChat {

static int64_t now_ms() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(steady_clock::now().time_since_epoch())
      .count();
}

static std::vector<uint8_t> gzip_compress(const std::vector<uint8_t> &data) {
  if (data.empty()) return {};
  z_stream zs;
  memset(&zs, 0, sizeof(zs));
  if (deflateInit2(&zs, Z_DEFAULT_COMPRESSION, Z_DEFLATED, 15 | 16, 8,
                   Z_DEFAULT_STRATEGY) != Z_OK) {
    return {};
  }
  zs.next_in = (Bytef *)data.data();
  zs.avail_in = data.size();
  int ret;
  std::vector<uint8_t> out;
  unsigned char buffer[32768];
  do {
    zs.next_out = buffer;
    zs.avail_out = sizeof(buffer);
    ret = deflate(&zs, Z_FINISH);
    if (out.size() < zs.total_out) {
      out.insert(out.end(), buffer, buffer + (zs.total_out - out.size()));
    }
  } while (ret == Z_OK);
  deflateEnd(&zs);
  return out;
}

static std::vector<uint8_t> gzip_decompress(const std::vector<uint8_t> &data) {
  if (data.empty()) return {};
  z_stream zs;
  memset(&zs, 0, sizeof(zs));
  if (inflateInit2(&zs, 15 | 16) != Z_OK) return {};
  zs.next_in = (Bytef *)data.data();
  zs.avail_in = data.size();
  int ret;
  std::vector<uint8_t> out;
  unsigned char buffer[32768];
  do {
    zs.next_out = buffer;
    zs.avail_out = sizeof(buffer);
    ret = inflate(&zs, Z_NO_FLUSH);
    if (out.size() < zs.total_out) {
      out.insert(out.end(), buffer, buffer + (zs.total_out - out.size()));
    }
  } while (ret == Z_OK);
  inflateEnd(&zs);
  return out;
}

static bool looks_like_float32_pcm(const std::vector<uint8_t> &data) {
  if (data.size() < 256 || (data.size() % 4) != 0) return false;
  const float *f = reinterpret_cast<const float *>(data.data());
  size_t n = std::min<size_t>(data.size() / 4, 256);
  size_t ok = 0;
  size_t tiny = 0;
  size_t mid = 0;
  for (size_t i = 0; i < n; ++i) {
    float v = f[i];
    if (!std::isfinite(v)) continue;
    float av = std::fabs(v);
    if (av > 1.5f) continue;
    ok++;
    if (av < 1e-6f) tiny++;
    if (av >= 1e-4f) mid++;
  }
  if (ok * 100 < n * 98) return false;
  if (mid * 100 < n * 2) return false;
  if (tiny * 100 > n * 95) return false;
  return true;
}

static std::vector<uint8_t> float32_to_int16_pcm(
    const std::vector<uint8_t> &data) {
  if ((data.size() % 4) != 0) return {};
  size_t n = data.size() / 4;
  std::vector<uint8_t> out(n * 2);
  const float *in = reinterpret_cast<const float *>(data.data());
  int16_t *o = reinterpret_cast<int16_t *>(out.data());
  for (size_t i = 0; i < n; ++i) {
    float v = in[i];
    if (!std::isfinite(v)) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    if (v < -1.0f) v = -1.0f;
    o[i] = static_cast<int16_t>(v * 32767.0f);
  }
  return out;
}

static std::vector<uint8_t> resample_int16_to_int16_pcm(
    const std::vector<uint8_t> &data, int in_rate, int out_rate) {
  if ((data.size() % 2) != 0) return {};
  if (in_rate <= 0 || out_rate <= 0) return {};

  const int16_t *in = reinterpret_cast<const int16_t *>(data.data());
  size_t in_n = data.size() / 2;
  if (in_n < 2) return {};

  size_t out_n =
      static_cast<size_t>((static_cast<uint64_t>(in_n) * out_rate) / in_rate);
  if (out_n < 1) out_n = 1;
  std::vector<uint8_t> out(out_n * 2);
  int16_t *o = reinterpret_cast<int16_t *>(out.data());

  for (size_t j = 0; j < out_n; ++j) {
    double pos = (static_cast<double>(j) * in_rate) / out_rate;
    size_t i0 = static_cast<size_t>(pos);
    if (i0 >= in_n - 1) i0 = in_n - 2;
    double frac = pos - static_cast<double>(i0);

    double v0 = static_cast<double>(in[i0]);
    double v1 = static_cast<double>(in[i0 + 1]);
    double v = v0 * (1.0 - frac) + v1 * frac;

    if (v > 32767.0) v = 32767.0;
    if (v < -32768.0) v = -32768.0;
    o[j] = static_cast<int16_t>(v);
  }
  return out;
}

static std::vector<uint8_t> resample_float32_to_int16_pcm(
    const std::vector<uint8_t> &data, int in_rate, int out_rate) {
  if ((data.size() % 4) != 0) return {};
  if (in_rate <= 0 || out_rate <= 0) return {};

  const float *in = reinterpret_cast<const float *>(data.data());
  size_t in_n = data.size() / 4;
  if (in_n < 2) return {};

  size_t out_n =
      static_cast<size_t>((static_cast<uint64_t>(in_n) * out_rate) / in_rate);
  if (out_n < 1) out_n = 1;
  std::vector<uint8_t> out(out_n * 2);
  int16_t *o = reinterpret_cast<int16_t *>(out.data());

  for (size_t j = 0; j < out_n; ++j) {
    double pos = (static_cast<double>(j) * in_rate) / out_rate;
    size_t i0 = static_cast<size_t>(pos);
    if (i0 >= in_n - 1) i0 = in_n - 2;
    double frac = pos - static_cast<double>(i0);

    float v0 = in[i0];
    float v1 = in[i0 + 1];
    if (!std::isfinite(v0)) v0 = 0.0f;
    if (!std::isfinite(v1)) v1 = 0.0f;
    float v = static_cast<float>(v0 * (1.0 - frac) + v1 * frac);

    if (v > 1.0f) v = 1.0f;
    if (v < -1.0f) v = -1.0f;
    o[j] = static_cast<int16_t>(v * 32767.0f);
  }
  return out;
}

static int detect_tts_payload_format(const std::vector<uint8_t> &payload) {
  if ((payload.size() % 2) != 0) return 0;
  if ((payload.size() % 4) == 0 && looks_like_float32_pcm(payload)) return 1;
  return 0;
}

static void open_append(std::ofstream &ofs, const std::string &primary_path,
                        const std::string &fallback_path) {
  if (ofs.is_open()) return;
  ofs.open(primary_path, std::ios::binary | std::ios::app);
  if (ofs.is_open()) return;
  ofs.clear();
  ofs.open(fallback_path, std::ios::binary | std::ios::app);
}

static std::string generate_uuid() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<> dis(0, 15);
  static std::uniform_int_distribution<> dis2(8, 11);

  std::stringstream ss;
  int i;
  ss << std::hex;
  for (i = 0; i < 8; i++) {
    ss << dis(gen);
  }
  ss << "-";
  for (i = 0; i < 4; i++) {
    ss << dis(gen);
  }
  ss << "-4";
  for (i = 0; i < 3; i++) {
    ss << dis(gen);
  }
  ss << "-";
  ss << dis2(gen);
  for (i = 0; i < 3; i++) {
    ss << dis(gen);
  }
  ss << "-";
  for (i = 0; i < 12; i++) {
    ss << dis(gen);
  }
  return ss.str();
}

static int lws_callback_wrapper(struct lws *wsi,
                                enum lws_callback_reasons reason, void *user,
                                void *in, size_t len) {
  VoiceChatClient *client = (VoiceChatClient *)user;
  if (!client && reason == LWS_CALLBACK_CLIENT_ESTABLISHED) {
    // In some LWS versions user data might not be set yet?
    // But usually we set it in connect_info.userdata
  }
  if (client) {
    return client->onCallback(wsi, reason, user, in, len);
  }
  return 0;
}

static const struct lws_protocols protocols[] = {{
                                                     "voice-chat-protocol",
                                                     lws_callback_wrapper,
                                                     0,
                                                     4096,
                                                 },
                                                 {NULL, NULL, 0, 0}};

VoiceChatClient::VoiceChatClient(const Config &config) : config_(config) {
  uuid_ = generate_uuid();
  session_id_ = generate_uuid();
}

VoiceChatClient::~VoiceChatClient() { stop(); }

void VoiceChatClient::start() {
  if (running_) return;
  running_ = true;
  greeting_sent_ = false;
  session_start_sent_ = false;
  tts_payload_format_.store(-1, std::memory_order_relaxed);
  last_play_ts_ms_.store(0, std::memory_order_relaxed);
  last_tx_audio_ts_ms_.store(0, std::memory_order_relaxed);

  audio_capture_ = std::make_shared<AudioCapture>();
  AudioCapture::Config capture_config;
  capture_config.sample_rate = AUDIO_INPUT_SAMPLE_RATE;
  capture_config.period_size = AUDIO_PERIOD_SIZE;
  capture_config.volume = 15;
  if (audio_capture_->Init(capture_config) != 0) {
    std::cerr << "AudioCapture Init failed" << std::endl;
  }

  audio_player_ = std::make_shared<AudioPlayer>();
  AudioPlayer::Config player_config;
  player_config.sample_rate = AUDIO_OUTPUT_SAMPLE_RATE;
  player_config.period_size = AUDIO_PERIOD_SIZE;
  player_config.volume = 30;
  if (audio_player_->Init(player_config) != 0) {
    std::cerr << "AudioPlayer Init failed" << std::endl;
  }

  // Start threads
  service_thread_ = std::thread(&VoiceChatClient::lwsServiceLoop, this);
  capture_thread_ = std::thread(&VoiceChatClient::audioCaptureLoop, this);
  playback_thread_ = std::thread(&VoiceChatClient::audioPlaybackLoop, this);
}

void VoiceChatClient::stop() {
  running_ = false;

  if (context_) {
    lws_cancel_service(context_);
  }

  if (service_thread_.joinable()) service_thread_.join();
  if (capture_thread_.joinable()) capture_thread_.join();

  {
    std::lock_guard<std::mutex> lock(playback_mutex_);
    playback_cv_.notify_all();
  }
  if (playback_thread_.joinable()) playback_thread_.join();

  if (audio_capture_) audio_capture_->Deinit();
  if (audio_player_) audio_player_->Deinit();
}

void VoiceChatClient::join() {
  if (service_thread_.joinable()) service_thread_.join();
}

// Audio system initialization moved to AudioCapture/AudioPlayer classes

void VoiceChatClient::connect() {
  // Enable lws logging for debugging
  // LLL_HEADER (64) | LLL_PARSER (32) | LLL_ERR (1) | LLL_WARN (2) | LLL_NOTICE
  // (4)
  lws_set_log_level(
      LLL_ERR | LLL_WARN | LLL_NOTICE | LLL_USER | LLL_CLIENT | LLL_HEADER,
      NULL);

  struct lws_context_creation_info info;
  memset(&info, 0, sizeof info);
  info.port = CONTEXT_PORT_NO_LISTEN;
  info.protocols = protocols;
  info.gid = -1;
  info.uid = -1;
  info.options = LWS_SERVER_OPTION_DO_SSL_GLOBAL_INIT;

  context_ = lws_create_context(&info);
  if (!context_) {
    std::cerr << "lws init failed" << std::endl;
    return;
  }

  struct lws_client_connect_info ccinfo;
  memset(&ccinfo, 0, sizeof ccinfo);
  ccinfo.context = context_;

  // Parse URL (Simple parsing, assuming
  // wss://openspeech.bytedance.com/api/v3/realtime/dialogue)
  ccinfo.address = "openspeech.bytedance.com";
  ccinfo.port = 443;
  ccinfo.path = "/api/v3/realtime/dialogue";
  ccinfo.host = ccinfo.address;
  ccinfo.origin = NULL;  // Do not send Origin header unless necessary
  ccinfo.protocol = NULL;
  ccinfo.ssl_connection = LCCSCF_USE_SSL |
                          LCCSCF_SKIP_SERVER_CERT_HOSTNAME_CHECK |
                          LCCSCF_ALLOW_SELFSIGNED;
#if defined(LCCSCF_HTTP_V1_ONLY)
  ccinfo.ssl_connection |= LCCSCF_HTTP_V1_ONLY;
#endif
  ccinfo.userdata = this;

  wsi_ = lws_client_connect_via_info(&ccinfo);
}

void VoiceChatClient::lwsServiceLoop() {
  connect();
  while (running_) {
    lws_service(context_, 100);
  }
  if (context_) {
    lws_context_destroy(context_);
    context_ = nullptr;
  }
}

int VoiceChatClient::onCallback(struct lws *wsi,
                                enum lws_callback_reasons reason, void *user,
                                void *in, size_t len) {
  switch (reason) {
    case LWS_CALLBACK_CLIENT_APPEND_HANDSHAKE_HEADER: {
      unsigned char **p = (unsigned char **)in;
      unsigned char *end = (*p) + len;

      std::cout << "[VoiceChat] Appending headers..." << std::endl;

      auto add_header = [&](const std::string &name, const std::string &val) {
        std::cout << "  " << name << " " << val << std::endl;
        // Note: lws_add_http_header_by_name requires the colon in the name for
        // some versions/custom headers
        std::string name_with_colon = name + ":";
        if (lws_add_http_header_by_name(
                wsi, (const unsigned char *)name_with_colon.c_str(),
                (const unsigned char *)val.c_str(), val.length(), p, end)) {
          std::cerr << "[VoiceChat] Failed to add header: " << name
                    << std::endl;
        }
      };

      add_header("X-Api-Resource-Id", config_.resource_id);
      add_header("X-Api-App-Key", config_.app_key);
      add_header("X-Api-App-ID", config_.app_id);
      add_header("X-Api-Access-Key", config_.access_key);
      add_header("X-Api-Connect-Id", uuid_);
      break;
    }
    case LWS_CALLBACK_CLIENT_ESTABLISHED:
      std::cout << "[VoiceChat] Connected" << std::endl;
      sendStartConnection();
      lws_callback_on_writable(wsi);
      break;

    case LWS_CALLBACK_CLIENT_RECEIVE:
      // Handle incoming data
      // Append to buffer
      {
        const uint8_t *data = (const uint8_t *)in;
        rx_buffer_.insert(rx_buffer_.end(), data, data + len);

        if (lws_is_final_fragment(wsi)) {
          handleMessage(rx_buffer_);
          rx_buffer_.clear();
        }
      }
      break;

    case LWS_CALLBACK_CLIENT_CLOSED:
      std::cout << "[VoiceChat] Connection Closed" << std::endl;
      wsi_ = nullptr;
      running_ = false;
      break;

    case LWS_CALLBACK_EVENT_WAIT_CANCELLED:
      if (wsi_) {
        lws_callback_on_writable(wsi_);
      }
      break;

    case LWS_CALLBACK_CLIENT_WRITEABLE: {
      std::lock_guard<std::mutex> lock(tx_mutex_);
      if (!tx_queue_.empty()) {
        auto &packet = tx_queue_.front();
        unsigned char *buf = new unsigned char[LWS_PRE + packet.size()];
        memcpy(&buf[LWS_PRE], packet.data(), packet.size());

        int n = lws_write(wsi, &buf[LWS_PRE], packet.size(), LWS_WRITE_BINARY);
        delete[] buf;

        if (n < 0) {
          std::cerr << "[VoiceChat] Write failed, errno: " << errno << " ("
                    << strerror(errno) << ")" << std::endl;
          return -1;
        }

        // Assume full write for now.
        tx_queue_.pop_front();

        if (!tx_queue_.empty()) {
          lws_callback_on_writable(wsi);
        }
      }
      break;
    }

    case LWS_CALLBACK_CLIENT_CONNECTION_ERROR:
      std::cerr << "[VoiceChat] Connection Error: "
                << (in ? (const char *)in : "unknown reason") << std::endl;
      wsi_ = nullptr;
      running_ = false;
      break;

    default:
      break;
  }
  return 0;
}

std::vector<uint8_t> VoiceChatClient::generateHeader(uint8_t msg_type,
                                                     uint8_t msg_flags,
                                                     uint8_t serial_method,
                                                     uint8_t compression) {
  std::vector<uint8_t> header;
  header.reserve(4);
  header.push_back((PROTOCOL_VERSION << 4) | DEFAULT_HEADER_SIZE);
  header.push_back((msg_type << 4) | msg_flags);
  header.push_back((serial_method << 4) | compression);
  header.push_back(0x00);  // Reserved
  return header;
}

void VoiceChatClient::sendPacket(const std::vector<uint8_t> &packet) {
  {
    std::lock_guard<std::mutex> lock(tx_mutex_);
    tx_queue_.push_back(packet);
  }
  if (context_) {
    lws_cancel_service(context_);
  }
}

void VoiceChatClient::sendStartConnection() {
  std::vector<uint8_t> header =
      generateHeader(CLIENT_FULL_REQUEST, MSG_WITH_EVENT, SERIALIZATION_JSON,
                     COMPRESSION_GZIP);
  std::string json_str = "{}";
  std::vector<uint8_t> payload(json_str.begin(), json_str.end());
  std::vector<uint8_t> compressed = gzip_compress(payload);

  std::vector<uint8_t> packet = header;
  // Msg ID: 1
  packet.push_back(0);
  packet.push_back(0);
  packet.push_back(0);
  packet.push_back(1);
  // Payload Size
  uint32_t size = compressed.size();
  packet.push_back((size >> 24) & 0xFF);
  packet.push_back((size >> 16) & 0xFF);
  packet.push_back((size >> 8) & 0xFF);
  packet.push_back(size & 0xFF);
  packet.insert(packet.end(), compressed.begin(), compressed.end());

  sendPacket(packet);
}

void VoiceChatClient::sendJson(const nlohmann::json &j, uint8_t msg_type,
                               uint32_t msg_id) {
  std::vector<uint8_t> header = generateHeader(
      msg_type, MSG_WITH_EVENT, SERIALIZATION_JSON, COMPRESSION_GZIP);
  std::string json_str = j.dump();
  std::vector<uint8_t> payload(json_str.begin(), json_str.end());
  std::vector<uint8_t> compressed = gzip_compress(payload);

  if (compressed.empty()) {
    std::cerr << "[VoiceChat] Failed to compress JSON payload for msg "
              << msg_id << std::endl;
    return;
  }

  std::vector<uint8_t> packet = header;
  // Msg ID
  packet.push_back((msg_id >> 24) & 0xFF);
  packet.push_back((msg_id >> 16) & 0xFF);
  packet.push_back((msg_id >> 8) & 0xFF);
  packet.push_back(msg_id & 0xFF);

  bool has_session = false;
  // Session ID (Length + Data) - Python code adds this for StartSession (100)
  // and others
  if (msg_id >= 100) {
    has_session = true;
    uint32_t sess_len = session_id_.length();
    packet.push_back((sess_len >> 24) & 0xFF);
    packet.push_back((sess_len >> 16) & 0xFF);
    packet.push_back((sess_len >> 8) & 0xFF);
    packet.push_back(sess_len & 0xFF);
    packet.insert(packet.end(), session_id_.begin(), session_id_.end());
  }

  // Payload Size
  uint32_t size = compressed.size();
  packet.push_back((size >> 24) & 0xFF);
  packet.push_back((size >> 16) & 0xFF);
  packet.push_back((size >> 8) & 0xFF);
  packet.push_back(size & 0xFF);
  packet.insert(packet.end(), compressed.begin(), compressed.end());

  printf(
      "[VoiceChat] Sending JSON Packet. MsgID: %u, HasSession: %d, Payload "
      "size: "
      "%u, Total: %zu\n",
      msg_id, has_session, size, packet.size());

  sendPacket(packet);
}

void VoiceChatClient::sendAudioData(const uint8_t *data, size_t len) {
  // Python: message_type=CLIENT_AUDIO_ONLY_REQUEST,
  // serial_method=NO_SERIALIZATION, compression=GZIP And it wraps it in a
  // structure similar to JSON messages: MsgID(200) + SessionID + PayloadSize +
  // Payload Use CLIENT_AUDIO_ONLY_REQUEST (0x2) as per Python implementation
  std::vector<uint8_t> header =
      generateHeader(CLIENT_AUDIO_ONLY_REQUEST, MSG_WITH_EVENT,
                     NO_SERIALIZATION, COMPRESSION_GZIP);

  std::vector<uint8_t> payload(data, data + len);
  std::vector<uint8_t> compressed = gzip_compress(payload);

  if (compressed.empty()) {
    // Don't log for silence/empty to avoid spam, but if actual data failed,
    // that's bad.
    if (len > 0)
      std::cerr << "[VoiceChat] GZIP compression failed for audio!"
                << std::endl;
    return;
  }

  std::vector<uint8_t> packet = header;

  // Msg ID: 200 (Task Request / Audio)
  uint32_t msg_id = 200;
  packet.push_back((msg_id >> 24) & 0xFF);
  packet.push_back((msg_id >> 16) & 0xFF);
  packet.push_back((msg_id >> 8) & 0xFF);
  packet.push_back(msg_id & 0xFF);

  // Session ID
  uint32_t sess_len = session_id_.length();
  packet.push_back((sess_len >> 24) & 0xFF);
  packet.push_back((sess_len >> 16) & 0xFF);
  packet.push_back((sess_len >> 8) & 0xFF);
  packet.push_back(sess_len & 0xFF);
  packet.insert(packet.end(), session_id_.begin(), session_id_.end());

  // Payload Size
  uint32_t size = compressed.size();
  packet.push_back((size >> 24) & 0xFF);
  packet.push_back((size >> 16) & 0xFF);
  packet.push_back((size >> 8) & 0xFF);
  packet.push_back(size & 0xFF);
  packet.insert(packet.end(), compressed.begin(), compressed.end());

  // Log occasionally or for first packet
  static int audio_log_counter = 0;
  if (audio_log_counter++ % 100 == 0) {
    printf(
        "[VoiceChat] Sending Audio Packet (GZIP). MsgID: 200, Payload size: "
        "%u, "
        "Total: %zu\n",
        size, packet.size());
  }

  sendPacket(packet);
  last_tx_audio_ts_ms_.store(now_ms(), std::memory_order_relaxed);
}

void VoiceChatClient::sendStartSession() {
  session_start_sent_ = true;
  nlohmann::json j;
  j["reqid"] = generate_uuid();
  j["asr"]["extra"]["end_smooth_window_ms"] = 500;
  j["tts"]["speaker"] = "zh_female_xiaohe_jupiter_bigtts";
  j["tts"]["audio_config"]["channel"] = 1;
  j["tts"]["audio_config"]["format"] = "pcm_s16le";
  j["tts"]["audio_config"]["sample_rate"] = 24000;
  j["dialog"]["bot_name"] = "豆包";
  j["dialog"]["system_role"] = "你使用活泼灵动的女声，性格开朗，热爱生活。";
  j["dialog"]["speaking_style"] = "你的说话风格简洁明了，语速适中，语调自然。";
  j["dialog"]["extra"]["recv_timeout"] = 10;
  j["dialog"]["extra"]["input_mod"] = "audio";

  sendJson(j, CLIENT_FULL_REQUEST, 100);
}

void VoiceChatClient::sendFinishConnection() {
  nlohmann::json j;
  j["reqid"] = generate_uuid();
  sendJson(j, CLIENT_FULL_REQUEST, 2);
}

void VoiceChatClient::sendGreeting() {
  // Sending "Hello" to trigger the bot
  nlohmann::json j;
  j["reqid"] = generate_uuid();
  j["content"] = "你好，我是豆包，有什么可以帮助你的？";
  sendJson(j, CLIENT_FULL_REQUEST, 300);
}

void VoiceChatClient::handleMessage(const std::vector<uint8_t> &data) {
  if (data.size() < 4) return;

  // Parse Header
  uint8_t version = (data[0] >> 4);
  uint8_t header_size = (data[0] & 0x0F);
  uint8_t msg_type = (data[1] >> 4);
  uint8_t msg_flags = (data[1] & 0x0F);
  uint8_t serial_method = (data[2] >> 4);
  uint8_t compression = (data[2] & 0x0F);

  size_t curr = header_size * 4;
  if (data.size() < curr) return;

  uint32_t event = 0;
  if (msg_flags & NEG_SEQUENCE) curr += 4;
  if (msg_flags & MSG_WITH_EVENT) {
    if (curr + 4 > data.size()) return;
    event = (data[curr] << 24) | (data[curr + 1] << 16) |
            (data[curr + 2] << 8) | data[curr + 3];
    curr += 4;
  }

  if (msg_type == SERVER_ACK || msg_type == SERVER_FULL_RESPONSE) {
    if (curr + 4 > data.size()) return;
    // Session ID
    int32_t sess_len = (data[curr] << 24) | (data[curr + 1] << 16) |
                       (data[curr + 2] << 8) | data[curr + 3];
    curr += 4;
    curr += sess_len;

    if (curr + 4 > data.size()) return;
    // Payload Size
    int32_t p_size = (data[curr] << 24) | (data[curr + 1] << 16) |
                     (data[curr + 2] << 8) | data[curr + 3];
    curr += 4;

    if (curr + p_size > data.size()) return;
    std::vector<uint8_t> payload(data.begin() + curr,
                                 data.begin() + curr + p_size);

    if (compression == COMPRESSION_GZIP) {
      payload = gzip_decompress(payload);
      if (payload.empty()) {
        std::cerr << "[VoiceChat] Gzip decompress failed for event " << event
                  << std::endl;
        return;
      }
    }

    if ((msg_type == SERVER_ACK && event == 1) || event == 50) {
      // StartConnection Ack or Event 50 (Connection Confirmed)
      if (!session_start_sent_) {
        if (event == 50)
          std::cout << "[VoiceChat] Connection Confirmed (Event 50)"
                    << std::endl;
        sendStartSession();
      }
      if (event == 1) return;
    }

    if ((msg_type == SERVER_ACK && event == 100) || event == 150) {
      // StartSession Ack (100) or Dialog Started (150)
      std::cout << "[VoiceChat] Session Started (Event " << event << ")"
                << std::endl;
      if (!greeting_sent_) {
        std::cout
            << "[VoiceChat] Enabling audio transmission and sending greeting..."
            << std::endl;
        greeting_sent_ = true;
        sendGreeting();
        last_tx_audio_ts_ms_.store(0, std::memory_order_relaxed);
      }
      // Continue processing for event 150 as it might contain payload
      if (event == 100) return;
    }

    if (serial_method == SERIALIZATION_JSON) {
      try {
        std::string s(payload.begin(), payload.end());
        auto j = nlohmann::json::parse(s);
        std::cout << "[VoiceChat] Received Event: " << event << std::endl;

        if (event == 350) {
          reset_playback_buffer_.store(true, std::memory_order_relaxed);
          std::lock_guard<std::mutex> lock(playback_mutex_);
          while (!playback_queue_.empty()) playback_queue_.pop();
        }

        if (event == 152 || event == 153) {
          std::cout << "[VoiceChat] Session Finished" << std::endl;
          running_ = false;
        } else if (event == 450) {
          reset_playback_buffer_.store(true, std::memory_order_relaxed);
          std::lock_guard<std::mutex> lock(playback_mutex_);
          while (!playback_queue_.empty()) playback_queue_.pop();
        }
      } catch (...) {
        std::cerr << "[VoiceChat] Failed to parse JSON payload for event "
                  << event << std::endl;
      }
    } else {
      if (payload.empty()) return;

      int fmt = tts_payload_format_.load(std::memory_order_relaxed);
      if (fmt < 0) {
        fmt = detect_tts_payload_format(payload);
        tts_payload_format_.store(fmt, std::memory_order_relaxed);
        std::cout << "[VoiceChat] TTS payload format locked: "
                  << (fmt == 1 ? "pcm(float32)" : "pcm_s16le(int16)")
                  << std::endl;
      }
      std::vector<uint8_t> pcm16;
      if (fmt == 1) {
        pcm16 = resample_float32_to_int16_pcm(payload, 24000,
                                              AUDIO_OUTPUT_SAMPLE_RATE);
      } else {
        pcm16 = resample_int16_to_int16_pcm(payload, 24000,
                                            AUDIO_OUTPUT_SAMPLE_RATE);
      }
      if (pcm16.empty()) return;

      std::lock_guard<std::mutex> lock(playback_mutex_);
      playback_queue_.push(pcm16);
      playback_cv_.notify_one();
    }
  } else if (msg_type == SERVER_ERROR_RESPONSE) {
    std::cerr << "[VoiceChat] Server Error Response" << std::endl;

    // Parse Error Response (Code + Payload Size + Payload)
    // Note: Error response does NOT have Session ID field structure like
    // ACK/FULL_RESPONSE

    if (curr + 8 > data.size()) return;

    uint32_t code = (data[curr] << 24) | (data[curr + 1] << 16) |
                    (data[curr + 2] << 8) | data[curr + 3];
    curr += 4;

    uint32_t p_size = (data[curr] << 24) | (data[curr + 1] << 16) |
                      (data[curr + 2] << 8) | data[curr + 3];
    curr += 4;

    if (curr + p_size > data.size()) return;
    std::vector<uint8_t> payload(data.begin() + curr,
                                 data.begin() + curr + p_size);

    if (compression == COMPRESSION_GZIP) {
      payload = gzip_decompress(payload);
    }

    std::string s(payload.begin(), payload.end());
    std::cerr << "[VoiceChat] Error Code: " << code << " Detail: " << s
              << std::endl;

    running_ = false;
  }
}

void VoiceChatClient::audioCaptureLoop() {
  int frame_size_bytes = AUDIO_PERIOD_SIZE * AUDIO_FORMAT_SIZE;
  std::vector<uint8_t> buffer(frame_size_bytes);
  std::vector<uint8_t> silence(frame_size_bytes, 0);

  while (running_) {
    // Read Frame
    if (audio_capture_ && audio_capture_->GetFrame(buffer) == 0) {
      // Send to Server
      if (greeting_sent_) {  // Only send after session start
        if (!buffer.empty()) {
          const int64_t last_play =
              last_play_ts_ms_.load(std::memory_order_relaxed);
          const int64_t now = now_ms();
          const int64_t last_tx =
              last_tx_audio_ts_ms_.load(std::memory_order_relaxed);
          const bool need_keepalive =
              (last_tx == 0) || ((now - last_tx) >= AUDIO_KEEPALIVE_MS);
          const bool echo_muting =
              (last_play > 0 && (now - last_play) < AUDIO_ECHO_MUTE_MS);

          int64_t sum = 0;
          int16_t *pcm = reinterpret_cast<int16_t *>(buffer.data());
          const size_t n = buffer.size() / 2;
          for (size_t i = 0; i < n; ++i) sum += std::abs(pcm[i]);
          const int avg = static_cast<int>(sum / static_cast<int64_t>(n));

          static int frame_count = 0;
          if (frame_count++ % 100 == 0) {
            printf("[VoiceChat] Audio Level (Frame %d): %d\n", frame_count,
                   avg);
          }

          if (echo_muting) {
            if (need_keepalive) sendAudioData(silence.data(), silence.size());
          } else if (avg >= AUDIO_VAD_AVG_THRESHOLD) {
            sendAudioData(buffer.data(), buffer.size());
          } else if (need_keepalive) {
            sendAudioData(silence.data(), silence.size());
          }
        }
      }
      // Do not send audio before session starts!
    } else {
      usleep(1000);
    }
  }
}

void VoiceChatClient::audioPlaybackLoop() {
  std::vector<uint8_t> audio_buffer;
  size_t frame_bytes = AUDIO_PERIOD_SIZE * AUDIO_FORMAT_SIZE;
  const size_t max_buffer_bytes = frame_bytes * 200;
  std::vector<uint8_t> frame;
  frame.resize(frame_bytes);

  while (running_) {
    std::vector<uint8_t> chunk;
    {
      std::unique_lock<std::mutex> lock(playback_mutex_);
      playback_cv_.wait(
          lock, [this] { return !playback_queue_.empty() || !running_; });

      if (!running_) break;

      chunk = playback_queue_.front();
      playback_queue_.pop();
    }

    if (chunk.empty()) continue;

    if (reset_playback_buffer_.exchange(false, std::memory_order_relaxed)) {
      audio_buffer.clear();
    }
    audio_buffer.insert(audio_buffer.end(), chunk.begin(), chunk.end());
    if (audio_buffer.size() > max_buffer_bytes) {
      size_t drop = audio_buffer.size() - max_buffer_bytes;
      drop = (drop / frame_bytes) * frame_bytes;
      if (drop > 0 && drop < audio_buffer.size()) {
        audio_buffer.erase(audio_buffer.begin(), audio_buffer.begin() + drop);
      }
    }

    while (audio_buffer.size() >= frame_bytes) {
      memcpy(frame.data(), audio_buffer.data(), frame_bytes);

      int ret = -1;
      if (audio_player_) {
        ret = audio_player_->SendFrame(frame.data(), frame_bytes, 2000);
      }

      if (ret == 0) {
        last_play_ts_ms_.store(now_ms(), std::memory_order_relaxed);
      } else {
        // Failed to send, drop frame
      }

      audio_buffer.erase(audio_buffer.begin(),
                         audio_buffer.begin() + frame_bytes);
    }
  }
}

}  // namespace VoiceChat
