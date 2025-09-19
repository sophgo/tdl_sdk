#include "tts_client.hpp"

#include <libwebsockets.h>
#include <zlib.h>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <json.hpp>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>

namespace Tts {

// Gzip压缩实现（替代Boost）
std::string TtsClient::gzip_compress(const std::string &data) {
  z_stream zs;
  memset(&zs, 0, sizeof(zs));
  // 使用gzip格式 (windowBits=15+16)
  if (deflateInit2(&zs, Z_DEFAULT_COMPRESSION, Z_DEFLATED, 15 + 16, 8,
                   Z_DEFAULT_STRATEGY) != Z_OK)
    return "";
  zs.next_in = (Bytef *)data.data();
  zs.avail_in = data.size();
  std::string out;
  char buffer[16384];
  int ret;
  do {
    zs.next_out = (Bytef *)buffer;
    zs.avail_out = sizeof(buffer);
    ret = deflate(&zs, zs.avail_in ? Z_NO_FLUSH : Z_FINISH);
    if (ret == Z_STREAM_ERROR) {
      deflateEnd(&zs);
      return "";
    }
    out.append(buffer, sizeof(buffer) - zs.avail_out);
  } while (ret != Z_STREAM_END);
  deflateEnd(&zs);
  return out;
}

// Gzip解压缩实现（替代Boost）
std::string TtsClient::gzip_decompress(const std::string &data) {
  z_stream zs;
  memset(&zs, 0, sizeof(zs));
  // 使用gzip格式 (windowBits=15+16)
  if (inflateInit2(&zs, 15 + 16) != Z_OK) return "";
  zs.next_in = (Bytef *)data.data();
  zs.avail_in = data.size();
  std::string out;
  char buffer[16384];
  int ret;
  do {
    zs.next_out = (Bytef *)buffer;
    zs.avail_out = sizeof(buffer);
    ret = inflate(&zs, Z_NO_FLUSH);
    if (ret == Z_STREAM_ERROR || ret == Z_DATA_ERROR || ret == Z_MEM_ERROR) {
      inflateEnd(&zs);
      return "";
    }
    out.append(buffer, sizeof(buffer) - zs.avail_out);
  } while (ret != Z_STREAM_END);
  inflateEnd(&zs);
  return out;
}

// 生成UUID（替代Boost）
std::string TtsClient::gen_uuid() {
  static std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<uint32_t> dist(0, 0xffffffff);
  uint32_t data[4];
  for (int i = 0; i < 4; ++i) data[i] = dist(rng);
  char buf[37];
  snprintf(buf, sizeof(buf), "%08x-%04x-%04x-%04x-%04x%08x", data[0],
           (data[1] >> 16) & 0xffff,
           (data[1] & 0x0fff) | 0x4000,          // version 4
           ((data[2] >> 16) & 0x0fff) | 0x8000,  // variant
           data[2] & 0xffff, data[3]);
  return std::string(buf);
}

// 工具函数：解析url为host和path
static void parse_url(const std::string &url, std::string &host,
                      std::string &path) {
  auto pos = url.find("://");
  auto host_start = (pos == std::string::npos) ? 0 : pos + 3;
  auto path_start = url.find('/', host_start);
  host = url.substr(host_start, path_start - host_start);
  path = (path_start == std::string::npos) ? "/" : url.substr(path_start);
}

// libwebsockets回调函数
int TtsClient::lws_callback(struct lws *wsi, enum lws_callback_reasons reason,
                            void *user, void *in, size_t len) {
  TtsClient *client = reinterpret_cast<TtsClient *>(lws_wsi_user(wsi));
  switch (reason) {
    case LWS_CALLBACK_CLIENT_ESTABLISHED:
      client->on_open(wsi);
      break;
    case LWS_CALLBACK_CLIENT_RECEIVE:
      client->on_message(wsi, static_cast<const char *>(in), len);
      break;
    case LWS_CALLBACK_CLIENT_CONNECTION_ERROR:
      client->on_fail(wsi, "Connection error");
      break;
    case LWS_CALLBACK_CLIENT_CLOSED:
      client->on_close(wsi);
      break;
    case LWS_CALLBACK_CLIENT_APPEND_HANDSHAKE_HEADER: {
      if (!client->_handshake_auth_added) {
        unsigned char **p = (unsigned char **)in;
        unsigned char *end = *p + len;
        // 仅注入 Bearer Token
        std::string auth = "Bearer; " + client->_token;
        int result = lws_add_http_header_by_name(
            wsi, (const unsigned char *)"Authorization:",
            (const unsigned char *)auth.c_str(), auth.size(), p, end);
        if (result < 0) {
          lwsl_err("Failed to add Authorization header\n");
        }
        client->_handshake_auth_added = true;
      }
      break;
    }
    default:
      break;
  }
  return 0;
}

// 构造函数
TtsClient::TtsClient()
    : _url("wss://openspeech.bytedance.com/api/v1/tts/ws_binary"),
      _header{0x11, 0x10, 0x11, 0x00},
      _connected(false),
      _finished(false),
      _thread_running(false),
      _stopped(false) {
  // 初始化协议
}

struct lws_protocols TtsClient::_protocols[] = {
    {/* name */ "tts.v1",
     /* callback */ TtsClient::lws_callback,
     /* per_session_size */ 0,
     /* rx_buffer_size */ 16384,
     /* id */ 0,
     /* user */ nullptr,
     /* tx_packet_size */ 0},
    // —— 终结哨兵，一定要全部字段都给出，否则编译不通过 ——
    {/* name */ NULL,
     /* callback */ NULL,
     /* per_session_size */ 0,
     /* rx_buffer_size */ 0,
     /* id */ 0,
     /* user */ NULL,
     /* tx_packet_size */ 0}};

// 析构函数
TtsClient::~TtsClient() {
  close();
  stop();
  join();
  _thread_running = false;
  if (_io_thread.joinable()) {
    _io_thread.join();
  }
  if (_context) {
    lws_context_destroy(_context);
  }
}

void TtsClient::stop() {
  if (!_stopped) {
    _stopped = true;
    close();
  }
}

// join the IO thread
void TtsClient::join() {
  if (_io_thread.joinable()) {
    _io_thread.join();
  }
}
// setters
void TtsClient::set_appid(const std::string &appid) { _appid = appid; }
void TtsClient::set_token(const std::string &token) { _token = token; }
void TtsClient::set_cluster(const std::string &cluster) { _cluster = cluster; }
void TtsClient::set_voice_type(const std::string &vt) { _voice_type = vt; }
void TtsClient::set_encoding(const std::string &encoding) {
  _encoding = encoding;
}

// 同步请求
bool TtsClient::sync_request(const std::string &text,
                             const std::string &operation,
                             const std::string &output_path, int timeout_sec) {
  _output_path = output_path;
  construct_request(text, operation);
  struct lws_context_creation_info info;
  memset(&info, 0, sizeof(info));
  info.port = CONTEXT_PORT_NO_LISTEN;
  info.protocols = _protocols;
  info.gid = -1;
  info.uid = -1;
  info.options = LWS_SERVER_OPTION_DISABLE_OS_CA_CERTS |
                 LWS_SERVER_OPTION_DO_SSL_GLOBAL_INIT;
  _context = lws_create_context(&info);
  if (!_context) {
    std::cerr << "Failed to create libwebsockets context\n";
    return false;
  }
  std::string host, path;
  parse_url(_url, host, path);
  struct lws_client_connect_info conn_info;
  memset(&conn_info, 0, sizeof(conn_info));
  conn_info.context = _context;
  conn_info.port = 443;
  conn_info.address = host.c_str();
  conn_info.path = path.c_str();
  conn_info.host = host.c_str();
  conn_info.origin = "https://www.volcengine.com";
  conn_info.protocol = _protocols[0].name;
  // conn_info.ssl_connection = 1;
  conn_info.ssl_connection = LCCSCF_USE_SSL | LCCSCF_ALLOW_SELFSIGNED |
                             LCCSCF_SKIP_SERVER_CERT_HOSTNAME_CHECK;
  conn_info.userdata = this;
  _wsi = lws_client_connect_via_info(&conn_info);
  if (!_wsi) {
    std::cerr << "Failed to connect\n";
    lws_context_destroy(_context);
    _context = nullptr;
    return false;
  }
  lws_set_wsi_user(_wsi, this);
  _thread_running = true;
  _io_thread = std::thread([this] {
    while (_thread_running) {
      lws_service(_context, 100);
    }
  });
  std::unique_lock<std::mutex> lk(_mtx);
  if (!_cv.wait_for(lk, std::chrono::seconds(timeout_sec),
                    [this] { return _finished.load(); })) {
    std::cerr << "TTS request timeout\n";
    close();
    return false;
  }
  return true;
}

// 关闭连接
void TtsClient::close() {
  //   std::lock_guard<std::mutex> lk(_mtx);
  //   if (_wsi) {
  //     lws_set_timeout(_wsi, PENDING_TIMEOUT_CLOSE_SEND, LWS_TO_KILL_ASYNC);
  //     _wsi = nullptr;
  //   }
  //   _finished = true;
  // }
  std::lock_guard<std::mutex> lk(_mtx);
  if (!_thread_running) {
    return;
  }

  _thread_running = false;
  if (_wsi) {
    lws_set_timeout(_wsi, PENDING_TIMEOUT_CLOSE_SEND, LWS_TO_KILL_ASYNC);
    _wsi = nullptr;
  }
  _finished = true;
  _cv.notify_one();
}

// 连接打开回调
void TtsClient::on_open(struct lws *wsi) {
  _connected = true;
  _wsi = wsi;
  send_data(wsi, _request_frame, true);
}

// 发送数据辅助函数
int TtsClient::send_data(struct lws *wsi, const std::string &data,
                         bool binary) {
  // lws_write要求buf前面有LWS_PRE字节的预留空间
  size_t total = LWS_PRE + data.size();
  std::vector<unsigned char> buf(total);
  memcpy(buf.data() + LWS_PRE, data.data(), data.size());
  return lws_write(wsi, buf.data() + LWS_PRE, data.size(),
                   binary ? LWS_WRITE_BINARY : LWS_WRITE_TEXT);
}

// 消息接收回调
void TtsClient::on_message(struct lws *wsi, const char *msg, size_t len) {
  // WebSocket fragment 支持
  bool is_first = lws_is_first_fragment(wsi);
  bool is_final = lws_is_final_fragment(wsi);
  if (is_first) {
    _recv_buffer.clear();
  }
  // append 本次收到的数据
  _recv_buffer.insert(_recv_buffer.end(), (const uint8_t *)msg,
                      (const uint8_t *)msg + len);

  // 只有在收到最后一个 fragment 时，才算是一个完整协议包
  if (!is_final) {
    return;
  }
  const uint8_t *data = _recv_buffer.data();
  size_t total_len = _recv_buffer.size();

  uint8_t hdr0 = data[0];
  uint8_t hdr_sz = (hdr0 & 0x0f) * 4;
  uint8_t msg_type = (data[1] & 0xf0) >> 4;
  uint8_t flags = data[1] & 0x0f;
  uint8_t comp = data[2] & 0x0f;

  const uint8_t *body = data + hdr_sz;
  size_t body_sz = total_len - hdr_sz;

  // Audio chunk
  if (msg_type == 0xB) {
    if (flags == 0) {
      // ACK
      return;
    }
    int32_t seq = ntohl(*reinterpret_cast<const uint32_t *>(body));
    uint32_t chunk_sz = ntohl(*reinterpret_cast<const uint32_t *>(body + 4));
    const uint8_t *audio = body + 8;

    // 打开文件并写入
    if (!_ofs.is_open()) {
      _ofs.open(_output_path, std::ios::binary);
      if (!_ofs) {
        std::cerr << "[ERROR] Failed to open " << _output_path << "\n";
        close();
        return;
      }
    }
    // 这里一次性写完整个 chunk
    _ofs.write(reinterpret_cast<const char *>(audio), chunk_sz);
    std::cout << "[TTS] seq=" << seq << ", chunk_sz=" << chunk_sz << std::endl;

    // 最后一个音频块
    if (seq < 0) {
      _ofs.close();
      close();
      {
        std::lock_guard<std::mutex> lk(_mtx);
        _finished = true;
      }
      _cv.notify_one();
    }
    return;
  }
  // 错误消息 (0xF)
  else if (msg_type == 0xF) {
    std::string err;
    if (body_sz > 8) {
      err = std::string(reinterpret_cast<const char *>(body + 8), body_sz - 8);
      if (comp == 1) err = gzip_decompress(err);
    }
    std::cerr << "TTS server error: " << err << "\n";
    close();
    {
      std::lock_guard<std::mutex> lk(_mtx);
      _finished = true;
    }
    _cv.notify_one();
    return;
  } else {
    std::cout << "Unhandled msg_type=0x" << std::hex << int(msg_type)
              << std::dec << std::endl;
  }
}

// 连接失败回调
void TtsClient::on_fail(struct lws *wsi, const char *error) {
  std::cerr << "TTS connection failed: " << (error ? error : "Unknown error")
            << "\n";
  close();
  {
    std::lock_guard<std::mutex> lk(_mtx);
    _finished = true;
  }
  _cv.notify_one();
}

// 连接关闭回调
void TtsClient::on_close(struct lws *wsi) {
  _wsi = nullptr;
  _connected = false;
}

// 构建请求帧
void TtsClient::construct_request(const std::string &text,
                                  const std::string &operation) {
  nlohmann::json j;
  j["app"] = {{"appid", _appid}, {"token", _token}, {"cluster", _cluster}};
  j["user"] = {{"uid", "tts_cpp_demo"}};
  j["audio"] = {{"voice_type", _voice_type}, {"rate", 16000},
                {"encoding", _encoding},     {"speed_ratio", 1.0},
                {"volume_ratio", 1.0},       {"pitch_ratio", 1.0},
                {"language", "cn"}};
  j["request"] = {{"reqid", gen_uuid()},
                  {"text", text},
                  {"text_type", "plain"},
                  {"operation", operation}};
  std::string raw = j.dump();
  std::string gz = gzip_compress(raw);
  uint32_t len_be = htonl(static_cast<uint32_t>(gz.size()));
  _request_frame.clear();
  _request_frame.append(reinterpret_cast<const char *>(_header.data()),
                        _header.size());
  _request_frame.append(reinterpret_cast<const char *>(&len_be),
                        sizeof(len_be));
  _request_frame.append(gz);
}

}  // namespace Tts