#include "asr_client.hpp"
#include <libwebsockets.h>
#include <zlib.h>
#include <algorithm>  // + 添加算法头文件
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <json.hpp>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace Asr {
// + 改进UUID生成，使用更安全的方法
std::string AsrClient::gen_uuid() {
  static std::mt19937_64 rng(
      std::chrono::steady_clock::now().time_since_epoch().count());
  std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);

  // 生成128位随机数
  uint64_t part1 = dist(rng);
  uint64_t part2 = dist(rng);

  // 按照RFC 4122标准设置版本和变体
  part1 &= 0x00000000ffffffff;  // 清除版本字段
  part1 |= 0x4000000000000000;  // 设置为版本4 (0100)

  part2 &= 0x3fffffffffffffff;  // 清除变体字段
  part2 |= 0x8000000000000000;  // 设置变体为10 (RFC 4122)

  char buf[37];
  snprintf(buf, sizeof(buf), "%016llx-%04hx-%04hx-%04hx-%04hx%012llx",
           (unsigned long long)(part1 >> 32),
           (unsigned short)((part1 >> 16) & 0xffff),
           (unsigned short)(part1 & 0xffff),
           (unsigned short)((part2 >> 48) & 0xffff),
           (unsigned short)((part2 >> 32) & 0xffff),
           (unsigned long long)(part2 & 0x00000000ffffffff));
  return std::string(buf);
}

// + 实现URL解析函数
void AsrClient::parse_url(const std::string &url, std::string &host,
                          std::string &path) {
  auto pos = url.find("://");
  auto host_start = (pos == std::string::npos) ? 0 : pos + 3;
  auto path_start = url.find('/', host_start);
  host = url.substr(host_start, path_start - host_start);
  path = (path_start == std::string::npos) ? "/" : url.substr(path_start);
}

// Gzip压缩实现，添加压缩级别参数
std::string AsrClient::gzip_compress(const std::string &data, int level) {
  z_stream zs;
  memset(&zs, 0, sizeof(zs));

  // + 检查zlib初始化错误
  if (deflateInit2(&zs, level, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY) !=
      Z_OK) {
    _last_error = "zlib deflateInit2 failed";
    return "";
  }

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
      _last_error = "zlib deflate failed";
      return "";
    }
    out.append(buffer, sizeof(buffer) - zs.avail_out);
  } while (ret != Z_STREAM_END);

  // + 检查deflateEnd错误
  if (deflateEnd(&zs) != Z_OK) {
    _last_error = "zlib deflateEnd failed";
    return "";
  }
  return out;
}

// Gzip解压缩实现
std::string AsrClient::gzip_decompress(const std::string &data) {
  z_stream zs;
  memset(&zs, 0, sizeof(zs));

  // + 检查zlib初始化错误
  if (inflateInit2(&zs, 15 + 16) != Z_OK) {
    _last_error = "zlib inflateInit2 failed";
    return "";
  }

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
      _last_error = "zlib inflate failed";
      return "";
    }
    out.append(buffer, sizeof(buffer) - zs.avail_out);
  } while (ret != Z_STREAM_END);

  // + 检查inflateEnd错误
  if (inflateEnd(&zs) != Z_OK) {
    _last_error = "zlib inflateEnd failed";
    return "";
  }
  return out;
}

void AsrClient::construct_param() {
  _full_req_param.push_back(_protocol_version << 4 | _header_size >> 2);
  _full_req_param.push_back(MessageType::FULL_CLIENT_REQUEST << 4 |
                            MessageTypeFlag::NO_SEQUENCE_NUMBER);
  _full_req_param.push_back(_message_serial << 4 | _message_compress);
  _full_req_param.push_back(_reserved);

  json req_obj = {{"app",
                   {
                       {"appid", _appid},
                       {"cluster", _cluster},
                       {"token", _token},
                   }},
                  {"user", {{"uid", _uid}}},
                  {"request",
                   {{"reqid", _reqid},
                    {"nbest", _nbest},
                    {"workflow", _workflow},
                    {"show_language", _show_language},
                    {"show_utterances", _show_utterances},
                    {"result_type", _result_type},
                    {"sequence", _seq}}},
                  {"audio",
                   {{"format", _audio_format},
                    {"rate", _sample_rate},
                    {"language", _language},
                    {"bits", _bit_width},
                    {"channel", _channels},
                    {"codec", _codec}}}};

  std::string payload = req_obj.dump();
  payload = gzip_compress(payload);
  auto payload_len_big = htonl(static_cast<uint32_t>(payload.size()));
  _full_req_param.append((const char *)&payload_len_big, sizeof(uint32_t));
  _full_req_param.append(payload.data(), payload.size());
  std::cout << "reqid: " << _reqid << std::endl;
}

// libwebsockets回调
int AsrClient::lws_callback(struct lws *wsi, enum lws_callback_reasons reason,
                            void *user, void *in, size_t len) {
  AsrClient *client = reinterpret_cast<AsrClient *>(lws_wsi_user(wsi));
  switch (reason) {
    case LWS_CALLBACK_CLIENT_ESTABLISHED:
      client->on_open(wsi);
      break;
    case LWS_CALLBACK_CLIENT_RECEIVE:
      client->on_message(wsi, static_cast<const char *>(in), len);
      break;
    case LWS_CALLBACK_CLIENT_CONNECTION_ERROR:
      client->on_error(wsi, "Connection error");
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

struct lws_protocols AsrClient::_protocols[] = {
    {/* name */ "asr.v2",
     /* callback */ AsrClient::lws_callback,
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

// 构造函数
AsrClient::AsrClient()
    : _url("wss://openspeech.bytedance.com/api/v2/asr"),
      _audio_format("raw"),
      _channels(1),
      _sample_rate(16000),
      _bit_width(16),
      _connected(false),
      _finished(false),
      _thread_running(false),
      _callback(nullptr) {
  _reqid = gen_uuid();
}

AsrClient::~AsrClient() {
  close();
  stop();
  // + 先等待线程结束，再销毁context
  if (_io_thread.joinable()) {
    _io_thread.join();
  }

  if (_context) {
    lws_context_destroy(_context);
    _context = nullptr;
  }
}

void AsrClient::stop() {
  if (!_stopped) {
    _stopped = true;
    close();
  }
}

// join the IO thread
void AsrClient::join() {
  if (_io_thread.joinable()) {
    _io_thread.join();
  }
}

void AsrClient::set_appid(const std::string &appid) { _appid = appid; }
void AsrClient::set_token(const std::string &token) { _token = token; }
void AsrClient::set_cluster(const std::string &cluster) { _cluster = cluster; }
void AsrClient::set_audio_format(const std::string &fmt, int ch, int sr,
                                 int bw) {
  _audio_format = fmt;
  _channels = ch;
  _sample_rate = sr;
  _bit_width = bw;
}
void AsrClient::set_callback(AsrCallback *cb) { _callback = cb; }

// 同步连接
bool AsrClient::sync_connect(int timeout_sec) {
  try {
    construct_param();
  } catch (const std::exception &e) {
    _last_error = std::string("Construct param failed: ") + e.what();
    return false;
  }

  struct lws_context_creation_info info;
  memset(&info, 0, sizeof(info));
  info.port = CONTEXT_PORT_NO_LISTEN;
  info.protocols = _protocols;
  info.gid = -1;
  info.uid = -1;
  info.options = LWS_SERVER_OPTION_DO_SSL_GLOBAL_INIT;
  _context = lws_create_context(&info);
  if (!_context) {
    _last_error = "Failed to create libwebsockets context";
    return false;
  }

  std::string host, path;
  parse_url(_url, host, path);
  // std::string qs = "?appid=" + _appid + "&token=" + _token;
  std::string qs = "?appid=" + _appid;
  if (!_cluster.empty()) qs += "&cluster=" + _cluster;
  std::string ws_path = path + qs;

  struct lws_client_connect_info conn_info;
  memset(&conn_info, 0, sizeof(conn_info));
  conn_info.context = _context;
  conn_info.port = 443;
  conn_info.address = host.c_str();
  conn_info.path = ws_path.c_str();
  conn_info.host = host.c_str();
  conn_info.origin = "https://www.volcengine.com";
  conn_info.protocol = _protocols[0].name;
  // conn_info.ssl_connection = 1;
  conn_info.ssl_connection = LCCSCF_USE_SSL | LCCSCF_ALLOW_SELFSIGNED |
                             LCCSCF_SKIP_SERVER_CERT_HOSTNAME_CHECK;
  conn_info.userdata = this;
  // conn_info.headers = ("authorization: Bearer " + _token + "\r\n").c_str();

  _wsi = lws_client_connect_via_info(&conn_info);
  if (!_wsi) {
    _last_error = "Failed to connect";
    if (_context) {
      lws_context_destroy(_context);
      _context = nullptr;
    }
    std::cerr << "Failed to connect: " << _last_error << std::endl;
    return false;
  }

  lws_set_wsi_user(_wsi, this);
  _thread_running = true;

  // + 使用lambda捕获this指针的引用
  _io_thread = std::thread([this] {
    while (_thread_running) {
      // + 添加超时，允许线程响应退出请求
      if (lws_service(_context, 100) < 0) {
        break;
      }
    }
  });

  std::unique_lock<std::mutex> lk(_mtx);
  if (!_cv.wait_for(lk, std::chrono::seconds(timeout_sec),
                    [this] { return _finished.load(); })) {
    _last_error = "ASR connect timeout";
    close();
    return false;
  }

  return _connected;
}

int AsrClient::send_audio(const std::string &audio, bool is_last) {
  std::string payload;
  payload.push_back(_protocol_version << 4 | _header_size >> 2);
  if (!is_last) {
    payload.push_back(AUDIO_ONLY_CLIENT_REQUEST << 4 | NO_SEQUENCE_NUMBER);
  } else {
    payload.push_back(AUDIO_ONLY_CLIENT_REQUEST << 4 |
                      NEGATIVE_SEQUENCE_SERVER_ASSGIN);
  }
  payload.push_back(_message_serial << 4 | _message_compress);
  payload.push_back(_reserved);

  std::string gzip_data = gzip_compress(audio);
  // std::cout << "gzip_data size: " << gzip_data.size() << std::endl;
  auto chunk_len_big = htonl(static_cast<uint32_t>(gzip_data.size()));
  payload.append((const char *)&chunk_len_big, 4);
  payload.append(gzip_data.data(), gzip_data.size());
  int result = send_data(_wsi, payload, true);
  if (result < 0) {
    return -1;
  }
  return 0;
}

// 关闭连接
void AsrClient::close() {
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

// 发送数据辅助函数
int AsrClient::send_data(struct lws *wsi, const std::string &data,
                         bool binary) {
  if (!wsi) {
    _last_error = "Invalid websocket pointer";
    return -1;
  }

  size_t total = LWS_PRE + data.size();
  std::vector<unsigned char> buf(total);
  memcpy(buf.data() + LWS_PRE, data.data(), data.size());

  int result = lws_write(wsi, buf.data() + LWS_PRE, data.size(),
                         binary ? LWS_WRITE_BINARY : LWS_WRITE_TEXT);

  if (result < 0) {
    _last_error = "Failed to send data";
  }
  return result;
}

// 回调实现
void AsrClient::on_open(struct lws *wsi) {
  {
    std::lock_guard<std::mutex> lk(_mtx);
    _connected = true;
    _wsi = wsi;
    _finished = true;
  }

  send_data(wsi, _full_req_param, true);
  if (_callback) _callback->on_open(this);

  _cv.notify_one();
}

void AsrClient::on_message(struct lws *wsi, const char *msg, size_t len) {
  std::string payload_msg;
  int ret = parse_response(std::string(msg, len), payload_msg);
  if (ret == 0 && !payload_msg.empty()) {
    if (_callback) _callback->on_message(this, payload_msg);
  } else {
    // 打印原始包的十六进制，方便调试
    std::cout << "Received non-standard or error message, hex: ";
    for (size_t i = 0; i < len; ++i) {
      printf("%02X ", (unsigned char)msg[i]);
    }
    std::cout << std::endl;
  }
}

void AsrClient::on_error(struct lws *wsi, const char *err) {
  std::string error_msg = err ? err : "Unknown error";
  if (_callback) _callback->on_error(this, error_msg);

  {
    std::lock_guard<std::mutex> lk(_mtx);
    _finished = true;
    _connected = false;
  }
  _cv.notify_one();
}

void AsrClient::on_close(struct lws *wsi) {
  {
    std::lock_guard<std::mutex> lk(_mtx);
    _wsi = nullptr;
    _connected = false;
    _finished = true;
  }

  if (_callback) _callback->on_close(this);

  _cv.notify_one();
}
int AsrClient::parse_response(const std::string &msg,
                              std::string &payload_msg) {
  int header_len = (msg[0] & 0x0f) << 2;
  int message_type = (msg[1] & 0xf0) >> 4;
  int message_serial = (msg[2] & 0xf0) >> 4;
  int message_compress = msg[2] & 0x0f;
  uint32_t payload_offset = 0;
  uint32_t payload_len = 0;
  std::string payload;
  json payload_obj;

  // std::cout << "message_type: " << message_type
  //           << ", message_serial: " << message_serial
  //           << ", message_compress: " << message_compress
  //           << ", header_len: " << header_len << ", msg.size(): " <<
  //           msg.size()
  //           << std::endl;
  if (static_cast<MessageType>(message_type) ==
      MessageType::FULL_SERVER_RESPONSE) {
    payload_len = *(unsigned int *)(msg.data() + header_len);
    payload_len = (payload_len >> 24) | ((payload_len >> 8) & 0x00FF00) |
                  ((payload_len << 8) & 0xFF0000) | (payload_len << 24);
    payload_offset = header_len + 4;
  } else if (static_cast<MessageType>(message_type) ==
             MessageType::SERVER_ACK) {
    uint32_t seq = *(unsigned int *)(msg.data() + header_len);
    seq = (seq >> 24) | ((seq >> 8) & 0x00FF00) | ((seq << 8) & 0xFF0000) |
          (seq << 24);
    if (msg.size() > 8) {
      payload_len = *(unsigned int *)(msg.data() + header_len + 4);
      payload_len = (payload_len >> 24) | ((payload_len >> 8) & 0x00FF00) |
                    ((payload_len << 8) & 0xFF0000) | (payload_len << 24);
      payload_offset = header_len + 8;
    }
  } else if (static_cast<MessageType>(message_type) ==
             MessageType::ERROR_MESSAGE_FROM_SERVER) {
    uint32_t error_code = *(unsigned int *)(msg.data() + header_len);
    error_code = (error_code >> 24) | ((error_code >> 8) & 0x00FF00) |
                 ((error_code << 8) & 0xFF0000) | (error_code << 24);
    payload_len = *(unsigned int *)(msg.data() + header_len + 4);
    payload_len = (payload_len >> 24) | ((payload_len >> 8) & 0x00FF00) |
                  ((payload_len << 8) & 0xFF0000) | (payload_len << 24);
    payload_offset = header_len + 8;
    std::cout << "error_code: " << error_code << std::endl;
    if (payload_len > 0) {
      std::string payload_raw = msg.substr(payload_offset, payload_len);
      std::string payload = gzip_decompress(payload_raw);
      std::cout << "error payload (decompressed): " << payload << std::endl;
    }
  } else {
    std::cout << "unsupported message type: " << message_type << std::endl;
    return -1;
  }

  if (static_cast<MessageCompress>(message_compress) == MessageCompress::GZIP &&
      payload_len > 0) {
    payload = gzip_decompress(msg.substr(payload_offset, payload_len));
  }
  if (static_cast<MessageSerial>(message_serial) == MessageSerial::JSON &&
      !payload.empty()) {
    payload_obj = json::parse(payload);
  }
  payload_msg = payload;
  if (payload_obj.contains("code") && payload_obj["code"].is_number_integer() &&
      payload_obj["code"] != json(1000)) {
    return -1;
  }
  if (payload_obj.contains("sequence") &&
      payload_obj["sequence"].is_number_integer() &&
      payload_obj["sequence"] < json(0)) {
    _recv_last_msg = true;
    return 0;
  }
  return 0;
}
}  // namespace Asr