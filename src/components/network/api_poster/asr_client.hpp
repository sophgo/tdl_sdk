#ifndef ASR_CLIENT_HPP
#define ASR_CLIENT_HPP

#include <libwebsockets.h>
#include <zlib.h>
#include <atomic>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <json.hpp>
#include <mutex>
#include <string>
#include <system_error>  // + 添加系统错误支持
#include <thread>
#include <vector>

namespace Asr {
class AsrClient;
class AsrCallback {
 public:
  // This message handler will be invoked once for each websocket connection
  // open.
  virtual void on_open(AsrClient *asr_client) = 0;

  // This message handler will be invoked once for each incoming message. It
  // prints the message and then sends a copy of the message back to the server.
  virtual void on_message(AsrClient *asr_client, std::string msg) = 0;

  virtual void on_error(AsrClient *asr_client, std::string msg) = 0;

  virtual void on_close(AsrClient *asr_client) = 0;
};

class AsrClient {
 public:
  using json = nlohmann::json;
  enum AudioType : uint8_t { LOCAL = 1, URL = 2 };

  enum AuthType : uint8_t { TOKEN = 1, SIGNATURE = 2 };

  enum ConnState { CONNECTING = 0, OPEN = 1, CLOSING = 2, CLOSED = 3 };

  AsrClient();
  AsrClient(const AsrClient &) = delete;
  AsrClient &operator=(const AsrClient &) = delete;
  ~AsrClient();
  void stop();

  void join();

  void set_appid(const std::string &appid);

  void set_token(const std::string &token);

  void set_auth_type(AuthType auth_type);

  // set secret key when using signature auth
  void set_secret_key(const std::string &sk);

  void set_audio_format(const std::string &format, int channels,
                        int sample_rate, int bits);

  void set_cluster(const std::string &cluster);

  void set_callback(AsrCallback *asr_callback);

  int connect();                           // 异步连接
  bool sync_connect(int timeout_sec = 5);  // 同步连接
  int send_audio(const std::string &data, bool is_last);
  void close();  // 只设置线程退出标志和通知，不做 context 销毁

  // + 添加错误信息获取接口
  std::string get_last_error() const { return _last_error; }

  ConnState get_state() const;
  void set_reqeust_handle(long handle);
  long get_reqeust_handle() const;
  void set_connected();
  bool get_connect_status() const;

 private:
  static int lws_callback(struct lws *, enum lws_callback_reasons, void *,
                          void *, size_t);
  void on_open(struct lws *);
  void on_message(struct lws *, const char *, size_t);
  void on_error(struct lws *, const char * = nullptr);
  void on_close(struct lws *);
  void construct_param();
  std::string gzip_compress(
      const std::string &,
      int level = Z_DEFAULT_COMPRESSION);  // + 添加压缩级别参数
  std::string gzip_decompress(const std::string &);
  std::string gen_uuid();
  int send_data(struct lws *, const std::string &, bool binary);
  int parse_response(const std::string &msg, std::string &payload_msg);
  // + 解析URL的辅助函数声明
  void parse_url(const std::string &url, std::string &host, std::string &path);

 private:
  enum MessageType : uint8_t {
    FULL_CLIENT_REQUEST = 0b0001,
    AUDIO_ONLY_CLIENT_REQUEST = 0b0010,
    FULL_SERVER_RESPONSE = 0b1001,
    SERVER_ACK = 0b1011,
    ERROR_MESSAGE_FROM_SERVER = 0b1111
  };

  enum MessageTypeFlag : uint8_t {
    NO_SEQUENCE_NUMBER = 0b0000,
    POSITIVE_SEQUENCE_CLIENT_ASSGIN = 0b0001,
    NEGATIVE_SEQUENCE_SERVER_ASSGIN = 0b0010,
    NEGATIVE_SEQUENCE_CLIENT_ASSIGN = 0b0011
  };

  enum MessageSerial : uint8_t {
    NO_SERIAL = 0b0000,
    JSON = 0b0001,
    CUSTOM_SERIAL = 0b1111
  };

  enum MessageCompress : uint8_t {
    NO_COMPRESS = 0b0000,
    GZIP = 0b0001,
    CUSTOM_COMPRESS = 0b1111
  };
  std::string _url{"wss://openspeech.bytedance.com/api/v2/asr"};
  std::string _full_req_param;

  std::string _reqid;
  int32_t _seq{1};

  std::string _appid;
  std::string _token;
  std::string _sk;
  AuthType _auth_type{TOKEN};
  std::string _cluster{""};
  std::string _uid{"asr_cpp_demo"};
  std::string _workflow{"audio_in,resample,partition,vad,fe,decode"};
  int _nbest{1};
  bool _show_language{false};
  bool _show_utterances{false};
  std::string _result_type{"full"};
  std::string _language{"zh-CN"};

  AudioType _audio_type{LOCAL};
  std::string _audio_format{"wav"};
  int _sample_rate{16000};
  int _bit_width{16};
  int _channels{1};
  std::string _codec{"raw"};
  bool _recv_last_msg{false};

  uint8_t _protocol_version{0b0001};
  uint8_t _header_size{4};
  MessageType _message_type{MessageType::FULL_CLIENT_REQUEST};
  MessageTypeFlag _message_type_flag{MessageTypeFlag::NO_SEQUENCE_NUMBER};
  MessageSerial _message_serial{MessageSerial::JSON};
  MessageCompress _message_compress{MessageCompress::GZIP};
  uint8_t _reserved{0};

  std::mutex _mtx;
  std::condition_variable _cv;
  bool _connected_notify{false};
  bool _connected{false};
  std::atomic<bool> _stopped{false};
  std::atomic<bool> _finished{false};
  std::atomic<bool> _thread_running{false};
  bool _use_sync_connect{false};
  std::thread _io_thread;
  AsrCallback *_callback{nullptr};

  struct lws_context *_context{nullptr};
  struct lws *_wsi{nullptr};
  static struct lws_protocols _protocols[];
  bool _handshake_auth_added = false;

  // + 新增错误信息存储
  mutable std::string _last_error;
};
}  // namespace Asr

#endif  // ASR_CLIENT_HPP