#ifndef TTS_CLIENT_TTS_CLIENT_HPP
#define TTS_CLIENT_TTS_CLIENT_HPP

#include <libwebsockets.h>
#include <atomic>
#include <condition_variable>
#include <fstream>
#include <json.hpp>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace Tts {
class TtsClient;
class TtsClient {
 public:
  using json = nlohmann::json;

  TtsClient();
  ~TtsClient();

  // 设置认证信息
  void set_appid(const std::string &appid);
  void set_token(const std::string &token);
  void set_cluster(const std::string &cluster);
  void set_voice_type(const std::string &voice_type);
  void set_encoding(const std::string &encoding);

  /**
   * 同步发送TTS请求并接收音频数据
   * @param text 要转换为语音的文本
   * @param operation 操作类型("submit"或"query")
   * @param output_path 音频输出文件路径
   * @param timeout_sec 超时时间(秒)
   * @return 操作是否成功
   */
  bool sync_request(const std::string &text, const std::string &operation,
                    const std::string &output_path, int timeout_sec = 5);

  // 关闭连接
  void close();
  void stop();
  void join();

 private:
  // libwebsockets回调函数
  static int lws_callback(struct lws *wsi, enum lws_callback_reasons reason,
                          void *user, void *in, size_t len);
  void on_open(struct lws *wsi);
  void on_message(struct lws *wsi, const char *msg, size_t len);
  void on_fail(struct lws *wsi, const char *error);
  void on_close(struct lws *wsi);

  // 构建请求帧
  void construct_request(const std::string &text, const std::string &operation);

  // Gzip压缩/解压缩
  std::string gzip_compress(const std::string &data);
  std::string gzip_decompress(const std::string &data);

  // 生成UUID
  std::string gen_uuid();

  int send_data(struct lws *wsi, const std::string &data, bool binary);

 private:
  struct lws_context *_context{nullptr};     // libwebsockets上下文
  struct lws *_wsi{nullptr};                 // WebSocket连接
  static struct lws_protocols _protocols[];  // 协议定义

  std::string _url{"wss://openspeech.bytedance.com/api/v1/tts/ws_binary"};
  std::string _appid, _token, _cluster, _voice_type, _encoding;
  std::vector<uint8_t> _header{0x11, 0x10, 0x11, 0x00};  // 4字节固定头部
  std::string _request_frame;  // 完整请求帧(头部+长度+压缩JSON)
  std::string _output_path;
  std::ofstream _ofs;
  std::thread _io_thread;  // 事件循环线程
  std::atomic<bool> _stopped{false};
  std::mutex _mtx;
  std::condition_variable _cv;
  std::atomic<bool> _connected{false};       // 连接状态
  std::atomic<bool> _finished{false};        // 请求完成状态
  std::atomic<bool> _thread_running{false};  // 线程运行状态
  bool _handshake_auth_added = false;
};

}  // namespace Tts

#endif  // TTS_CLIENT_TTS_CLIENT_HPP