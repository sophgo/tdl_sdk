#include "unified_api_client.hpp"
#include <iostream>

UnifiedApiClient::UnifiedApiClient() {
  initClients();
  registerMethods();
}

UnifiedApiClient::~UnifiedApiClient() = default;

void UnifiedApiClient::initClients() {
  sophnetClient = std::make_unique<APIClient::SophnetClient>();
  volcengineClient = std::make_unique<APIClient::VolcengineClient>();
  aliyunClient = std::make_unique<APIClient::AliyunClient>();
  // asrClient = std::make_unique<Asr::AsrClient>();
  // ttsClient = std::make_unique<Tts::TtsClient>();
}

void UnifiedApiClient::registerMethods() {
  class InlineAsrCallback : public Asr::AsrCallback {
   public:
    InlineAsrCallback(std::string &out, std::mutex &m,
                      std::condition_variable &cv, bool &done)
        : transcript(out), mtx(m), cv(cv), finished(done) {}

    void on_open(Asr::AsrClient *) override {
      std::cout << "on_open" << std::endl;
    }
    void on_message(Asr::AsrClient *, std::string msg) override {
      // 每收到一条就累积到 transcript
      // std::lock_guard<std::mutex> lk(mtx);
      // transcript += msg;
      std::cout << msg << std::endl;
      // 解析 JSON，sequence<0 时认为 final 并 notify
      {
        auto j = nlohmann::json::parse(msg);
        if (j.value("sequence", 0) < 0) {
          std::string last_text = j["result"][0].value("text", "");
          // 线程安全地写入 transcript
          {
            std::lock_guard<std::mutex> lk(mtx);
            transcript = last_text;
          }
          finished = true;
          cv.notify_one();
        }
      }
    }
    void on_close(Asr::AsrClient *) override {
      std::cout << "on_close " << std::endl;
    }
    void on_error(Asr::AsrClient *, std::string msg) override {
      std::cerr << "[ASR ERROR] " << msg << "\n";
      std::lock_guard<std::mutex> lk(mtx);
      finished = true;
      cv.notify_one();
    }

   private:
    std::string &transcript;
    std::mutex &mtx;
    std::condition_variable &cv;
    bool &finished;
  };

  // SophnetClient
  methodMap["sophnet"]["chat"] = [this](const nlohmann::json &p) {
    auto r = sophnetClient->chat(p.value("api_key", ""), p.value("text", ""));
    if (!r.success) return createErrorResponse(r.error_message);
    return nlohmann::json{{"status", "ok"}, {"content", r.content}};
  };
  methodMap["sophnet"]["analyzeImage"] = [this](const nlohmann::json &p) {
    auto r = sophnetClient->analyzeImage(
        p.value("api_key", ""), p.value("text", ""), p.value("image_path", ""));
    if (!r.success) return createErrorResponse(r.error_message);
    return nlohmann::json{{"status", "ok"}, {"content", r.content}};
  };

  // VolcengineClient
  methodMap["volcengine"]["stylizeImage"] = [this](const nlohmann::json &p) {
    auto r = volcengineClient->stylizeImage(
        p.value("ak", ""), p.value("sk", ""), p.value("req_key", ""),
        p.value("sub_req_key", ""), p.value("image_path", ""),
        p.value("output_path", ""));
    if (!r.success) return createErrorResponse(r.error_message);
    return nlohmann::json{{"status", "ok"}, {"content", r.content}};
  };
  methodMap["volcengine"]["backgroundChange"] =
      [this](const nlohmann::json &p) {
        auto r = volcengineClient->backgroundchange(
            p.value("ak", ""), p.value("sk", ""), p.value("ref_prompt", ""),
            p.value("image_path", ""), p.value("output_path", ""));
        if (!r.success) return createErrorResponse(r.error_message);
        return nlohmann::json{{"status", "ok"}, {"content", r.content}};
      };
  methodMap["volcengine"]["pictureToPicture"] =
      [this](const nlohmann::json &p) {
        auto r = volcengineClient->picturetopicture(
            p.value("ak", ""), p.value("sk", ""), p.value("ref_prompt", ""),
            p.value("image_path", ""), p.value("output_path", ""));
        if (!r.success) return createErrorResponse(r.error_message);
        return nlohmann::json{{"status", "ok"}, {"content", r.content}};
      };
  methodMap["volcengine"]["humanSegment"] = [this](const nlohmann::json &p) {
    auto r = volcengineClient->humansegment(
        p.value("ak", ""), p.value("sk", ""), p.value("image_path", ""),
        p.value("output_path", ""));
    if (!r.success) return createErrorResponse(r.error_message);
    return nlohmann::json{{"status", "ok"}, {"content", r.content}};
  };
  methodMap["volcengine"]["humanAgetrans"] = [this](const nlohmann::json &p) {
    int target_age = p.value("target_age", 0);
    auto r = volcengineClient->humanagetrans(
        p.value("ak", ""), p.value("sk", ""), target_age,
        p.value("image_path", ""), p.value("output_path", ""));
    if (!r.success) return createErrorResponse(r.error_message);
    return nlohmann::json{{"status", "ok"}, {"content", r.content}};
  };

  // AliyunClient
  methodMap["aliyun"]["imgeditor"] = [this](const nlohmann::json &p) {
    auto r = aliyunClient->imgeditor(
        p.value("api_key", ""), p.value("function", ""),
        p.value("image_path", ""), p.value("output_path", ""),
        p.value("ref_prompt", ""));
    if (!r.success) return createErrorResponse(r.error_message);
    return nlohmann::json{{"status", "ok"}, {"content", r.content}};
  };

  // —— ASRClient ——
  methodMap["asr"]["recognize"] = [this](const nlohmann::json &p) {
    Asr::AsrClient Client;
    // --- 准备同步等待的工具 ---
    std::mutex m;
    std::condition_variable cv;
    bool done = false;
    std::string transcript;

    // 1) 参数准备
    auto appid = p.value("appid", "");
    auto token = p.value("token", "");
    auto cluster = p.value("cluster", "");
    auto audio_path = p.value("audio_path", "");
    auto audio_format = p.value("audio_format", "raw");
    int channels = p.value("channels", 1);
    int sample_rate = p.value("sample_rate", 16000);
    int bit_depth = p.value("bit_depth", 16);

    // 2) 初始化 client + callback
    InlineAsrCallback cb(transcript, m, cv, done);
    Client.set_appid(appid);
    Client.set_token(token);
    Client.set_cluster(cluster);
    Client.set_audio_format(audio_format, channels, sample_rate, bit_depth);
    Client.set_callback(&cb);

    // 3) 建立连接
    if (!Client.sync_connect()) {
      return createErrorResponse("ASR connect failed");
    }

    // 4) 读文件 & 发音频
    std::ifstream ifs(audio_path, std::ios::binary);
    if (!ifs.is_open())
      return createErrorResponse("Failed to open audio: " + audio_path);
    const int CHUNK = 32000;
    std::unique_ptr<char[]> buf(new char[CHUNK]);
    while (true) {
      ifs.read(buf.get(), CHUNK);
      auto len = ifs.gcount();
      if (len <= 0) break;
      // 新版：所有数据都设为非终结帧
      std::string audio_str;
      audio_str.append(buf.get(), len);
      int ret = Client.send_audio(audio_str, false);
      if (ret != 0) {
        std::cout << "send audio result error, " << ret << std::endl;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    ifs.close();
    // 显式发一帧空数据，标记 EOF
    Client.send_audio("", /* is_last */ true);

    {
      std::unique_lock<std::mutex> lk(m);
      cv.wait(lk, [&] { return done; });
    }
    // 6) 返回结果
    Client.close();
    Client.stop();
    Client.join();
    return nlohmann::json{{"status", "ok"}, {"content", transcript}};
  };

  // —— TTSClient ——
  methodMap["tts"]["synthesize"] = [this](const nlohmann::json &p) {
    // 1. 准备参数
    auto appid = p.value("appid", "");
    auto token = p.value("token", "");
    auto cluster = p.value("cluster", "volcano_tts");
    auto voice_type = p.value("voice_type", "BV700_V2_streaming");
    auto encoding = p.value("encoding", "pcm");  // wav / pcm / ogg_opus / mp3
    auto text = p.value("text", "");
    auto output_path = p.value("output_path", "");
    auto operation =
        p.value("operation", "submit");  // “submit”=流式，“query”=非流式
    auto timeout_sec = p.value("timeout_sec", 15);

    // 2. 调用 TTS 同步合成
    auto client = std::make_unique<Tts::TtsClient>();
    client->set_appid(appid);
    client->set_token(token);
    client->set_cluster(cluster);
    client->set_voice_type(voice_type);
    client->set_encoding(encoding);

    std::cout << "[TTS] 发起合成请求...\n";
    auto t0 = std::chrono::steady_clock::now();

    // bool ok = ttsClient->sync_request(text, operation, output_path,
    // timeout_sec);
    bool ok = client->sync_request(text, operation, output_path, timeout_sec);
    auto t1 = std::chrono::steady_clock::now();
    auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    if (!ok) {
      std::cerr << "[TTS] 合成失败或超时\n";
      return createErrorResponse("TTS synth failed or timed out");
    }
    std::cout << "[TTS] 合成完成，耗时 " << ms << " ms，文件已保存到:\n  "
              << output_path << "\n";
    client->close();
    client->stop();
    client->join();
    // ttsClient->join();
    // 3. 返回合成文件路径
    return nlohmann::json{{"status", "ok"}, {"content", output_path}};
  };
}

nlohmann::json UnifiedApiClient::call(const std::string &clientType,
                                      const std::string &methodName,
                                      const nlohmann::json &params) {
  auto cit = methodMap.find(clientType);
  if (cit == methodMap.end())
    return createErrorResponse("Client not found: " + clientType);

  auto &m = cit->second;
  auto mit = m.find(methodName);
  if (mit == m.end())
    return createErrorResponse("Method not found: " + methodName);

  try {
    return mit->second(params);
  } catch (const std::exception &e) {
    return createErrorResponse(e.what());
  }
}

bool UnifiedApiClient::isClientInitialized(
    const std::string &clientType) const {
  return methodMap.count(clientType) > 0;
}

nlohmann::json UnifiedApiClient::createErrorResponse(
    const std::string &errorMsg) const {
  return nlohmann::json{{"status", "error"}, {"message", errorMsg}};
}