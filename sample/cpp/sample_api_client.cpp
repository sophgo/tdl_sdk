#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <json.hpp>
#include <sstream>
#include <thread>
#include "network/api_poster/unified_api_client.hpp"
#if defined(__CV180X__) || defined(__CV181X__) || defined(__CV184X__)
#include "network/api_poster/voice_chat_client.hpp"
#endif

using json = nlohmann::json;

static std::atomic<bool> g_stop{false};

static void signal_handler(int sig) { g_stop = true; }

static void usage(const char* prog) {
  std::cerr
      << "Usage:\n"
      << "  " << prog << " <client> <function> <params_json_or_path>\n\n"
      << "Examples:\n"
      << "  " << prog
      << " volcengine pictureToPicture '{\"ak\":\"...\",\"sk\":\"...\",...}'\n"
      << "  " << prog << " volcengine pictureToPicture /path/to/params.json\n";
  std::exit(1);
}

// 读取整个文件到 string
static std::string readFile(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs) {
    throw std::runtime_error("Cannot open file: " + path);
  }
  std::ostringstream buf;
  buf << ifs.rdbuf();
  return buf.str();
}

int main(int argc, char* argv[]) {
  // 优先使用命令行参数；也可改为 getenv("CLIENT") 之类
  if (argc != 4) {
    usage(argv[0]);
  }

  std::string client = argv[1];
  std::string function = argv[2];
  std::string p3 = argv[3];

  // 判断第三个参数：如果以 '{' 开头，就当 JSON 串；否则当作文件路径加载
  std::string json_text;
  try {
    if (!p3.empty() && p3.front() == '{') {
      json_text = p3;
    } else {
      json_text = readFile(p3);
    }
  } catch (const std::exception& e) {
    std::cerr << "Error reading params: " << e.what() << "\n";
    return 1;
  }

  json params;
  try {
    params = json::parse(json_text);
  } catch (const std::exception& e) {
    std::cerr << "Error parsing JSON params: " << e.what() << "\n";
    return 1;
  }

  // 调用统一 API
  UnifiedApiClient apiClient;

#if defined(__CV180X__) || defined(__CV181X__) || defined(__CV184X__)
  if (client == "doubao" && function == "voice_chat") {
    signal(SIGINT, signal_handler);

    json p = params;
    // Start Chat
    json result = apiClient.call("doubao", "voice_chat", p);
    if (result.contains("error")) {
      std::cerr << "VoiceChat start failed: " << result["error"] << std::endl;
      return 1;
    }

    std::cout << "VoiceChat client started. Press Ctrl+C to stop." << std::endl;

    while (!g_stop) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    apiClient.call("voice_chat", "stop", {});
    return 0;
  }
#endif
  auto resp = apiClient.call(client, function, params);

  if (resp.value("status", "") == "ok") {
    std::cout << "[" << client << "." << function
              << "] 输出文件: " << resp["content"].get<std::string>() << "\n";
  } else {
    std::cerr << "[" << client << "." << function
              << "] 错误: " << resp.value("message", "") << "\n";
    return 1;
  }

  return 0;
}