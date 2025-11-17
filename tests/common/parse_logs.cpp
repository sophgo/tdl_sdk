#include "parse_logs.hpp"

namespace fs = std::experimental::filesystem;
struct Record {
  std::string model = "N/A";
  std::string preprocess = "N/A";
  std::string tpu_time = "N/A";
  std::string post = "N/A";
  std::string total = "N/A";
  std::string cpu_avg = "N/A";
  std::string tpu_avg = "N/A";
  std::string cpu_max = "N/A";
  std::string tpu_max = "N/A";
  std::string model_memory = "N/A";
  std::string file_size = "N/A";
};

void summary(const std::string &log_dir, const std::string &output_md) {
  // \S表示匹配任何非空白字符 \s表示匹配任意空白字符 ...
  std::regex re_model(
      R"(model_path:\s*.*/([^/\s]+?)_(?:cv\d{3,4}x|bm\d{4}x).(?:bmodel|cvimodel))");
  std::regex re_timer(
      R"(\[Timer\]\s*preprocess:(\S+),tpu:(\S+),post:(\S+),total:(\S+))");
  std::regex re_cpu(R"(CPU usage:\s*avg=(\S+)%?\s+max=(\S+)%?)");
  // 支持两种 TPU 日志格式：带 max 或只有 avg
  std::regex re_tpu_with_max(R"(TPU usage:\s*avg=(\S+)%?\s+max=(\S+)%?)");
  std::regex re_tpu_avg_only(R"(TPU usage:\s*avg=(\S+)%?)");
  std::regex re_memory_required(
      R"(Model memory requirements:\s*([\d\.]+)\s*MB)");
  std::regex re_file_size(R"(file size:\s*([\d\.]+)\s*MB)");

  std::vector<Record> records;

  if (!fs::exists(log_dir)) {
    std::cerr << "[ERROR] Log directory not found: " << log_dir << std::endl;
    return;
  }

  for (const auto &entry : fs::directory_iterator(log_dir)) {
    // 仅统计文件名中包含 ".performance" 的日志，并限定扩展名为 .log
    const auto &p = entry.path();
    const std::string fname = p.filename().string();
    if (fname.find(".performance") == std::string::npos) continue;
    if (!fs::is_regular_file(p) || p.extension() != ".log") continue;

    std::ifstream fin(p);
    if (!fin) {
      std::cerr << "[WARN] Failed to open: " << p << std::endl;
      continue;
    }

    std::string content((std::istreambuf_iterator<char>(fin)),
                        std::istreambuf_iterator<char>());

    std::smatch match_model, match_timer, match_cpu, match_tpu, match_memory,
        match_file_size;

    Record r;
    if (std::regex_search(content, match_model, re_model)) {
      r.model = match_model[1].str();
    }

    if (std::regex_search(content, match_timer, re_timer)) {
      r.preprocess = match_timer[1];
      r.tpu_time = match_timer[2];
      r.post = match_timer[3];
      r.total = match_timer[4];
    }

    if (std::regex_search(content, match_memory, re_memory_required)) {
      r.model_memory = match_memory[1].str();
    }

    if (std::regex_search(content, match_file_size, re_file_size)) {
      r.file_size = match_file_size[1].str();
    }

    if (std::regex_search(content, match_cpu, re_cpu)) {
      r.cpu_avg = match_cpu[1];
      r.cpu_max = match_cpu[2];
    }

    if (std::regex_search(content, match_tpu, re_tpu_with_max)) {
      r.tpu_avg = match_tpu[1];
      r.tpu_max = match_tpu[2];
    } else if (std::regex_search(content, match_tpu, re_tpu_avg_only)) {
      r.tpu_avg = match_tpu[1];
    }

    records.push_back(r);
  }

  if (records.empty()) {
    std::cout << "No valid log files found in " << log_dir << std::endl;
    return;
  }

  // --- 输出 Markdown 文件 ---
  std::ofstream fout(output_md);
  fout << "# 性能测试汇总报告\n\n";
  fout << "| 模型名称 | preprocess(ms) | tpu(ms) | post(ms) | total(ms) | "
          "memory_required(mb) | file_size(mb) | CPU avg(%) | CPU max(%) | TPU "
          "avg(%) | TPU "
          "max(%) |\n";
  fout << "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n";

  for (const auto &r : records) {
    fout << "| " << r.model << " | " << r.preprocess << " | " << r.tpu_time
         << " | " << r.post << " | " << r.total << " | " << r.model_memory
         << " | " << r.file_size << " | " << r.cpu_avg << " | " << r.cpu_max
         << " | " << r.tpu_avg << " | " << r.tpu_max << " |\n";
  }

  fout.close();
  std::cout << "summary finished: " << output_md << std::endl;
}

// 兼容的无参包装：默认从 ./log 读取并生成 summary.md
void summary() { summary("./log", "summary.md"); }