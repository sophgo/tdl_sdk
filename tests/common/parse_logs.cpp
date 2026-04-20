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
      R"(model_path:\s*.*/([^/\s]+))");  // 捕获最后一个斜杠后的完整文件名
  std::regex re_timer(
      R"(\[Timer\]\s*preprocess:(\S+),tpu:(\S+),post:(\S+),total:(\S+))");
  std::regex re_cpu(R"(CPU usage:\s*avg=(\S+)%?\s+max=(\S+)%?)");
  // 支持两种 TPU 日志格式：带 max 或只有 avg
  std::regex re_tpu(R"(TPU usage:\s*avg=(\S+)%?\s+max=(\S+)%?)");
  std::regex re_memory_required(
      R"(Model memory requirements:\s*([\d\.]+)\s*MB)");
  std::regex re_file_size(R"(file size:\s*([\d\.]+)\s*MB)");

  std::vector<Record> records;

  if (!fs::exists(log_dir)) {
    std::cerr << "[ERROR] Log directory not found: " << log_dir << std::endl;
    return;
  }

  for (const auto &entry : fs::directory_iterator(log_dir)) {
    const auto &p = entry.path();
    const std::string fname = p.filename().string();

    // 仅统计文件名中包含 ".performance" 的日志，并限定扩展名为 .log
    if (fname.find(".performance") == std::string::npos) continue;
    if (!fs::is_regular_file(p) || p.extension() != ".log") continue;

    std::ifstream fin(p);
    if (!fin) {
      std::cerr << "[WARN] Failed to open: " << p << std::endl;
      continue;
    }

    std::string content((std::istreambuf_iterator<char>(fin)),
                        std::istreambuf_iterator<char>());

    // 收集每类信息的所有匹配（保持出现顺序）
    std::vector<std::string> v_model;
    std::vector<std::string> v_pre, v_tpu_time, v_post, v_total;
    std::vector<std::string> v_cpu_avg, v_cpu_max;
    std::vector<std::string> v_tpu_avg, v_tpu_max;
    std::vector<std::string> v_memory, v_filesize;

    auto it_end = std::sregex_iterator();

    // model
    for (auto it =
             std::sregex_iterator(content.begin(), content.end(), re_model);
         it != it_end; ++it) {
      v_model.push_back((*it)[1].str());
    }

    // timer
    for (auto it =
             std::sregex_iterator(content.begin(), content.end(), re_timer);
         it != it_end; ++it) {
      v_pre.push_back((*it)[1].str());
      v_tpu_time.push_back((*it)[2].str());
      v_post.push_back((*it)[3].str());
      v_total.push_back((*it)[4].str());
    }

    // memory
    for (auto it = std::sregex_iterator(content.begin(), content.end(),
                                        re_memory_required);
         it != it_end; ++it) {
      v_memory.push_back((*it)[1].str());
    }

    // file size
    for (auto it =
             std::sregex_iterator(content.begin(), content.end(), re_file_size);
         it != it_end; ++it) {
      v_filesize.push_back((*it)[1].str());
    }

    // CPU usage
    for (auto it = std::sregex_iterator(content.begin(), content.end(), re_cpu);
         it != it_end; ++it) {
      v_cpu_avg.push_back((*it)[1].str());
      v_cpu_max.push_back((*it)[2].str());
    }

    // TPU usage
    for (auto it = std::sregex_iterator(content.begin(), content.end(), re_tpu);
         it != it_end; ++it) {
      v_tpu_avg.push_back((*it)[1].str());
      v_tpu_max.push_back((*it)[2].str());
    }

    // 计算本文件应生成的 Record 数量：
    auto max_size = [](std::initializer_list<size_t> l) {
      size_t m = 0;
      for (size_t s : l) m = std::max(m, s);
      return m;
    };

    size_t n = max_size({v_model.size(), v_memory.size(), v_filesize.size(),
                         v_cpu_avg.size(), v_cpu_max.size(), v_tpu_avg.size(),
                         v_tpu_max.size(), v_tpu_time.size(), v_post.size(),
                         v_total.size()});

    // 取值策略：
    // - 若该字段出现次数 = n：按索引取；否则用 N/A
    auto pick = [n](const std::vector<std::string> &v,
                    size_t i) -> std::string {
      if (v.size() != n) return "N/A";
      return v[i];
    };

    if (n == 0) {
      // 该文件没有任何有效匹配，push 一个默认 Record
      Record r;
      records.push_back(r);
    } else {
      for (size_t i = 0; i < n; ++i) {
        Record r;
        r.model = pick(v_model, i);
        r.preprocess = pick(v_pre, i);
        r.tpu_time = pick(v_tpu_time, i);
        r.post = pick(v_post, i);
        r.total = pick(v_total, i);
        r.model_memory = pick(v_memory, i);
        r.file_size = pick(v_filesize, i);
        r.cpu_avg = pick(v_cpu_avg, i);
        r.cpu_max = pick(v_cpu_max, i);
        r.tpu_avg = pick(v_tpu_avg, i);
        r.tpu_max = pick(v_tpu_max, i);
        records.push_back(r);
      }
    }
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