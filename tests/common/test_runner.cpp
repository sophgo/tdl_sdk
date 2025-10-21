#include <algorithm>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <json.hpp>
#include <sstream>
#include <string>
#include <vector>
#include "regression_utils.hpp"

namespace fs = std::experimental::filesystem;

struct TestSuiteConfig {
  std::string name;
  std::string test_type;
  std::vector<std::string> test_files;
};

std::vector<std::string> get_test_files(nlohmann::json &suite_config,
                                        const std::string &platform) {
  std::vector<std::string> files;

  if (suite_config.find("files") == suite_config.end()) {
    return files;
  }

  auto file = suite_config["files"];
  // extract files from common
  // exclude files from exclude
  if (file.find("common") != file.end()) {
    files.insert(files.end(), file["common"].begin(), file["common"].end());
  }
  std::cout << suite_config["name"] << " common files: " << files.size()
            << std::endl;
  if (file.find("exclude") != file.end()) {
    auto exclude = file["exclude"];
    if (exclude.find(platform) != exclude.end()) {
      for (auto &ex_file : exclude[platform]) {
        files.erase(std::remove(files.begin(), files.end(), ex_file),
                    files.end());
      }
    }
  }

  return files;
}
std::vector<TestSuiteConfig> get_test_suites(nlohmann::json &config) {
  std::vector<TestSuiteConfig> test_suites;
  std::string platform = cvitdl::unitest::get_platform_str();
  for (auto &suite : config["test_suites"]) {
    std::cout << "suite: " << suite << std::endl;
    TestSuiteConfig test_suite;
    test_suite.name = suite["name"];
    std::vector<std::string> platforms = suite["platforms"];
    if (std::find(platforms.begin(), platforms.end(), platform) ==
            platforms.end() &&
        std::find(platforms.begin(), platforms.end(), "ALL") ==
            platforms.end()) {
      continue;
    }
    test_suite.test_type = suite["type"];
    test_suite.test_files = get_test_files(suite, platform);
    if (test_suite.test_files.empty() && test_suite.test_type == "regression") {
      std::cout << "suite: " << suite["name"] << " no files to run"
                << std::endl;
      continue;
    }
    test_suites.push_back(test_suite);
  }
  return test_suites;
}

int main(int argc, char *argv[]) {
  if (argc != 4 && argc != 5 && argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " <config.json> <model_dir> <dataset_dir> "
                 "[test_flag(func,perf,generate_func,generate_perf)] "
                 "[test_type(regression,unit,all)]"
              << std::endl;
    return 1;
  }
  std::string config_path = argv[1];

  std::string model_dir = argv[2];
  std::string dataset_dir = argv[3];

  std::string test_flag = "func";
  if (argc == 5) {
    test_flag = argv[4];
  }

  std::string test_type = "all";
  if (argc == 6) {
    test_type = argv[5];
  }

  std::ifstream config_file(config_path);
  nlohmann::json config;
  config_file >> config;
  std::vector<TestSuiteConfig> test_suites = get_test_suites(config);

  std::cout << "model_dir: " << model_dir << std::endl;
  std::cout << "dataset_dir: " << dataset_dir << std::endl;
  std::cout << "test_flag: " << test_flag << std::endl;
  std::cout << "test_type: " << test_type << std::endl;
  std::cout << "test_suites: " << test_suites.size() << std::endl;

  int total_runs = 0;
  int failed_runs = 0;
  std::vector<std::string> failed_items;

  // prepare log directory for per-run outputs
  std::string log_dir;
  if (config.find("log_dir") != config.end()) {
    log_dir = config["log_dir"].get<std::string>();
  } else if (config.find("report_dir") != config.end()) {
    log_dir =
        (fs::path(config["report_dir"].get<std::string>()) / "logs").string();
  } else {
    log_dir = "";  // disabled
  }
  std::cout << "log_dir: " << log_dir << std::endl;
  if (!log_dir.empty()) {
    try {
      fs::path p(log_dir);
      if (!fs::exists(p)) fs::create_directories(p);
    } catch (...) {
      // ignore
      log_dir.clear();
    }
  }

  auto sanitize = [](const std::string &s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
      if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
          (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-') {
        out.push_back(c);
      } else {
        out.push_back('_');
      }
    }
    return out;
  };

  for (auto &suite : test_suites) {
    if (test_type != "all" && test_type != suite.test_type) {
      continue;
    }

    std::vector<std::string> files = suite.test_files;
    if (files.empty()) {
      files.push_back("");
    }
    std::cout << "suite: " << suite.name << " files: " << files.size()
              << std::endl;
    for (auto &file : files) {
      std::string cmd = "./test_main " + model_dir + " " + dataset_dir;
      if (!file.empty()) {
        cmd += " " + file + " " + test_flag;
      }
      cmd += " --gtest_filter=" + suite.name;

      std::string log_path;
      std::string cmd_exec = cmd;
      if (!log_dir.empty()) {
        std::string base = sanitize(suite.name);
        if (!file.empty()) base += std::string("__") + sanitize(file);
        base += ".log";
        log_path = (fs::path(log_dir) / base).string();
        cmd_exec += " > \"" + log_path + "\" 2>&1";
      }

      std::cout << cmd_exec << std::endl;
      int rc = system(cmd_exec.c_str());
      total_runs += 1;
      if (rc != 0) {
        failed_runs += 1;
        std::stringstream item;
        item << suite.name;
        if (!file.empty()) item << ":" << file;
        item << " (rc=" << rc << ")";
        if (!log_path.empty()) item << " log=\"" << log_path << "\"";
        item << " cmd=\"" << cmd_exec << "\"";
        failed_items.push_back(item.str());

        // print tail of log for quick context
        try {
          if (!log_path.empty()) {
            std::ifstream ifs(log_path);
            std::vector<std::string> lines;
            std::string l;
            while (std::getline(ifs, l)) lines.push_back(l);
            size_t tail = lines.size() > 50 ? 50 : lines.size();
            if (tail > 0) {
              std::cout << "---- tail of log (" << log_path << ") ----"
                        << std::endl;
              for (size_t i = lines.size() - tail; i < lines.size(); ++i) {
                std::cout << lines[i] << std::endl;
              }
              std::cout << "-----------------------------------------"
                        << std::endl;
            }
          }
        } catch (...) {
          // ignore
        }
      }
    }
  }
  std::cout << "================ Results ================" << std::endl;
  if (failed_runs == 0) {
    std::cout << "[" << total_runs << "/" << total_runs << "] ALL TEST PASSED"
              << std::endl;
  } else {
    std::cout << "[" << failed_runs << "/" << total_runs << "] TEST FAILED"
              << std::endl;
    for (const auto &it : failed_items) {
      std::cout << "  - " << it << std::endl;
    }
  }

  // optional: write report if report_dir provided
  try {
    if (config.find("report_dir") != config.end()) {
      std::string report_dir = config["report_dir"].get<std::string>();
      if (!report_dir.empty()) {
        fs::path rpt_dir(report_dir);
        if (!fs::exists(rpt_dir)) {
          fs::create_directories(rpt_dir);
        }
        fs::path rpt_path = rpt_dir / "test_runner_report.txt";
        std::ofstream ofs(rpt_path.string());
        if (ofs.is_open()) {
          ofs << "model_dir: " << model_dir << "\n";
          ofs << "dataset_dir: " << dataset_dir << "\n";
          ofs << "test_flag: " << test_flag << "\n";
          ofs << "test_type: " << test_type << "\n";
          ofs << "total_runs: " << total_runs << "\n";
          ofs << "failed_runs: " << failed_runs << "\n";
          if (!failed_items.empty()) {
            ofs << "failed_items:"
                << "\n";
            for (const auto &it : failed_items) ofs << "  - " << it << "\n";
          }
          ofs.close();
        }
      }
    }
  } catch (...) {
    // ignore file IO errors
  }
  return 0;
}