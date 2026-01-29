#include "image_analysis_task.hpp"
#include <algorithm>
#include <chrono>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>
#include "network/api_poster/api_client.hpp"

namespace fs = std::experimental::filesystem;

ImageAnalysisTask::ImageAnalysisTask(const std::string& data_path)
    : data_path_(data_path), time_str_("") {}

json ImageAnalysisTask::handle_event(const json& request,
                                     const std::string& description) {
  // Not used for cyclic task in this design, but good to have
  return json();
}

json ImageAnalysisTask::run_analysis_step() {
  std::string analysis_dir = data_path_ + "/llm_analysis";

  if (!fs::exists(analysis_dir) || !fs::is_directory(analysis_dir)) {
    LOGI("analysis_dir %s not exists or not a directory", analysis_dir.c_str());
    return json();
  }

  std::vector<std::string> date_dirs;
  for (const auto& entry : fs::directory_iterator(analysis_dir)) {
    if (fs::is_directory(entry.status())) {
      std::string dir_name = entry.path().filename().string();
      if (dir_name.length() >= 15 && dir_name.find('_') == 8) {
        date_dirs.push_back(dir_name);
      }
    }
  }

  if (date_dirs.empty()) {
    LOGI("no date dir found in %s", analysis_dir.c_str());
    return json();
  }

  // 按日期时间排序，获取最新的目录
  std::sort(date_dirs.rbegin(), date_dirs.rend());
  std::string latest_dir = date_dirs[0];

  if (latest_dir == time_str_) {
    return json();
  }

  time_str_ = latest_dir;
  std::string latest_dir_path = analysis_dir + "/" + latest_dir;

  std::vector<std::string> jpg_files;
  for (const auto& entry : fs::directory_iterator(latest_dir_path)) {
    if (fs::is_regular_file(entry.status()) &&
        entry.path().extension() == ".jpg") {
      jpg_files.push_back(entry.path().string());
    }
  }

  if (jpg_files.empty()) {
    LOGI("no jpg file found in %s", latest_dir_path.c_str());
    return json();
  }

  json result;
  result["event"] = "image_analysis";
  result["payload"]["message_list"] = json::array();

  for (const auto& jpg_path : jpg_files) {
    std::string base64_data =
        APIClient::CommonFunctions::loadImageAsBase64(jpg_path);
    if (!base64_data.empty()) {
      json image_info;
      image_info["image"] = base64_data;
      image_info["frame_id"] = fs::path(jpg_path).stem().string();
      result["payload"]["message_list"].push_back(image_info);
    }
  }

  if (result["payload"]["message_list"].empty()) {
    LOGI("no image info found in %s", latest_dir_path.c_str());
    return json();
  }

  LOGI("image_analysis_task message_list size %d\n", jpg_files.size());

  return result;
}
