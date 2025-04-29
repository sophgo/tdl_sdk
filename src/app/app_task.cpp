#include "app/app_task.hpp"
#include <fstream>
#include <iostream>
#include "framework/utils/tdl_log.hpp"
AppTask::AppTask(const std::string &task_name,
                 const std::string &json_config_file) {
  task_name_ = task_name;
  std::ifstream inf(json_config_file);
  if (!inf.is_open()) {
    throw std::runtime_error("Unable to open JSON config file: " +
                             json_config_file);
  }

  try {
    // 2. 直接用 >> 运算符读入
    inf >> json_config_;
  } catch (const nlohmann::json::parse_error &e) {
    std::cerr << "JSON parse error at byte " << e.byte << ": " << e.what()
              << std::endl;
    throw std::runtime_error("Failed to parse JSON config");
  }
}

AppTask::~AppTask() {}

std::vector<std::string> AppTask::getChannelNames() {
  std::vector<std::string> channel_names;
  for (auto &kv : pipeline_channels_) {
    channel_names.push_back(kv.first);
  }
  return channel_names;
}

int AppTask::getProcessingChannelNum() { return pipeline_channels_.size(); }

int32_t AppTask::removeChannel(const std::string &channel_name) {
  LOGI("to remove channel %s", channel_name.c_str());
  pipeline_channels_[channel_name]->stop();
  pipeline_channels_.erase(channel_name);
  return 0;
}
