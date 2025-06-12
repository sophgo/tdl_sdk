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

std::string AppTask::getChannelNodeName(const std::string &channel_name,
                                        size_t index) {
  if (pipeline_channels_.count(channel_name) == 0) {
    LOGE("channel %s not found!\n", channel_name.c_str());
    return std::string("");
  } else {
    return pipeline_channels_[channel_name]->getNodeName(index);
  }
}

int AppTask::getProcessingChannelNum() { return pipeline_channels_.size(); }

int32_t AppTask::removeChannel(const std::string &channel_name) {
  LOGI("to remove channel %s", channel_name.c_str());
  pipeline_channels_[channel_name]->stop();
  pipeline_channels_.erase(channel_name);
  return 0;
}

int32_t AppTask::setFrame(const std::string &pipeline_name,
                          std::shared_ptr<BaseImage> image, uint64_t frame_id) {
  if (pipeline_channels_.count(pipeline_name) != 0) {
    if (image->getWidth() == 0) {
      return 1;
    }
    PtrFrameInfo frame_info = std::make_unique<PipelineFrameInfo>();
    frame_info->node_data_["image"] = Packet::make(image);
    frame_info->frame_id_ = frame_id;
    pipeline_channels_[pipeline_name]->addFreeFrame(std::move(frame_info));
    return 0;

  } else {
    LOGE("pipeline %s not found!\n", pipeline_name.c_str());
    printf("pipeline %s not found!\n", pipeline_name.c_str());
    return -1;
  }
}
