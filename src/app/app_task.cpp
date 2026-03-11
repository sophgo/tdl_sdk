#include "app/app_task.hpp"
#include <fstream>
#include <iostream>
#include "framework/utils/tdl_log.hpp"
#include "nn/tdl_model_factory.hpp"
AppTask::AppTask(const std::string &task_name,
                 const std::string &json_config_file, bool skip_input_alloc) {
  task_name_ = task_name;
  skip_input_alloc_ = skip_input_alloc;
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
    PtrFrameInfo frame_info = std::make_unique<PipelineFrameInfo>();
    frame_info->node_data_["image"] = Packet::make(image);
    frame_info->frame_id_ = frame_id;
    return pipeline_channels_[pipeline_name]->setPipelineFrame(
        std::move(frame_info));

  } else {
    LOGE("pipeline %s not found!\n", pipeline_name.c_str());
    return -1;
  }
}

bool AppTask::isExternalFrameChannel(const std::string &channel_name) {
  if (pipeline_channels_.count(channel_name) == 0) {
    LOGE("channel %s not found!\n", channel_name.c_str());
    assert(false);
    return false;
  } else {
    return pipeline_channels_[channel_name]->isExternalFrame();
  }
}

int AppTask::getChannelMaxProcessingNum(const std::string &channel_name) {
  if (pipeline_channels_.count(channel_name) == 0) {
    LOGE("channel %s not found!\n", channel_name.c_str());
    assert(false);
    return 0;
  } else {
    return pipeline_channels_[channel_name]->getMaxProcessingNum();
  }
}

std::shared_ptr<BaseModel> AppTask::createModel(ModelType model_type) {
  if (skip_input_alloc_) {
    // 使用 skip_input_alloc = true 的方式创建模型
    std::shared_ptr<BaseModel> model =
        TDLModelFactory::getInstance().getModelWithoutOpen(model_type);
    if (model == nullptr) {
      LOGE("Failed to get model without open, model_type: %d",
           static_cast<int>(model_type));
      return nullptr;
    }

    // 设置 skip_input_alloc = true
    NetParam &net_param = model->getNetParam();
    net_param.skip_input_alloc = true;

    // 打开模型（此时 addInput 不会分配输入 tensor 内存）
    int32_t ret = model->modelOpen();
    if (ret != 0) {
      LOGE("Failed to open model, model_type: %d",
           static_cast<int>(model_type));
      return nullptr;
    }

    return model;
  } else {
    // 默认方式：直接创建并打开模型（会分配输入 tensor 内存）
    return TDLModelFactory::getInstance().getModel(model_type);
  }
}
