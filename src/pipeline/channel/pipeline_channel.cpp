#include "pipeline/pipeline_channel.hpp"
#include "framework/utils/tdl_log.hpp"
PipelineChannel::PipelineChannel(std::string name, int32_t frame_buffer_size)
    : name_(name) {
  for (int32_t i = 0; i < frame_buffer_size; i++) {
    PtrFrameInfo frame_info = std::make_unique<PipelineFrameInfo>();
    free_queue_.push(std::move(frame_info));
  }
  LOGI("pipeline channel %s constructor, frame_buffer_size:%d", name_.c_str(),
       frame_buffer_size);
}

PipelineChannel::~PipelineChannel() {
  LOGI("pipeline channel %s destructor", name_.c_str());
  stop();

  for (auto &node : nodes_) {
    node->unregisterChannel(this);
  }
  nodes_.clear();
}

int32_t PipelineChannel::addNode(std::shared_ptr<PipelineNode> node) {
  if (nodes_.size() == 0) {
    node->setFristNode(true);
  }
  nodes_.push_back(node);
  node->registerChannel(this);
  return 0;
}

void PipelineChannel::start() {
  is_running_ = true;
  for (auto &node : nodes_) {
    node->start();
  }
}

void PipelineChannel::stop() { is_running_ = false; }

int32_t PipelineChannel::toNextNode(PipelineNode *node,
                                    PtrFrameInfo frame_info) {
  LOGI("channel:%s,to add frame to node:%s next,frame_id:%lu", name_.c_str(),
       node->getNodeName().c_str(), frame_info->frame_id_);
  if (node == nullptr) {
    LOGE("node is nullptr,channel:%s,node:%s", name_.c_str(),
         node->getNodeName().c_str());
    assert(false);
    return -1;
  }
  int node_idx = -1;
  for (size_t i = 0; i < nodes_.size(); i++) {
    if (nodes_[i].get() == node) {
      node_idx = (int)i;
      break;
    }
  }
  if (node_idx == -1) {
    LOGE("node not found,channel:%s,node:%s", name_.c_str(),
         nodes_[node_idx]->getNodeName().c_str());
    assert(false);
    return -1;
  }
  if (node_idx == static_cast<int>(nodes_.size()) - 1) {
    LOGI("channel:%s,to add final frame,size:%d,frame_id:%lu", name_.c_str(),
         int(final_queue_.sizeUnsafe()), frame_info->frame_id_);
    final_queue_.push(std::move(frame_info));
    LOGI("channel:%s,add final frame done,size:%d", name_.c_str(),
         int(final_queue_.sizeUnsafe()));
  } else {
    nodes_[node_idx + 1]->addProcessFrame(this, std::move(frame_info));
  }
  return 0;
}

PtrFrameInfo PipelineChannel::getProcessedFrame(int wait_ms) {
  return final_queue_.pop(wait_ms);
}

PtrFrameInfo PipelineChannel::getFreeFrame(int wait_ms) {
  LOGI("channel:%s,to get free frame,size:%d", name_.c_str(),
       int(free_queue_.sizeUnsafe()));
  return free_queue_.pop(wait_ms);
}

int32_t PipelineChannel::addFreeFrame(PtrFrameInfo frame_info) {
  LOGI("channel:%s,to add free frame,size:%d,frame_id:%lu", name_.c_str(),
       int(free_queue_.sizeUnsafe()), frame_info->frame_id_);
  if (frame_info == nullptr) {
    LOGE("frame_info is nullptr,channel:%s", name_.c_str());
    assert(false);
    return -1;
  }
  if (clear_frame_func_) {
    clear_frame_func_(frame_info);
  } else {
    LOGW("not cleared frame added to free frame,channel:%s", name_.c_str());
  }
  LOGI("channel:%s,add free frame,size:%d,frame_id:%lu", name_.c_str(),
       int(free_queue_.sizeUnsafe()), frame_info->frame_id_);
  free_queue_.push(std::move(frame_info));
  LOGI("channel:%s,add free frame done,size:%d", name_.c_str(),
       int(free_queue_.sizeUnsafe()));
  return 0;
}
