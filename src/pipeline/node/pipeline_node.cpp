#include "pipeline/pipeline_node.hpp"
#include <unistd.h>
#include "framework/utils/tdl_log.hpp"
#include "pipeline/pipeline_channel.hpp"
PipelineNode::PipelineNode(Packet worker, int max_pending_frame)
    : worker_(worker), max_pending_frame_(max_pending_frame) {
  is_running_ = false;
}

PipelineNode::~PipelineNode() {}

void *PipelineNode::process(void *arg) {
  PipelineNode *node = (PipelineNode *)arg;

  int wait_ts = 5;
  LOGI("pipeline node %s process start", node->name_.c_str());
  while (node->is_running_) {
    if (pthread_mutex_trylock(&node->lock_) !=
        0) {  // to avoid dead lock when npuvideo shutdown
      usleep(100);
      continue;
    }
    auto channels = node->channels_;
    pthread_mutex_unlock(&node->lock_);
    if (channels.size() == 0) {
      usleep(1000);
      continue;
    }
    std::vector<PtrFrameInfo> batch_frames;
    std::vector<PipelineChannel *> batch_channels;
    for (auto &channel : channels) {
      PipelineChannel *p_chn = channel;

      if (!p_chn->isRunning()) {
        LOGI("channel:%s,is not running,skip", p_chn->name().c_str());
        usleep(100);
        continue;
      }

      PtrFrameInfo frame_info = nullptr;
      LOGI("node:%s,to get process frame,input_queues_size:%d",
           node->name_.c_str(), int(node->input_queues_[p_chn].sizeUnsafe()));
      if (node->is_frist_node_) {
        frame_info = p_chn->getFreeFrame(wait_ts);
      } else {
        frame_info = node->input_queues_[p_chn].pop(wait_ts);
      }
      if (frame_info == nullptr) {
        continue;
      }
      batch_channels.push_back(p_chn);
      batch_frames.push_back(std::move(frame_info));
    }

    if (batch_frames.size() == 0) {
      usleep(100);
      LOGI("node:%s,no frame to process", node->name_.c_str());
      continue;
    }
    LOGI("node:%s,got process frame,size:%d", node->name_.c_str(),
         int(batch_frames.size()));
    for (auto &frame_info : batch_frames) {
      if (node->process_func_) {
        int32_t ret = node->process_func_(frame_info, node->worker_);
        if (ret != 0) {
          LOGE("process func return %d", ret);
          // assert(false);
        }
        LOGI("node:%s,process frame done,frame_id:%lu", node->name_.c_str(),
             frame_info->frame_id_);
      }
    }
    LOGI("node:%s,to send frame to next node,size:%d", node->name_.c_str(),
         int(batch_frames.size()));
    for (size_t i = 0; i < batch_frames.size(); i++) {
      batch_channels[i]->toNextNode(node, std::move(batch_frames[i]));
    }
    batch_frames.clear();
    batch_channels.clear();
  }
  LOGI("pipeline node %s process end", node->name_.c_str());
  return nullptr;
}

void PipelineNode::registerChannel(PipelineChannel *p_chn) {
  pthread_mutex_lock(&lock_);
  channels_.push_back(p_chn);
  pthread_mutex_unlock(&lock_);
}

void PipelineNode::unregisterChannel(PipelineChannel *p_chn) {
  pthread_mutex_lock(&lock_);
  if (std::find(channels_.begin(), channels_.end(), p_chn) != channels_.end()) {
    channels_.erase(std::find(channels_.begin(), channels_.end(), p_chn));
  } else {
    LOGE("channel not found,node:%s,channel:%s", name_.c_str(),
         p_chn->name().c_str());
    assert(false);
  }
  pthread_mutex_unlock(&lock_);
  if (channels_.size() == 0) {
    LOGI("pipeline node %s unregister all channels, stop", name_.c_str());
    stop();
  }
}

int32_t PipelineNode::addProcessFrame(PipelineChannel *p_chn,
                                      PtrFrameInfo frame_info) {
  LOGI("node:%s,to add process frame,channel:%s,frame_id:%lu", name_.c_str(),
       p_chn->name().c_str(), frame_info->frame_id_);
  if (std::find(channels_.begin(), channels_.end(), p_chn) == channels_.end()) {
    LOGE("channel not found,node:%s,channel:%s", name_.c_str(),
         p_chn->name().c_str());
    assert(false);
    return -1;
  }
  input_queues_[p_chn].push(std::move(frame_info));
  if (input_queues_[p_chn].size() > static_cast<size_t>(max_pending_frame_)) {
    LOGE("drop frame in channel:%s,node:%s", p_chn->name().c_str(),
         name_.c_str());
    PtrFrameInfo frame_info = input_queues_[p_chn].pop();
    p_chn->addFreeFrame(std::move(frame_info));
  }
  LOGI("node:%s,add process frame done,channel:%s,size:%d", name_.c_str(),
       p_chn->name().c_str(), int(input_queues_[p_chn].sizeUnsafe()));
  return 0;
}

int32_t PipelineNode::setFristNode(bool is_frist_node) {
  is_frist_node_ = is_frist_node;
  return 0;
}

void PipelineNode::setName(std::string name) { name_ = name; }

int32_t PipelineNode::start() {
  is_running_ = true;
  if (thread_ == 0) {
    LOGI("pipeline node %s start,to create thread", name_.c_str());
    pthread_create(&thread_, nullptr, process, this);
  }
  return 0;
}

int32_t PipelineNode::stop() {
  is_running_ = false;
  LOGI("pipeline node %s stop,to join thread", name_.c_str());
  pthread_join(thread_, nullptr);
  thread_ = 0;
  return 0;
}

void PipelineNode::setProcessFunc(
    std::function<int32_t(PtrFrameInfo &, Packet &)> process_func) {
  process_func_ = process_func;
}
