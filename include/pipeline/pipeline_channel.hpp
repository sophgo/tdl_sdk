#ifndef FILE_PIPELINE_CHANNEL_HPP
#define FILE_PIPELINE_CHANNEL_HPP

#include "pipeline/pipeline_node.hpp"

class PipelineChannel {
 public:
  PipelineChannel(std::string name, int32_t frame_buffer_size);
  ~PipelineChannel();

  int32_t addNode(std::shared_ptr<PipelineNode> node);
  int32_t toNextNode(PipelineNode *node, PtrFrameInfo frame_info);
  // to get the frame processed by last node
  PtrFrameInfo getProcessedFrame(int wait_ms = 5);
  PtrFrameInfo getFreeFrame(int wait_ms = 5);
  // to add the frame to first node
  int32_t addFreeFrame(PtrFrameInfo frame_info);
  std::string getNodeName(size_t index);
  int32_t setPipelineFrame(PtrFrameInfo frame_info);
  void setExternalFrame(bool external_frame) {
    external_frame_ = external_frame;
  }
  bool isExternalFrame() { return external_frame_; }
  int getMaxProcessingNum();
  void start();
  void stop();
  bool isRunning() { return is_running_; }
  void setClearFrameFunc(std::function<void(PtrFrameInfo &)> clear_frame_func) {
    clear_frame_func_ = clear_frame_func;
  }
  std::string name() { return name_; }
  std::shared_ptr<PipelineNode> getNode(const std::string &node_name);

 private:
  BlockingQueue<PtrFrameInfo> final_queue_;
  BlockingQueue<PtrFrameInfo> free_queue_;
  std::function<void(PtrFrameInfo &)> clear_frame_func_ = nullptr;
  std::string name_;
  // BlockingQueue<PtrFrameInfo> free_queue_;

  std::vector<std::shared_ptr<PipelineNode>> nodes_;
  bool is_running_ = false;
  bool external_frame_ = true;
};
#endif
