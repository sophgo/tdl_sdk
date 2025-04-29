#ifndef PIPELINE_NODE_HPP
#define PIPELINE_NODE_HPP

#include <functional>
#include <vector>
#include "framework/common/blocking_queue.hpp"
#include "framework/model/base_model.hpp"
#include "pipeline/pipeline_data_types.hpp"
class PipelineChannel;

class PipelineNode {
 public:
  PipelineNode(Packet worker, int max_pending_frame = 5);

  virtual ~PipelineNode();
  void setName(std::string name);
  std::string getNodeName() { return name_; }
  void setProcessFunc(
      std::function<int32_t(PtrFrameInfo &, Packet &)> process_func);
  virtual void registerChannel(PipelineChannel *p_chn);
  virtual void unregisterChannel(PipelineChannel *p_chn);
  static void *process(void *arg);

  int32_t addProcessFrame(PipelineChannel *p_chn, PtrFrameInfo frame_info);
  int32_t start();
  int32_t stop();
  int32_t setFristNode(bool is_frist_node);

 private:
  void init();  // create process thread
  int32_t max_pending_frame_;
  std::string name_;

 protected:
  std::vector<PipelineChannel *> channels_;
  Packet worker_;
  pthread_mutex_t lock_ = PTHREAD_MUTEX_INITIALIZER;
  pthread_t thread_ = 0;
  // data waiting for processing
  std::map<PipelineChannel *, BlockingQueue<PtrFrameInfo>> input_queues_;
  bool is_running_ = false;
  bool is_frist_node_ = false;
  std::function<int32_t(PtrFrameInfo &, Packet &)> process_func_ = nullptr;
};

class NodeFactory {
 public:
  // if same model,would return same node instance,this is singleton
  std::shared_ptr<PipelineNode> createModelNode(
      std::shared_ptr<BaseModel> model);
  //   std::shared_ptr<PipelineNode> createTrackerNode();

  std::map<std::shared_ptr<BaseModel>, std::shared_ptr<PipelineNode>>
      model_node_instances_;
};

#endif
