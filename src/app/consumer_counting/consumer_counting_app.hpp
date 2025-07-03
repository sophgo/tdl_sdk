#ifndef CONSUMER_COUNTING_APP_HPP
#define CONSUMER_COUNTING_APP_HPP

#include "app/app_task.hpp"
#include "components/video_decoder/video_decoder_type.hpp"
#include "nn/tdl_model_factory.hpp"

class ConsumerCountingAPP : public AppTask {
 public:
  ConsumerCountingAPP(const std::string &task_name,
                      const std::string &json_config);
  ~ConsumerCountingAPP() {}

  int32_t addPipeline(const std::string &pipeline_name,
                      int32_t frame_buffer_size,
                      const nlohmann::json &nodes_cfg);
  int32_t getResult(const std::string &pipeline_name, Packet &result) override;
  int32_t init() override;
  int32_t release() override;

 private:
  std::shared_ptr<PipelineNode> getVideoNode(const nlohmann::json &node_config);
  std::shared_ptr<PipelineNode> getHeadPersonDetectionNode(
      const nlohmann::json &node_config);
  std::shared_ptr<PipelineNode> getTrackNode(const nlohmann::json &node_config);
  std::shared_ptr<PipelineNode> ConsumerCountingNode(
      const nlohmann::json &node_config);

  std::string model_dir_;

  std::map<std::string, std::shared_ptr<BaseModel>> model_map_;

  NodeFactory node_factory_;
};

#endif
