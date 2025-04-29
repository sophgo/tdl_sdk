
#include "pipeline/pipeline_node.hpp"

std::shared_ptr<PipelineNode> NodeFactory::createModelNode(
    std::shared_ptr<BaseModel> model) {
  std::shared_ptr<PipelineNode> node = nullptr;
  if (model_node_instances_.find(model) != model_node_instances_.end()) {
    node = model_node_instances_[model];
  } else {
    Packet worker = Packet::Make(model);
    node = std::make_shared<PipelineNode>(worker);
    model_node_instances_[model] = node;
  }

  return node;
}
