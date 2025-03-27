#ifndef LLM_MODEL_HPP
#define LLM_MODEL_HPP

#include "net/bm_llm_net.hpp"

class LLMModel {
 public:
  LLMModel();
  virtual ~LLMModel();
  int32_t modelOpen(const std::string& model_path);
  int32_t inferenceFirst(const std::vector<int>& tokens, int& output_token);
  int32_t inferenceNext(int& output_token);
  int32_t inferenceGenerate(const std::vector<int>& tokens, int EOS,
                            std::vector<int>& output_tokens);
  virtual int32_t onModelOpened() { return 0; }
  virtual int32_t onModelClosed() { return 0; }
  LLMInferParam getInferParam() { return llm_net_.getInferParam(); }

 protected:
  BMLLMNet llm_net_;
};

#endif  // LLM_BASE_MODEL_HPP