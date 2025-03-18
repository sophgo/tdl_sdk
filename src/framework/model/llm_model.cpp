#include "model/llm_model.hpp"
#include "utils/tdl_log.hpp"
LLMModel::LLMModel() {}

LLMModel::~LLMModel() {}

int32_t LLMModel::modelOpen(const std::string& model_path) {
  std::vector<int> devices = {0};
  return llm_net_.init(devices, model_path);
}

int32_t LLMModel::inferenceFirst(const std::vector<int>& tokens,
                                 int& output_token) {
  if (tokens.size() == 0) {
    LOGE("tokens is empty");
    return -1;
  }
  output_token = llm_net_.forwardFirst(tokens);
  return 0;
}

int32_t LLMModel::inferenceNext(int& output_token) {
  output_token = llm_net_.forwardNext();
  return 0;
}

int32_t LLMModel::inferenceGenerate(const std::vector<int>& tokens, int EOS,
                                    std::vector<int>& output_tokens) {
  if (tokens.empty()) {
    printf("Sorry: your question is empty!!\n");
    return {};
  }

  // make sure token not too large
  int SEQLEN = llm_net_.getMaxSeqLen();
  int history_length = tokens.size();
  if (history_length > SEQLEN - 10) {
    printf("Error: your question is too large!\n");
    return {};
  }

  std::vector<int> result_tokens;
  int max_new_tokens = llm_net_.getInferParam().max_new_tokens;
  int token = llm_net_.forwardFirst(tokens);
  while (token != EOS && result_tokens.size() < SEQLEN &&
         result_tokens.size() <= history_length + max_new_tokens) {
    result_tokens.emplace_back(token);
    token = llm_net_.forwardNext();
  }

  output_tokens = result_tokens;
  return 0;
}