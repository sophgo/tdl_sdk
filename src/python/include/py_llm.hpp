#ifndef TDL_SDK_PYTHON_PY_LLM_HPP
#define TDL_SDK_PYTHON_PY_LLM_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "model/llm_model.hpp"  // LLMModel基类
#include "qwen.hpp"             // Qwen模型
#include "utils/tdl_log.hpp"    // 日志
namespace py = pybind11;
namespace pytdl {

// LLM参数结构体
struct LLMInferParam {
  int max_new_tokens{2048};
  float top_p{0.7f};
  float temperature{0.95f};
  float repetition_penalty{1.0f};
  int repetition_last_n{64};
  std::string generation_mode{"chat"};
  std::string prompt_mode{"chat"};
};

// Python LLM包装类
class PyLLMBase {
 protected:
  std::shared_ptr<LLMModel> model_;

 public:
  PyLLMBase() = default;
  virtual ~PyLLMBase() = default;

  void modelOpen(const std::string& model_path) {
    int ret = model_->modelOpen(model_path);
  }

  void modelClose() {
    if (model_) {
      std::cout << "Closing model..." << std::endl;
      model_->onModelClosed();
      std::cout << "Model closed successfully" << std::endl;
      model_.reset();
    }
  }

  int inferenceFirst(const std::vector<int>& input_tokens) {
    int output_token = 0;
    int ret = model_->inferenceFirst(input_tokens, output_token);

    return output_token;
  }

  int inferenceNext() {
    int output_token = 0;
    int ret = model_->inferenceNext(output_token);

    return output_token;
  }

  std::vector<int> inferenceGenerate(const std::vector<int>& input_tokens,
                                     int eos_token) {
    std::vector<int> output_tokens;
    int ret = model_->inferenceGenerate(input_tokens, eos_token, output_tokens);

    return output_tokens;
  }

  py::dict getInferParam() const {
    auto param = model_->getInferParam();
    py::dict d;
    d["max_new_tokens"] = param.max_new_tokens;
    d["top_p"] = param.top_p;
    d["temperature"] = param.temperature;
    d["repetition_penalty"] = param.repetition_penalty;
    d["repetition_last_n"] = param.repetition_last_n;
    d["generation_mode"] = param.generation_mode;
    d["prompt_mode"] = param.prompt_mode;
    return d;
  }
};

class PyQwen : public PyLLMBase {
 public:
  PyQwen() { model_ = std::make_shared<Qwen>(); }
};

}  // namespace pytdl

#endif  // TDL_SDK_PYTHON_PY_LLM_HPP