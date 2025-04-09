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
#include "qwen2VL.hpp"          // Qwen2VL模型
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

class PyQwen2VL {
 private:
  std::unique_ptr<Qwen2VL> model_;
  int device_id_;

 public:
  PyQwen2VL() : device_id_(0), model_(std::make_unique<Qwen2VL>()) {}

  void init(int device_id, const std::string& model_path) {
    device_id_ = device_id;
    model_->init(device_id, model_path);
  }

  void deinit() {
    if (model_) {
      model_->deinit();
    }
  }

  int forward_first(const std::vector<int>& tokens,
                    const std::vector<int>& position_id,
                    const std::vector<float>& pixel_values,
                    const std::vector<int>& posids,
                    const std::vector<float>& attnmask, int img_offset,
                    int pixel_num) {
    // 复制输入，因为原始函数参数是非const引用
    std::vector<int> tokens_copy = tokens;
    std::vector<int> position_id_copy = position_id;
    std::vector<float> pixel_values_copy = pixel_values;
    std::vector<int> posids_copy = posids;
    std::vector<float> attnmask_copy = attnmask;

    return model_->forward_first(tokens_copy, position_id_copy,
                                 pixel_values_copy, posids_copy, attnmask_copy,
                                 img_offset, pixel_num);
  }

  int forward_next() { return model_->forward_next(); }

  py::dict get_model_info() const {
    py::dict info;
    if (model_) {
      info["token_length"] = model_->token_length;
      info["SEQLEN"] = model_->SEQLEN;
      info["HIDDEN_SIZE"] = model_->HIDDEN_SIZE;
      info["NUM_LAYERS"] = model_->NUM_LAYERS;
      info["MAX_POS"] = model_->MAX_POS;
      info["generation_mode"] = model_->generation_mode;
      info["MAX_PIXELS"] = model_->MAX_PIXELS;
    }
    return info;
  }
};

}  // namespace pytdl

#endif  // TDL_SDK_PYTHON_PY_LLM_HPP