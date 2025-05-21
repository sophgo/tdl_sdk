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
#include "utils/qwen_vl_helper.hpp"
#include "utils/tdl_log.hpp"  // 日志
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

  void modelOpen(const std::string &model_path) {
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

  int inferenceFirst(const std::vector<int> &input_tokens) {
    int output_token = 0;
    int ret = model_->inferenceFirst(input_tokens, output_token);

    return output_token;
  }

  int inferenceNext() {
    int output_token = 0;
    int ret = model_->inferenceNext(output_token);

    return output_token;
  }

  std::vector<int> inferenceGenerate(const std::vector<int> &input_tokens,
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

// 添加PyQwen2VL类
class PyQwen2VL {
 public:
  // 添加成员变量以供 Python 绑定
  int SEQLEN;
  int token_length;
  int HIDDEN_SIZE;
  int NUM_LAYERS;
  int MAX_POS;
  int MAX_PIXELS;
  uint64_t VIT_DIMS;
  std::string generation_mode;
  PyQwen2VL() = default;
  ~PyQwen2VL() {
    if (model_) {
      model_->deinit();
      model_.reset();
    }
  }

  void init(int dev_id, const std::string &model_path) {
    if (!model_) {
      model_ = std::make_shared<Qwen2VL>();
    }
    model_->init(dev_id, model_path);
    // 初始化后同步成员变量
    this->SEQLEN = model_->SEQLEN;
    this->token_length = model_->token_length;
    this->HIDDEN_SIZE = model_->HIDDEN_SIZE;
    this->NUM_LAYERS = model_->NUM_LAYERS;
    this->MAX_POS = model_->MAX_POS;
    this->MAX_PIXELS = model_->MAX_PIXELS;
    this->VIT_DIMS = model_->VIT_DIMS;
  }

  void deinit() {
    if (model_) {
      model_->deinit();
      model_.reset();
    }
  }

  int forward_first(const std::vector<int> &tokens,
                    const std::vector<int> &position_ids,
                    const std::vector<float> &pixel_values,
                    const std::vector<int> &posids,
                    const std::vector<float> &attnmask, int img_offset,
                    int pixel_num) {
    if (!model_) {
      throw std::runtime_error("Model not initialized");
    }
    int result = model_->forward_first(
        const_cast<std::vector<int> &>(tokens),
        const_cast<std::vector<int> &>(position_ids),
        const_cast<std::vector<float> &>(pixel_values),
        const_cast<std::vector<int> &>(posids),
        const_cast<std::vector<float> &>(attnmask), img_offset, pixel_num);
    // 更新成员变量
    this->token_length = model_->token_length;
    return result;
  }

  int forward_next() {
    if (!model_) {
      throw std::runtime_error("Model not initialized");
    }
    int result = model_->forward_next();
    // 更新成员变量
    this->token_length = model_->token_length;
    return result;
  }

  void set_generation_mode(const std::string &mode) {
    if (!model_) {
      throw std::runtime_error("Model not initialized");
    }
    model_->generation_mode = mode;
  }

  std::string get_generation_mode() const {
    if (!model_) {
      throw std::runtime_error("Model not initialized");
    }
    return model_->generation_mode;
  }

 private:
  std::shared_ptr<Qwen2VL> model_;
};

// 封装QwenVLHelper::fetchVideo为Python可调用的函数
py::array_t<float> fetch_video(const std::string &video_path,
                               const double desired_fps = 2.0,
                               const int desired_nframes = 0,
                               const int max_video_sec = 0) {
  std::vector<std::vector<cv::Mat>> frames = QwenVLHelper::fetchVideo(
      video_path, desired_fps, desired_nframes, max_video_sec);

  // 判断返回的帧是否为空
  if (frames.empty()) {
    return py::array_t<float>();  // 返回空数组
  }

  // 获取视频帧的维度信息
  int frame_count = frames.size();
  int channels = frames[0].size();  // 通常是3个通道 (BGR)
  int height = frames[0][0].rows;
  int width = frames[0][0].cols;

  // 创建返回的numpy数组，形状为 [frame_count, channels, height, width]
  py::array_t<float> py_result({frame_count, channels, height, width});

  // 获取数组的可写缓冲区
  py::buffer_info buf = py_result.request();
  float *ptr = static_cast<float *>(buf.ptr);

  // 复制数据：对每一帧的每个通道，先将数据类型转换为 float，再使用 memcpy
  // 写入目标数组 注意: 每个 cv::Mat 数据默认类型为 unsigned
  // char，因此需要先转换类型 如果通道数为3，则调整通道顺序从 BGR 转换为 RGB
  if (channels == 3) {
    // 定义映射，将目标顺序 RGB 对应到原始 BGR 的索引：
    // 目标通道 0 (R) 来自原始通道 2，目标通道 1 (G) 来自原始通道 1，
    // 目标通道 2 (B) 来自原始通道 0
    int channel_map[3] = {2, 1, 0};
    for (int f = 0; f < frame_count; ++f) {
      for (int c = 0; c < channels; ++c) {
        cv::Mat floatMat;
        // 使用映射后的通道，将 cv::Mat 从 unsigned char 转换为
        // float，不进行归一化
        frames[f][channel_map[c]].convertTo(floatMat, CV_32F);
        // 计算目标起始地址，每一帧有 channels * height * width 个 float 数据，
        // 当前通道在该帧中的偏移为 c * (height * width)
        float *dst =
            ptr + f * (channels * height * width) + c * (height * width);
        std::memcpy(dst, floatMat.data, height * width * sizeof(float));
      }
    }
  } else {
    // 如果通道数不为3，则直接按照原顺序复制
    for (int f = 0; f < frame_count; ++f) {
      for (int c = 0; c < channels; ++c) {
        cv::Mat floatMat;
        frames[f][c].convertTo(floatMat, CV_32F);
        float *dst =
            ptr + f * (channels * height * width) + c * (height * width);
        std::memcpy(dst, floatMat.data, height * width * sizeof(float));
      }
    }
  }

  return py_result;
}

// 封装QwenVLHelper::testFetchVideoTs为Python可调用的函数
py::dict test_fetch_video_ts(const std::string &video_path,
                             const double desired_fps = 2.0,
                             const int desired_nframes = 0,
                             const int max_video_sec = 0) {
  // 调用C++实现的testFetchVideoTs方法
  std::map<std::string, float> result = QwenVLHelper::testFetchVideoTs(
      video_path, desired_fps, desired_nframes, max_video_sec);

  // 将std::map转换为Python的dict
  py::dict py_result;
  for (const auto &pair : result) {
    py_result[pair.first.c_str()] = pair.second;
  }

  return py_result;
}

}  // namespace pytdl

#endif  // TDL_SDK_PYTHON_PY_LLM_HPP