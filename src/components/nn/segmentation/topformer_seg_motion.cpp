#include "segmentation/topformer_seg_motion.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include "utils/detection_helper.hpp"

#include "common/ccl.hpp"
#include "utils/tdl_log.hpp"

namespace {

template <typename T>
void argmaxMask(const T* logits, int channels, int height, int width,
                float qscale, std::vector<uint8_t>& class_mask) {
  int hw = height * width;
  class_mask.resize(hw);
  for (int i = 0; i < hw; ++i) {
    float max_value = std::numeric_limits<float>::lowest();
    int max_class = 0;
    for (int c = 0; c < channels; ++c) {
      float value = static_cast<float>(logits[c * hw + i]) * qscale;
      if (value > max_value) {
        max_value = value;
        max_class = c;
      }
    }
    class_mask[i] = static_cast<uint8_t>(max_class);
  }
}
}  // namespace

TopformerSegMotion::TopformerSegMotion()
#if defined(__CMODEL_CV181X__)
    : ccl_instance_(nullptr),
#else
    : ccl_instance_(createConnectInstance()),
#endif
      cached_frame_count_(0),
      min_area_thresh_(256),
      with_mask_(false) {
  net_param_.model_config.mean = {0.0, 0.0, 0.0};
  net_param_.model_config.std = {254.97195, 254.97195, 254.97195};
  net_param_.model_config.rgb_order = "gray";
  keep_aspect_ratio_ = true;
}

TopformerSegMotion::~TopformerSegMotion() {
#if !defined(__CMODEL_CV181X__)
  destroyConnectedComponent(ccl_instance_);
#endif
  ccl_instance_ = nullptr;
}

int32_t TopformerSegMotion::inference(
    const std::shared_ptr<BaseImage>& image,
    std::shared_ptr<ModelOutputInfo>& out_data,
    const std::map<std::string, float>& parameters) {
  with_mask_ = (parameters.count("with_mask") != 0);

  if (image == nullptr) {
    LOGE("Input image is null");
    return -1;
  }
  if (preprocessor_ == nullptr) {
    LOGE("Preprocessor is not set");
    return -1;
  }

  if (parameters.count("min_area") != 0) {
    min_area_thresh_ = std::max(1, static_cast<int>(parameters.at("min_area")));
  }

  const std::vector<std::string>& input_names = net_->getInputNames();
  int num_inputs = input_names.size();
  if (num_inputs < 2) {
    LOGE("TopformerSegMotion requires at least 2 input layers, current:%d",
         num_inputs);
    return -1;
  }

  int current_input_idx = num_inputs - 1;
  const std::string& current_input_name = input_names[current_input_idx];
  batch_rescale_params_[current_input_name].clear();
  if (preprocess_params_.count(current_input_name) == 0) {
    LOGE("No preprocess params for input layer:%s", current_input_name.c_str());
    return -1;
  }

  model_timer_.TicToc("runstart");

  std::shared_ptr<BaseTensor> current_input_tensor =
      net_->getInputTensor(current_input_name);

  if (image->getImageType() == ImageType::TENSOR_FRAME) {
    if (net_->skipInputAlloc()) {
      int32_t ret = net_->setInputTensorFromImage(current_input_name, image);
      if (ret != 0) {
        LOGE("Failed to set input tensor from image");
        return -1;
      }

    } else {
      current_input_tensor->copyFromImage(image, 0);
    }
    batch_rescale_params_[current_input_name].push_back(
        std::vector<float>({1.0f, 1.0f, 0.0f, 0.0f}));

  } else {
    const PreprocessParams& preprocess_params =
        preprocess_params_[current_input_name];

    preprocessor_->preprocessToTensor(image, preprocess_params, 0,
                                      current_input_tensor);

    std::vector<float> rescale_params = preprocessor_->getRescaleConfig(
        preprocess_params, image->getWidth(), image->getHeight());

    batch_rescale_params_[current_input_name].push_back(rescale_params);
  }

  TensorInfo current_input_info = net_->getTensorInfo(current_input_name);
  if (current_input_info.sys_mem == nullptr ||
      current_input_info.tensor_size == 0) {
    LOGE("Invalid current input tensor memory");
    return -1;
  }

  std::vector<int8_t> current_preprocessed(current_input_info.tensor_size);
  std::memcpy(current_preprocessed.data(), current_input_info.sys_mem,
              current_input_info.tensor_size);

  std::shared_ptr<ModelBoxSegmentationInfo> empty_box =
      std::make_shared<ModelBoxSegmentationInfo>();
  empty_box->image_width = image->getWidth();
  empty_box->image_height = image->getHeight();
  empty_box->mask_width = 0;
  empty_box->mask_height = 0;

  int required_cached_frames = num_inputs - 1;
  if (cached_frame_count_ < required_cached_frames) {
    if (num_inputs == 3) {
      if (cached_frame_count_ == 0) {
        cached_input_0_ = current_preprocessed;
      } else {
        cached_input_1_ = current_preprocessed;
      }
    } else if (num_inputs == 2) {
      cached_input_0_ = current_preprocessed;
    }
    cached_frame_count_++;
    out_data = empty_box;
    return 0;
  }

  TensorInfo input_info_0 = net_->getTensorInfo(input_names[0]);
  if (input_info_0.tensor_size != static_cast<int>(cached_input_0_.size()) ||
      input_info_0.sys_mem == nullptr) {
    LOGE("Cached input 0 size mismatch");
    return -1;
  }
  std::memcpy(input_info_0.sys_mem, cached_input_0_.data(),
              cached_input_0_.size());

  if (num_inputs == 3) {
    TensorInfo input_info_1 = net_->getTensorInfo(input_names[1]);
    if (input_info_1.tensor_size != static_cast<int>(cached_input_1_.size()) ||
        input_info_1.sys_mem == nullptr) {
      LOGE("Cached input 1 size mismatch");
      return -1;
    }
    std::memcpy(input_info_1.sys_mem, cached_input_1_.data(),
                cached_input_1_.size());
  }

  model_timer_.TicToc("preprocess");
  net_->updateInputTensors();
  net_->forward();
  model_timer_.TicToc("tpu");
  net_->updateOutputTensors();

  int32_t ret = outputParse(image, out_data);
  if (ret != 0) {
    LOGE("outputParse failed");
    return ret;
  }

  if (num_inputs == 3) {
    cached_input_0_.swap(cached_input_1_);
    cached_input_1_ = std::move(current_preprocessed);
  } else if (num_inputs == 2) {
    cached_input_0_ = std::move(current_preprocessed);
  }
  model_timer_.TicToc("post");

  return 0;
}

int32_t TopformerSegMotion::outputParse(
    const std::shared_ptr<BaseImage>& image,
    std::shared_ptr<ModelOutputInfo>& out_data) {
  if (image == nullptr) {
    LOGE("Input image is null");
    return -1;
  }

  int num_inputs = net_->getInputNames().size();
  int current_input_idx = num_inputs - 1;
  std::string input_tensor_name = net_->getInputNames()[current_input_idx];
  const std::vector<std::string>& output_names = net_->getOutputNames();
  if (output_names.empty()) {
    LOGE("Output layer is empty");
    return -1;
  }

  const std::string& output_name = output_names[0];
  TensorInfo output_info = net_->getTensorInfo(output_name);
  if (output_info.shape.size() < 4) {
    LOGE("Invalid output shape");
    return -1;
  }

  int channels = output_info.shape[1];
  int out_h = output_info.shape[2];
  int out_w = output_info.shape[3];
  if (channels < 2 || out_h <= 0 || out_w <= 0) {
    LOGE("Unexpected output shape c:%d h:%d w:%d", channels, out_h, out_w);
    return -1;
  }

  std::shared_ptr<BaseTensor> output_tensor =
      net_->getOutputTensor(output_name);
  std::vector<uint8_t> class_mask;
  if (output_info.data_type == TDLDataType::FP32) {
    argmaxMask(output_tensor->getBatchPtr<float>(0), channels, out_h, out_w,
               1.0f, class_mask);
  } else if (output_info.data_type == TDLDataType::INT8) {
    argmaxMask(output_tensor->getBatchPtr<int8_t>(0), channels, out_h, out_w,
               output_info.qscale, class_mask);
  } else if (output_info.data_type == TDLDataType::UINT8) {
    argmaxMask(output_tensor->getBatchPtr<uint8_t>(0), channels, out_h, out_w,
               output_info.qscale, class_mask);
  } else {
    LOGE("Unsupported output data type:%d",
         static_cast<int>(output_info.data_type));
    return -1;
  }

  int fg_class_id = (num_inputs == 2) ? 1 : 2;
  std::vector<uint8_t> fg_mask(out_h * out_w, 0);
  for (int i = 0; i < out_h * out_w; ++i) {
    fg_mask[i] = (class_mask[i] == fg_class_id) ? 255 : 0;
  }

  int num_boxes = 0;
  int* p_boxes = nullptr;
#if defined(__CMODEL_CV181X__)
  p_boxes = nullptr;
  num_boxes = 0;
#elif defined(__CV181X__) || defined(__CV186X__) || defined(__CV184X__) || \
    defined(__CMODEL_CV184X__)
  p_boxes = extractConnectedComponent(fg_mask.data(), out_w, out_h, out_w,
                                      min_area_thresh_ / 64, ccl_instance_,
                                      &num_boxes);
#else
  p_boxes = extractConnectedComponent(fg_mask.data(), out_w, out_h, out_w,
                                      min_area_thresh_ / 64, ccl_instance_,
                                      &num_boxes);
#endif

  std::vector<float> scale_params = batch_rescale_params_[input_tensor_name][0];

  std::shared_ptr<ModelBoxSegmentationInfo> box_info =
      std::make_shared<ModelBoxSegmentationInfo>();
  box_info->image_width = image->getWidth();
  box_info->image_height = image->getHeight();
  box_info->mask_width = out_w;
  box_info->mask_height = out_h;
  box_info->box_seg.reserve(num_boxes);

  const std::vector<std::string>& input_names = net_->getInputNames();
  TensorInfo tensor_info = net_->getTensorInfo(input_names[current_input_idx]);

  float scale_x =
      static_cast<float>(tensor_info.shape[3]) / static_cast<float>(out_w);
  float scale_y =
      static_cast<float>(tensor_info.shape[2]) / static_cast<float>(out_h);
  for (int j = 0; j < num_boxes; ++j) {
    float x1 = static_cast<float>(p_boxes[j * 5 + 2]) * scale_x;
    float y1 = static_cast<float>(p_boxes[j * 5 + 1]) * scale_y;
    float x2 = static_cast<float>(p_boxes[j * 5 + 4]) * scale_x;
    float y2 = static_cast<float>(p_boxes[j * 5 + 3]) * scale_y;

    ObjectBoxSegmentationInfo bbox;
    bbox.class_id = fg_class_id;
    bbox.score = 1.0f;
    bbox.x1 = x1;
    bbox.y1 = y1;
    bbox.x2 = x2;
    bbox.y2 = y2;

    if (with_mask_) {
      size_t mask_size =
          static_cast<size_t>(out_h) * static_cast<size_t>(out_w);
      if (mask_size > 0) {
        bbox.mask = reinterpret_cast<uint8_t*>(malloc(mask_size));
        if (bbox.mask != nullptr) {
          std::memcpy(bbox.mask, class_mask.data(), mask_size);
        }
      }
    }

    DetectionHelper::rescaleBbox(bbox, scale_params);
    box_info->box_seg.emplace_back(std::move(bbox));
  }

  out_data = box_info;
  return 0;
}
