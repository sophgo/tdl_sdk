#include "model/base_model.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>

#include "cvi_tdl_log.hpp"

void print_netparam(const NetParam& net_param) {
  LOGI("scale:%f,%f,%f,mean:%f,%f,%f,format:%d,keepAspectRatio:%d",
       net_param.pre_params.scale[0], net_param.pre_params.scale[1],
       net_param.pre_params.scale[2], net_param.pre_params.mean[0],
       net_param.pre_params.mean[1], net_param.pre_params.mean[2],
       net_param.pre_params.dstImageFormat,
       net_param.pre_params.keepAspectRatio);
}

BaseModel::BaseModel() {
  memset(&net_param_, 0, sizeof(NetParam));
#ifdef __SOPHON__
  net_param_.platform = InferencePlatform::BM168X;

#else
  net_param_.platform = InferencePlatform::CVITEK;

#endif
}

int32_t BaseModel::setPreprocessor(
    std::shared_ptr<BasePreprocessor> preprocessor) {
  preprocessor_ = preprocessor;
  return 0;
}
int32_t BaseModel::modelOpen(const std::string& model_path) {
  net_param_.model_file_path = model_path;
  print_netparam(net_param_);
  int32_t ret = setupNetwork(net_param_);
  if (ret != 0) {
    LOGE("Net setup failed");
    return ret;
  }
  LOGI("modelOpen success");
  ret = onModelOpened();
  if (ret != 0) {
    LOGE("onModelOpened failed");
    return ret;
  } else {
    LOGI("onModelOpened success");
  }
  return 0;
}
void BaseModel::setInputBatchSize(const std::string& layer_name,
                                  int batch_size) {
  auto input_tensor = net_->getInputTensor(layer_name);
  if (input_tensor == nullptr) {
    LOGE("input_tensor is nullptr");
    return;
  }
  input_tensor->reshape(batch_size, input_tensor->getShape()[1],
                        input_tensor->getShape()[2],
                        input_tensor->getShape()[3]);
  input_batch_size_ = batch_size;
}

int BaseModel::getFitBatchSize(int left_size) const {
  auto& supported_batch_sizes = net_->getSupportedBatchSizes();
  for (const auto& batch_size : supported_batch_sizes) {
    if (batch_size <= left_size) {
      return batch_size;
    }
  }
  return 0;
}

int BaseModel::getDeviceId() const {
  if (net_) {
    return net_->getDeviceId();
  } else {
    LOGE("Net has not been setup");
    return 0;
  }
}

int32_t BaseModel::setupNetwork(NetParam& net_param) {
  std::cout << "setupNetwork" << std::endl;
  net_ = NetFactory::createNet(net_param, net_param.platform);
  int32_t ret = net_->setup();
  if (ret != 0) {
    std::cout << "Net setup failed" << std::endl;
    return ret;
  }

  const std::vector<std::string>& input_names = net_->getInputNames();
  for (auto& name : input_names) {
    TensorInfo tensor_info = net_->getTensorInfo(name);
    PreprocessParams preprocess_params = net_param_.pre_params;
    for (int i = 0; i < 3; i++) {
      preprocess_params.mean[i] *= tensor_info.qscale;
      preprocess_params.scale[i] *= tensor_info.qscale;
    }
    preprocess_params.dstHeight = tensor_info.shape[2];
    preprocess_params.dstWidth = tensor_info.shape[3];
    preprocess_params.dstPixDataType = tensor_info.data_type;
    // preprocess_params.dstImageFormat = ImageFormat::BGR_PLANAR;
    preprocess_params_[name] = preprocess_params;
  }
}

int32_t BaseModel::inference(
    const std::vector<std::shared_ptr<BaseImage>>& images,
    std::vector<void*>& out_datas, const int src_width, const int src_height) {
  if (images.empty()) {
    LOGE("Input images is empty");
    return -1;
  }
  if (preprocessor_ == nullptr) {
    LOGE("Preprocessor is not set");
    return -1;
  }

  int batch_size = images.size();
  int process_idx = 0;
  std::string input_layer_name = net_->getInputNames()[0];
  PreprocessParams preprocess_params = preprocess_params_[input_layer_name];
  while (process_idx < batch_size) {
    int fit_batch_size = getFitBatchSize(batch_size - process_idx);
    setInputBatchSize(input_layer_name, fit_batch_size);
    std::vector<std::shared_ptr<BaseImage>> batch_images;
    batch_rescale_params_.clear();

    for (int i = 0; i < fit_batch_size; i++) {
      batch_images.push_back(images[process_idx + i]);
      preprocessor_->preprocessToTensor(images[process_idx + i],
                                        preprocess_params, i,
                                        net_->getInputTensor(input_layer_name));
    }
    for (auto& img : batch_images) {
      std::vector<float> rescale_params = preprocessor_->getRescaleConfig(
          preprocess_params, img->getWidth(), img->getHeight());
      batch_rescale_params_.push_back(rescale_params);
    }
    net_->updateInputTensors();
    net_->forward();
    net_->updateOutputTensors();
    std::vector<void*> batch_results;
    outputParse(batch_images, batch_results);
    out_datas.insert(out_datas.end(), batch_results.begin(),
                     batch_results.end());
    process_idx += fit_batch_size;
  }
  return 0;
}
