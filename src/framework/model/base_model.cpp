#include "model/base_model.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>

#include "preprocess/base_preprocessor.hpp"
#include "utils/common_utils.hpp"
#include "utils/tdl_log.hpp"
void print_netparam(const NetParam& net_param) {
  LOGI("scale:%f,%f,%f,mean:%f,%f,%f,format:%d,aspect_ratio:%d",
       net_param.pre_params.scale[0], net_param.pre_params.scale[1],
       net_param.pre_params.scale[2], net_param.pre_params.mean[0],
       net_param.pre_params.mean[1], net_param.pre_params.mean[2],
       net_param.pre_params.dstImageFormat,
       net_param.pre_params.keepAspectRatio);
}

BaseModel::BaseModel() {
  memset(&net_param_, 0, sizeof(NetParam));
  net_param_.platform = get_platform();
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
  if (preprocessor_ == nullptr) {
    preprocessor_ =
        PreprocessorFactory::createPreprocessor(net_param_.platform);
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


int32_t BaseModel::getPreprocessParameters(PreprocessParams &pre_param){

  pre_param.dstImageFormat = net_param_.pre_params.dstImageFormat;
  pre_param.keepAspectRatio = net_param_.pre_params.keepAspectRatio;

  memcpy(pre_param.mean, net_param_.pre_params.mean, sizeof(pre_param.mean));
  memcpy(pre_param.scale, net_param_.pre_params.scale, sizeof(pre_param.scale));

  return 0;

}

int32_t BaseModel::setPreprocessParameters(PreprocessParams &pre_param){

  net_param_.pre_params.dstImageFormat = pre_param.dstImageFormat;
  net_param_.pre_params.keepAspectRatio = pre_param.keepAspectRatio;

  memcpy(net_param_.pre_params.mean, pre_param.mean, sizeof(pre_param.mean));
  memcpy(net_param_.pre_params.scale, pre_param.scale, sizeof(pre_param.scale));

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
    LOGI(
        "input_name:%s,qscale:%f,mean:%f,%f,%f,scale:%f,%f,%f,dstHeight:%d,"
        "dstWidth:%d,dstPixDataType:%d",
        name.c_str(), tensor_info.qscale, preprocess_params.mean[0],
        preprocess_params.mean[1], preprocess_params.mean[2],
        preprocess_params.scale[0], preprocess_params.scale[1],
        preprocess_params.scale[2], preprocess_params.dstHeight,
        preprocess_params.dstWidth, (int)preprocess_params.dstPixDataType);

    preprocess_params_[name] = preprocess_params;
  }
  return 0;

}


int32_t BaseModel::setupNetwork(NetParam& net_param) {
  std::cout << "setupNetwork" << std::endl;
  net_ = NetFactory::createNet(net_param, net_param.platform);
  int32_t ret = net_->setup();
  if (ret != 0) {
    std::cout << "Net setup failed" << std::endl;
    assert(false);
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
    LOGI(
        "input_name:%s,qscale:%f,mean:%f,%f,%f,scale:%f,%f,%f,dstHeight:%d,"
        "dstWidth:%d,dstPixDataType:%d",
        name.c_str(), tensor_info.qscale, preprocess_params.mean[0],
        preprocess_params.mean[1], preprocess_params.mean[2],
        preprocess_params.scale[0], preprocess_params.scale[1],
        preprocess_params.scale[2], preprocess_params.dstHeight,
        preprocess_params.dstWidth, (int)preprocess_params.dstPixDataType);

    preprocess_params_[name] = preprocess_params;
  }
  return 0;
}

int32_t BaseModel::inference(
    const std::vector<std::shared_ptr<BaseImage>>& images,
    std::vector<std::shared_ptr<ModelOutputInfo>>& out_datas,
    const std::map<std::string, float>& parameters) {
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
  const PreprocessParams& preprocess_params =
      preprocess_params_[input_layer_name];
  LOGI(
      "BaseModel::inference "
      "preprocess_params:%f,%f,%f,mean:%f,%f,%f,scale:%f,%f,%f,dstHeight:%d,"
      "dstWidth:%d,dstPixDataType:%d",
      preprocess_params.scale[0], preprocess_params.scale[1],
      preprocess_params.scale[2], preprocess_params.mean[0],
      preprocess_params.mean[1], preprocess_params.mean[2],
      preprocess_params.scale[0], preprocess_params.scale[1],
      preprocess_params.scale[2], preprocess_params.dstHeight,
      preprocess_params.dstWidth, preprocess_params.dstPixDataType);

  std::shared_ptr<BaseTensor> input_tensor =
      net_->getInputTensor(input_layer_name);
  while (process_idx < batch_size) {
    int fit_batch_size = getFitBatchSize(batch_size - process_idx);
    setInputBatchSize(input_layer_name, fit_batch_size);
    std::vector<std::shared_ptr<BaseImage>> batch_images;
    batch_rescale_params_.clear();

    for (int i = 0; i < fit_batch_size; i++) {
      batch_images.push_back(images[process_idx + i]);
      if (images[process_idx + i]->getImageType() == ImageType::TENSOR_FRAME) {
        if (images[process_idx + i]->getVirtualAddress()[0] !=
            input_tensor->getBatchPtr<uint8_t>(i)) {
          LOGE(
              "image memory address is not equal to input tensor memory "
              "address");
          assert(false);
        }
      } else {
        preprocessor_->preprocessToTensor(
            images[process_idx + i], preprocess_params, i,
            net_->getInputTensor(input_layer_name));
        std::vector<float> rescale_params = preprocessor_->getRescaleConfig(
            preprocess_params, images[process_idx + i]->getWidth(),
            images[process_idx + i]->getHeight());
        batch_rescale_params_.push_back(rescale_params);
      }
    }
    net_->updateInputTensors();
    net_->forward();
    net_->updateOutputTensors();
    std::vector<std::shared_ptr<ModelOutputInfo>> batch_results;
    outputParse(batch_images, batch_results);
    out_datas.insert(out_datas.end(), batch_results.begin(),
                     batch_results.end());
    process_idx += fit_batch_size;
  }
  return 0;
}

int32_t BaseModel::inference(
    const std::shared_ptr<BaseImage>& image,
    const std::shared_ptr<ModelOutputInfo>& model_object_infos,
    std::vector<std::shared_ptr<ModelOutputInfo>>& out_datas,
    const std::map<std::string, float>& parameters) {
  LOGW("inference not implemented");
  return -1;
}
void BaseModel::setTypeMapping(
    const std::map<int, TDLObjectType>& type_mapping) {
  type_mapping_ = type_mapping;
}
void BaseModel::setModelThreshold(float threshold) {
  model_threshold_ = threshold;
}
