#include "model/base_model.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>

#include "preprocess/base_preprocessor.hpp"
#include "utils/common_utils.hpp"
#include "utils/tdl_log.hpp"
void print_netparam(const NetParam& net_param) {
  std::stringstream ss;
  ss << "mean:[";
  for (auto& mean : net_param.model_config.mean) {
    ss << mean << ",";
  }
  ss << "]\nstd:[";
  for (auto& std : net_param.model_config.std) {
    ss << std << ",";
  }
  ss << "]\nformat:" << net_param.model_config.rgb_order;
  LOGI("%s", ss.str().c_str());
}

BaseModel::BaseModel() {
  // bug:should not use memset for struct has stl container
  //  memset(&net_param_, 0, sizeof(NetParam));

  net_param_.platform = CommonUtils::getPlatform();
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

int32_t BaseModel::getPreprocessParameters(PreprocessParams& pre_param,
                                           const std::string& input_name) {
  if (input_name.empty()) {
    pre_param = preprocess_params_[net_->getInputNames()[0]];
  } else {
    if (preprocess_params_.find(input_name) == preprocess_params_.end()) {
      LOGE("input_name:%s not found", input_name.c_str());
      return -1;
    }
    pre_param = preprocess_params_[input_name];
  }

  return 0;
}

int32_t BaseModel::setPreprocessParameters(const PreprocessParams& pre_param,
                                           const std::string& input_name) {
  std::string input_name_ = input_name;
  if (input_name.empty()) {
    input_name_ = net_->getInputNames()[0];

  } else {
    if (preprocess_params_.find(input_name) == preprocess_params_.end()) {
      LOGE("input_name:%s not found", input_name.c_str());
      return -1;
    }
  }
  PreprocessParams preprocess_params = pre_param;
  TensorInfo tensor_info = net_->getTensorInfo(input_name_);
  for (int i = 0; i < 3; i++) {
    preprocess_params.mean[i] *= tensor_info.qscale;
    preprocess_params.scale[i] *= tensor_info.qscale;
  }
  preprocess_params.dst_height = tensor_info.shape[2];
  preprocess_params.dst_width = tensor_info.shape[3];
  preprocess_params.dst_pixdata_type = tensor_info.data_type;
  LOGI(
      "input_name:%s,qscale:%f,mean:%f,%f,%f,scale:%f,%f,%f,dst_height:%"
      "d,"
      "dst_width:%d,dst_pixdata_type:%d",
      input_name_.c_str(), tensor_info.qscale, preprocess_params.mean[0],
      preprocess_params.mean[1], preprocess_params.mean[2],
      preprocess_params.scale[0], preprocess_params.scale[1],
      preprocess_params.scale[2], preprocess_params.dst_height,
      preprocess_params.dst_width, (int)preprocess_params.dst_pixdata_type);

  preprocess_params_[input_name_] = preprocess_params;

  return 0;
}

int32_t BaseModel::setupNetwork(NetParam& net_param) {
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
    PreprocessParams preprocess_param;
    memset(&preprocess_param, 0, sizeof(preprocess_param));
    if (net_param_.model_config.mean.size() != 3 ||
        net_param_.model_config.std.size() != 3) {
      LOGE("mean or std size is not 3");
      assert(false);
      return -1;
    }
    preprocess_param.keep_aspect_ratio = keep_aspect_ratio_;
    if (net_param_.model_config.rgb_order == "rgb") {
      preprocess_param.dst_image_format = ImageFormat::RGB_PLANAR;
    } else if (net_param_.model_config.rgb_order == "bgr") {
      preprocess_param.dst_image_format = ImageFormat::BGR_PLANAR;
    } else if (net_param_.model_config.rgb_order == "gray") {
      preprocess_param.dst_image_format = ImageFormat::GRAY;
    } else {
      preprocess_param.dst_image_format = ImageFormat::UNKOWN;
      LOGW("img_format:%s,dst image format is unkown",
           net_param_.model_config.rgb_order.c_str());
    }
    for (int i = 0; i < 3; i++) {
      preprocess_param.mean[i] =
          net_param_.model_config.mean[i] / net_param_.model_config.std[i];
      preprocess_param.scale[i] = 1.0 / net_param_.model_config.std[i];
      preprocess_param.mean[i] *= tensor_info.qscale;
      preprocess_param.scale[i] *= tensor_info.qscale;
    }
    preprocess_param.dst_height = tensor_info.shape[2];
    preprocess_param.dst_width = tensor_info.shape[3];
    preprocess_param.dst_pixdata_type = tensor_info.data_type;
    LOGI(
        "input_name:%s,qscale:%f,mean:%f,%f,%f,scale:%f,%f,%f,dst_height:%"
        "d,"
        "dst_width:%d,dst_pixdata_type:%d",
        name.c_str(), tensor_info.qscale, preprocess_param.mean[0],
        preprocess_param.mean[1], preprocess_param.mean[2],
        preprocess_param.scale[0], preprocess_param.scale[1],
        preprocess_param.scale[2], preprocess_param.dst_height,
        preprocess_param.dst_width, (int)preprocess_param.dst_pixdata_type);

    preprocess_params_[name] = preprocess_param;
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
      "preprocess_params:%f,%f,%f,mean:%f,%f,%f,scale:%f,%f,%f,dst_height:%"
      "d,"
      "dst_width:%d,dst_pixdata_type:%d",
      preprocess_params.scale[0], preprocess_params.scale[1],
      preprocess_params.scale[2], preprocess_params.mean[0],
      preprocess_params.mean[1], preprocess_params.mean[2],
      preprocess_params.scale[0], preprocess_params.scale[1],
      preprocess_params.scale[2], preprocess_params.dst_height,
      preprocess_params.dst_width, preprocess_params.dst_pixdata_type);
  model_timer_.TicToc("runstart");
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
    model_timer_.TicToc("preprocess");
    net_->updateInputTensors();
    net_->forward();
    model_timer_.TicToc("tpu");
    net_->updateOutputTensors();
    std::vector<std::shared_ptr<ModelOutputInfo>> batch_results;
    outputParse(batch_images, batch_results);
    model_timer_.TicToc("post");
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

int32_t BaseModel::inference(const std::shared_ptr<BaseImage>& image,
                             std::shared_ptr<ModelOutputInfo>& out_data,
                             const std::map<std::string, float>& parameters) {
  std::vector<std::shared_ptr<BaseImage>> images = {image};
  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  int32_t ret = inference(images, out_datas, parameters);
  if (ret != 0) {
    LOGE("inference failed");
    return ret;
  }
  out_data = out_datas[0];
  return 0;
}

void BaseModel::setTypeMapping(
    const std::map<int, TDLObjectType>& type_mapping) {
  type_mapping_ = type_mapping;
}
void BaseModel::setModelThreshold(float threshold) {
  model_threshold_ = threshold;
}
