#include "net/base_net.hpp"

#include "utils/tdl_log.hpp"
BaseNet::BaseNet(const NetParam& net_param) { net_param_ = net_param; }

int32_t BaseNet::setup() { return 0; }

int32_t BaseNet::addInput(const std::string& name) { return 0; }

int32_t BaseNet::addOutput(const std::string& name) { return 0; }

// Retrieve input and output tensors
std::shared_ptr<BaseTensor> BaseNet::getInputTensor(const std::string& name) {
  if (name == "") {
    return input_tensor_hash_[input_tensor_names_[0]];
  }
  return input_tensor_hash_[name];
}
std::shared_ptr<BaseTensor> BaseNet::getOutputTensor(const std::string& name) {
  if (name == "") {
    return output_tensor_hash_[output_tensor_names_[0]];
  }
  return output_tensor_hash_[name];
}

TensorInfo BaseNet::getTensorInfo(const std::string& name) {
  TensorInfo tensor_info;
  if (input_output_tensor_infos_.find(name) !=
      input_output_tensor_infos_.end()) {
    tensor_info = input_output_tensor_infos_[name];
  } else {
    LOGE("Tensor info not found for %s", name.c_str());
    memset(&tensor_info, 0, sizeof(TensorInfo));
  }

  return tensor_info;
}

const std::vector<int>& BaseNet::getSupportedBatchSizes(
    const std::string& name) {
  if (name == "") {
    return supported_batch_sizes_.at(input_tensor_names_[0]);
  }
  return supported_batch_sizes_.at(name);
}

// Update input/output tensors
int32_t BaseNet::updateInputTensors() { return 0; }
int32_t BaseNet::updateOutputTensors() { return 0; }

// Perform forward propagation
int32_t BaseNet::forward(bool sync) { return 0; }

// Getters
int32_t BaseNet::getDeviceId() const { return device_id_; }
const std::vector<std::string>& BaseNet::getInputNames() const {
  return input_tensor_names_;
}
const std::vector<std::string>& BaseNet::getOutputNames() const {
  return output_tensor_names_;
}
