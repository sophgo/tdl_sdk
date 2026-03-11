#include "net/base_net.hpp"

#include "image/base_image.hpp"
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
    tensor_info = TensorInfo{};
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

int32_t BaseNet::setInputTensorFromImage(
    const std::string& name, const std::shared_ptr<BaseImage>& image) {
  std::string tensor_name = name;
  if (tensor_name.empty() && !input_tensor_names_.empty()) {
    tensor_name = input_tensor_names_[0];
  }

  auto it = input_tensor_hash_.find(tensor_name);
  if (it == input_tensor_hash_.end()) {
    LOGE("Input tensor %s not found", tensor_name.c_str());
    return -1;
  }

  std::shared_ptr<BaseTensor> tensor = it->second;
  if (tensor == nullptr) {
    LOGE("Input tensor %s is nullptr", tensor_name.c_str());
    return -1;
  }

  // Get image memory info
  std::vector<uint8_t*> image_ptrs = image->getVirtualAddress();
  std::vector<uint64_t> image_phy_addrs = image->getPhysicalAddress();

  if (image_ptrs.empty() || image_phy_addrs.empty()) {
    LOGE("Image memory is not allocated");
    return -1;
  }

  // Use image's memory for tensor
  void* host_memory = image_ptrs[0];
  uint64_t device_address = image_phy_addrs[0];
  std::vector<int> shape = tensor->getShape();

  int32_t ret = tensor->shareMemory(host_memory, device_address, shape);
  if (ret != 0) {
    LOGE("Failed to share memory for input tensor %s", tensor_name.c_str());
    return -1;
  }

  // Update tensor info
  if (input_output_tensor_infos_.find(tensor_name) !=
      input_output_tensor_infos_.end()) {
    TensorInfo& tensor_info = input_output_tensor_infos_[tensor_name];
    tensor_info.sys_mem = static_cast<uint8_t*>(host_memory);
    tensor_info.phy_addr = device_address;
  }

  LOGI(
      "setInputTensorFromImage success, tensor:%s, host_memory:%p, "
      "device_address:0x%llx",
      tensor_name.c_str(), host_memory, (unsigned long long)device_address);

  return 0;
}
