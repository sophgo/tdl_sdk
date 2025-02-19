#ifndef INCLUDE_BASE_NET_H_
#define INCLUDE_BASE_NET_H_

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common/common_types.hpp"
#include "tensor/base_tensor.hpp"

class BaseNet {
 public:
  explicit BaseNet(const NetParam& net_param);

  virtual ~BaseNet() {}

  // Setup the network
  virtual int32_t setup() = 0;

  // Add input and output nodes
  virtual int32_t addInput(const std::string& name);
  virtual int32_t addOutput(const std::string& name);

  // Retrieve input and output tensors
  virtual std::shared_ptr<BaseTensor> getInputTensor(const std::string& name);
  virtual std::shared_ptr<BaseTensor> getOutputTensor(const std::string& name);

  TensorInfo getTensorInfo(const std::string& name);
  const std::vector<int>& getSupportedBatchSizes(const std::string& name = "");

  // Update input/output tensors
  virtual int32_t updateInputTensors();
  virtual int32_t updateOutputTensors();

  virtual int32_t forward(bool sync = true) = 0;

  // Getters
  int32_t getDeviceId() const;
  const std::vector<std::string>& getInputNames() const;
  const std::vector<std::string>& getOutputNames() const;
  virtual int32_t setInputTensorPhyAddr(const std::string& tensor_name,
                                        uint64_t phy_addr);

 protected:
  NetParam net_param_;
  std::map<std::string, std::shared_ptr<BaseTensor>> input_tensor_hash_;
  std::map<std::string, std::shared_ptr<BaseTensor>> output_tensor_hash_;

  // TODO:not used in inference,fix me
  std::map<std::string, std::vector<int>> supported_batch_sizes_;

  std::map<std::string, TensorInfo> input_output_tensor_infos_;

  std::vector<std::string> input_tensor_names_;
  std::vector<std::string> output_tensor_names_;
  int device_id_ = 0;

 private:
  BaseNet(const BaseNet&) = delete;
  BaseNet& operator=(const BaseNet&) = delete;
};

class NetFactory {
 public:
  static std::shared_ptr<BaseNet> createNet(const NetParam& net_param,
                                            InferencePlatform platform);
};

#endif  // INCLUDE_BASE_NET_H_
