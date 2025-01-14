#ifndef CVI_NET_H
#define CVI_NET_H

#include "net/base_net.hpp"
#include "net/base_tensor.hpp"

class CviNet : public BaseNet {
 public:
  CviNet(const NetParam& param);
  virtual ~CviNet();

  int32_t setup() override;
  int32_t forward(bool sync = true) override;
  int32_t addInput(const std::string& name) override;
  int32_t addOutput(const std::string& name) override;
  std::shared_ptr<BaseTensor> getInputTensor(const std::string& name) override;
  std::shared_ptr<BaseTensor> getOutputTensor(const std::string& name) override;
  int32_t updateInputTensors() override;
  int32_t updateOutputTensors() override;
  int32_t setInputTensorPhyAddr(const std::string& tensor_name,
                                uint64_t phy_addr) override;

 private:
  void setupTensorInfo(void* cvi_tensor, int32_t num_tensors,
                       std::map<std::string, TensorInfo>& tensor_info);
  void* model_handle_ = nullptr;
  void* input_tensors_ = nullptr;
  void* output_tensors_ = nullptr;
};

#endif  // CVI_NET_H
