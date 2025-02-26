#ifndef CVI_NET_H
#define CVI_NET_H

#include <cviruntime.h>

#include "net/base_net.hpp"
#include "tensor/base_tensor.hpp"
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

 private:
  void setupTensorInfo(CVI_TENSOR* cvi_tensor, int32_t num_tensors,
                       std::map<std::string, TensorInfo>& tensor_info);
  void* model_handle_ = nullptr;
  CVI_TENSOR* input_tensors_ = nullptr;
  CVI_TENSOR* output_tensors_ = nullptr;
  std::shared_ptr<BaseMemoryPool> memory_pool_;
};

#endif  // CVI_NET_H
