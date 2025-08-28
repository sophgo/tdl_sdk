#ifndef TDL_SDK_NET_BM168X_NET_HPP
#define TDL_SDK_NET_BM168X_NET_HPP

#include <bmruntime_interface.h>

#include "net/base_net.hpp"

class BM168xNet : public BaseNet {
 public:
  BM168xNet(const NetParam& net_param);
  virtual ~BM168xNet();

  int32_t setup() override;
  int32_t forward(bool sync = true) override;
  int32_t addInput(const std::string& name) override;
  int32_t addOutput(const std::string& name) override;
  int32_t updateInputTensors() override;
  int32_t updateOutputTensors() override;

 private:
  TensorInfo extractTensorInfo(bool is_input, int idx);
  void updateTensorInfo(const std::string& name,
                        const std::shared_ptr<BaseTensor>& tensor);
  bm_handle_t bm_handle_;
  int store_mode_ = 0;
  void* p_bmrt_ = 0;
  std::string net_name_;

  const bm_net_info_t* net_info_;
  std::map<std::string, int> input_name_index_;
  std::map<std::string, int> output_name_index_;
  bm_tensor_t* input_tensors_ = 0;
  bm_tensor_t* output_tensors_ = 0;
  std::shared_ptr<BaseMemoryPool> memory_pool_;
};
#endif
