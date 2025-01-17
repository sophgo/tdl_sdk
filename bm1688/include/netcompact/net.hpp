#ifndef INCLUDE_COMPACT_NET_H_
#define INCLUDE_COMPACT_NET_H_

#include <iostream>
#include <map>

#include "netcompact/tensor.hpp"

enum InputMemType{
  INPUT_MEM_DEVICE=0,
  INPUT_MEM_HOST
};
enum ResizeMode{
  IMG_PAD_RESIZE=0,
  IMG_STRETCH_RESIZE
};
typedef struct _stNetParam{
  std::vector<float> mean;
  std::vector<float> scale;
  bool use_rgb = false;//bgr default
  ResizeMode resize_mode =IMG_PAD_RESIZE;
  std::string model_file;
  InputMemType input_mem_type = INPUT_MEM_DEVICE;//model could load input from host memory or device memory
  int device_id;
  std::string net_name;//指定使用的网络名称，假如一个模型中有多个网络，必须指定该变量
  std::vector<std::string> input_names;//if leave empty, read input node from model file
  std::vector<std::string> output_names;//if leave empty, read output node from model file
  int pad_value=0;
}stNetParam;

namespace nncompact {
class Net {
public:
  explicit Net(const stNetParam &net_param);

  virtual ~Net() {}

  virtual void setup() {}

  virtual void add_input(const std::string &name);

  virtual void add_output(const std::string &name);

  virtual std::shared_ptr<Tensor> get_input_tensor(const std::string &name);

  virtual void update_input_tensors() {}

  virtual void forward(bool sync = true) {}

  virtual void update_output_tensors() {}

  virtual void update();

  virtual std::shared_ptr<Tensor> get_output_tensor(const std::string &name);

  virtual void print_net_mode();

  virtual const void *get_net_info() { return nullptr; }

  virtual void set_store_mode(int mode) { store_mode_ = mode; }

  virtual void *get_device_output_tensors() { return nullptr; }

  virtual void *get_handle() { return nullptr; }

  virtual void *get_bmrt() { return nullptr; }

  int get_device_id() { return device_id_; }
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::map<std::string,std::vector<int>> available_batches_;
  std::map<std::string,std::vector<int>> input_shapes_;
protected:
  stNetParam net_param_;
  std::map<std::string, std::shared_ptr<Tensor>> input_tensor_hash_;
  std::map<std::string, std::shared_ptr<Tensor>> output_tensor_hash_;
  // bm_data_type_t input_data_type_;
  int store_mode_ = 0;
  int device_id_ = 0;

private:
  Net(const Net &) = delete;

  Net &operator=(const Net &) = delete;
};
} // namespace nncompact

#endif
