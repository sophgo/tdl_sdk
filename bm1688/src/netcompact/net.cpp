#include "netcompact/net.hpp"
#include <iostream>
#include <log/Logger.hpp>

namespace nncompact {
Net::Net(const stNetParam& param) {net_param_ = param;}

void Net::add_input(const std::string& name) {
  if (input_tensor_hash_.find(name) != input_tensor_hash_.end()) {
    LOG(WARNING) << ("Layer " + name + " is already exist in net").c_str()
                 << std::endl;
    return;
  }
  input_tensor_hash_[name] = std::make_shared<Tensor>();
}

void Net::add_output(const std::string& name) {
  if (output_tensor_hash_.find(name) != output_tensor_hash_.end()) {
    LOG(WARNING) << ("Layer " + name + " is already exist in net").c_str()
                 << std::endl;
  }
  output_tensor_hash_[name] = std::make_shared<Tensor>();
  return;
}

std::shared_ptr<Tensor> Net::get_input_tensor(const std::string& name) {
  return input_tensor_hash_[name];
}

std::shared_ptr<Tensor> Net::get_output_tensor(const std::string& name) {
  return output_tensor_hash_[name];
}

void Net::update() {
  update_input_tensors();
  forward();
  update_output_tensors();
}

void Net::print_net_mode() {
  std::cout << "Base Net" << std::endl;
}
}
