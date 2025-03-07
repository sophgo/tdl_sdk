#ifndef _WRAPPER_TYPE_DEF_HPP_
#define _WRAPPER_TYPE_DEF_HPP_

#include "model/base_model.hpp"

class TDLContext {
 public:
  TDLContext();
  ~TDLContext();
  void init();
  void destroy();
  void add_model(const std::string& model_name, const std::string& model_path);
  void remove_model(const std::string& model_name);
}

#endif
