#ifndef INCLUDE_COMPACT_BM1688Net_CMODEL_H_
#define INCLUDE_COMPACT_BM1688Net_CMODEL_H_

// #ifdef USE_BM1684

// #include <ufw/net.hpp>
#include <memory>
#include <netcompact/net.hpp>

#include "bmcnnctx.h"
#include "bmnet.h"
#include "bmruntime_interface.h"

namespace nncompact {

class ModelInstance {
 public:
  ModelInstance() {}

  ~ModelInstance();

  static void *get_model_bmrt(const std::string &model_path, int device_id);
  std::map<std::string, void *> model_bmrts_;
};

class BM1688Net : public Net {
 public:
  explicit BM1688Net(const stNetParam &net_param) : Net(net_param) {}

  virtual ~BM1688Net();  // {if (bmcnn_ctx_ != NULL)
                         // bmcnn::bmcnn_ctx_destroy(bmcnn_ctx_);}//TODO:should
                         // use as singleton
  virtual void setup();

  virtual void add_input(const std::string &name);

  virtual void add_output(const std::string &name);

  virtual void update_input_tensors();

  virtual void forward(bool sync = true);

  virtual void update_output_tensors();

  virtual const void *get_net_info();

  std::vector<int> get_supported_batches();

  virtual void *get_handle() { return (void *)bm_handle_; }

  virtual void *get_bmrt() { return p_bmrt_; }

  void *get_device_output_tensors();

 private:
  bm_handle_t bm_handle_;
  std::map<std::string, std::shared_ptr<Tensor>> input_tensor_device_;
  void *p_bmrt_ = 0;
  std::string net_name_;

  const bm_net_info_t *net_info_;
  std::map<std::string, int> input_name_index_;
  std::map<std::string, int> output_name_index_;
  bm_tensor_t *input_tensors_ = 0;
  bm_tensor_t *output_tensors_ = 0;
  std::map<std::string, std::shared_ptr<Tensor>> output_tensor_device_;

  BM1688Net(const BM1688Net &) = delete;

  BM1688Net &operator=(const BM1688Net &) = delete;
};
}  // namespace nncompact
// #endif

#endif
