
#include "netcompact/net/bm1688net.hpp"

#include <bmruntime_interface.h>
#include <bmruntime_legacy.h>

#include <common/common.hpp>
#include <cstring>
#include <string>
#include <vector>

#include "netcompact/tensor.hpp"

namespace nncompact {

ModelInstance::~ModelInstance() {
  for (auto kv : model_bmrts_) {
    if (kv.second) {
      bmrt_destroy(kv.second);
      LOG(INFO) << "model " << kv.first << " destroyed" << std::endl;
    }
  }
  model_bmrts_.clear();
}

void *ModelInstance::get_model_bmrt(const std::string &model_path,
                                    int device_id) {
  static ModelInstance inst;
  std::string str_model_flag =
      model_path + std::string("_device_") + std::to_string(device_id);
  if (inst.model_bmrts_.count(str_model_flag) == 0) {
    bm_handle_t handle = BMContext::cnn_bm168x_handle(device_id);
    if (handle == nullptr) {
      LOG(FATAL) << "get handle failed,device_id:" << device_id;
    }
    void *p_bmrt = bmrt_create(handle);
    bool flag = bmrt_load_bmodel(p_bmrt, model_path.c_str());
    if (!flag) {
      LOG(FATAL) << model_path << "load failed";
    }
    inst.model_bmrts_[str_model_flag] = p_bmrt;
  }
  return inst.model_bmrts_[str_model_flag];
}

BM1688Net::~BM1688Net() {
  if (input_tensors_) {
    delete[] input_tensors_;
    input_tensors_ = 0;
  }
  if (output_tensors_) {
    delete[] output_tensors_;
    output_tensors_ = 0;
  }
  p_bmrt_ = 0;
  net_info_ = 0;
  LOG(INFO) << "destroy 168xnet:" << net_name_;
}

void BM1688Net::setup() {
  std::string bmcnn_ctx_dir = net_param_.model_file;

  if (p_bmrt_ != 0) {
    LOG(WARNING) << net_name_ << "has been setup , do not setup again";
    return;
  }
  int device_id = net_param_.device_id;
  device_id_ = device_id;
  int dir_len = bmcnn_ctx_dir.length();
  if (bmcnn_ctx_dir.at(dir_len - 1) == '/')
    bmcnn_ctx_dir = bmcnn_ctx_dir.substr(0, dir_len - 1);

  bm_handle_ = BMContext::cnn_bm168x_handle(device_id);
  if (bm_handle_ == nullptr) {
    LOG(FATAL) << "get handle failed,device_id:" << device_id;
  }

  p_bmrt_ = ModelInstance::get_model_bmrt(bmcnn_ctx_dir, device_id);
  // if name was not set,use the name inside bmodel as default
  std::string net_name = net_param_.net_name;
  LOG(INFO) << "net_name: " << net_name;
  if (net_name == "") {
    const char **net_names = NULL;
    int net_num = bmrt_get_network_number(p_bmrt_);
    bmrt_get_network_names(p_bmrt_, &net_names);
    if (net_num != 1) {
      std::stringstream ss;
      for (int i = 0; i < net_num; i++) ss << net_names[i] << ",";
      LOG(FATAL) << "no net_name has been config,found " << net_num
                 << ",names:" << ss.str();
    } else {
      net_name = net_names[0];
      LOG(INFO) << "net_name auto find " << net_name;
    }
    free(net_names);
  }
  LOG(INFO) << "start to setup " << net_name << " on device:" << device_id
            << " from:" << bmcnn_ctx_dir;
  net_info_ = bmrt_get_network_info(p_bmrt_, net_name.c_str());
  input_names_.clear();
  output_names_.clear();
  available_batches_.clear();
  for (int i = 0; i < net_info_->input_num; i++) {
    input_name_index_[net_info_->input_names[i]] = i;
    input_names_.push_back(net_info_->input_names[i]);
    LOG(INFO) << "input " << i << ",name:" << net_info_->input_names[i];
    auto &shape0 = net_info_->stages[0].input_shapes[i];
    if (shape0.num_dims != 4) {
      std::cout << "input :" << input_names_[i]
                << ",dim error,got:" << shape0.num_dims << "expect 4"
                << std::endl;
      LOG(FATAL) << "input :" << input_names_[i]
                 << ",dim error,got:" << shape0.num_dims << "expect 4";
    }
    for (int k = 0; k < shape0.num_dims; k++) {
      input_shapes_[input_names_[i]].push_back(shape0.dims[k]);
    }

    for (int j = 0; j < net_info_->stage_num; j++) {
      auto &bmrt_shape = net_info_->stages[j].input_shapes[i];
      available_batches_[input_names_[i]].push_back(bmrt_shape.dims[0]);
    }
  }
  // TODO(fuquan.ke) fix me, add specified outputs only
  for (int i = 0; i < net_info_->output_num; i++) {
    output_name_index_[net_info_->output_names[i]] = i;
    // add_output(net_info_->output_names[i]);
    output_names_.push_back(net_info_->output_names[i]);
    // LOG(INFO) << "output " << i << ",name:" << net_info_->output_names[i];
  }
  input_tensors_ = new bm_tensor_t[net_info_->input_num];
  output_tensors_ = new bm_tensor_t[net_info_->output_num];
  net_name_ = net_name;
  LOG(INFO) << net_name << " is setup";
}

void BM1688Net::add_input(const std::string &name) {
  if (input_tensor_hash_.find(name) != input_tensor_hash_.end()) {
    LOG(WARNING) << ("Layer " + name + " is already exist in net").c_str()
                 << std::endl;
    return;
  }

  if (input_name_index_.count(name) == 0) {
    LOG(FATAL) << "input layer:" << name << " dont existed in the model";
  }
  int input_blob_idx = input_name_index_[name];
  int data_size = bmrt_data_type_size(net_info_->input_dtypes[input_blob_idx]);

  if (net_param_.input_mem_type == INPUT_MEM_DEVICE) {
    input_tensor_hash_[name] =
        std::make_shared<Tensor>(data_size, 1, bm_handle_);
  } else {
    input_tensor_hash_[name] =
        std::make_shared<Tensor>(data_size, 0, bm_handle_);
    input_tensor_device_[name] =
        std::make_shared<Tensor>(data_size, 1, bm_handle_);
  }
  LOG(INFO) << "finish add input:" << name << ",data type size:" << data_size;
  //  std::cout<<"add input:"<<name<<",data_size:"<<data_size<<std::endl;
}

void BM1688Net::add_output(const std::string &name) {
  if (output_tensor_hash_.find(name) != output_tensor_hash_.end()) {
    LOG(WARNING) << ("Layer " + name + " is already exist in net").c_str()
                 << std::endl;
    return;
  }
  LOG(INFO) << "start to add output :" << name;

  if (output_name_index_.count(name) == 0) {
    LOG(FATAL) << "output name " << name << " not existed in the model";
  }
  int output_blob_idx = output_name_index_[name];
  int data_size =
      bmrt_data_type_size(net_info_->output_dtypes[output_blob_idx]);
#ifdef USE_ARM
  // for soc mode,use mmap to get device memory data
  output_tensor_hash_[name] =
      std::make_shared<Tensor>(data_size, 1, bm_handle_);
#else
  output_tensor_hash_[name] =
      std::make_shared<Tensor>(data_size, 0, bm_handle_);
  output_tensor_device_[name] =
      std::make_shared<Tensor>(data_size, 1, bm_handle_);
#endif
}

const void *BM1688Net::get_net_info() {
  if (net_info_ == 0) {
    std::cout << "error,net_info_ is not setup" << std::endl;
    LOG(FATAL) << net_param_.net_name << " net_info_ is not setup";
  }
  return net_info_;
}

void BM1688Net::update_input_tensors() {
  int batch_n = 1;
  for (auto it = input_tensor_hash_.begin(); it != input_tensor_hash_.end();
       it++) {
    std::string name = it->first;
    Tensor *input_tensor = it->second.get();
    std::vector<int> input_shape = input_tensor->get_shape();
    batch_n = input_shape[0];
    int tensor_idx = input_name_index_[name];
    int stage_index = -1;
    for (int i = 0; i < available_batches_[name].size(); i++) {
      if (input_shape[0] == available_batches_[name][i]) {
        stage_index = i;
        break;
      }
    }
    if (stage_index == -1) {
      LOG(INFO) << input_shape[0];
      LOG(FATAL) << net_param_.net_name
                 << "batch not supported,batch:" << input_shape[0];
    }
    LOG(INFO) << "to get stage:" << stage_index
              << ",stagenum:" << net_info_->stage_num;
    auto &bmrt_shape = net_info_->stages[stage_index].input_shapes[tensor_idx];
    for (int i = 0; i < 4; i++) {
      LOG(INFO) << i << ",tensor:" << input_shape[i]
                << ",bmrt:" << bmrt_shape.dims[i];
      if (input_shape[i] != bmrt_shape.dims[i]) LOG(FATAL) << "shape not equal";
    }

    bm_device_mem_t input_dev;
    if (net_param_.input_mem_type != INPUT_MEM_DEVICE) {
      input_tensor_device_[name]->reshape(input_shape[0], input_shape[1],
                                          input_shape[2], input_shape[3]);
      input_tensor_device_[name]->from_host_mem(
          (const char *)input_tensor->get_data());
      input_dev =
          *(bm_device_mem_t *)input_tensor_device_[name]->get_device_data();
    } else {
      input_dev = *(bm_device_mem_t *)input_tensor->get_device_data();
    }
    // input_dev.size = input_tensor->get_size();
    bmrt_shape.dims[0] = input_shape[0];
    input_tensors_[tensor_idx].dtype = net_info_->input_dtypes[tensor_idx];
    input_tensors_[tensor_idx].shape = bmrt_shape;  //  num_dims = 4;
    // memcpy(input_tensors_[tensor_idx].shape.dims,&input_shape[0],4*sizeof(int));
    input_tensors_[tensor_idx].device_mem = input_dev;
    input_tensors_[tensor_idx].st_mode = BM_STORE_1N;
  }

  // for bmrt version, the output buffer should be prepared at the same time
  for (auto it = output_tensor_hash_.begin(); it != output_tensor_hash_.end();
       it++) {
    std::string name = it->first;
    std::shared_ptr<Tensor> output_tensor = it->second;
    int tensor_idx = output_name_index_[name];
    auto out_shape = net_info_->stages[0].output_shapes[tensor_idx];

    std::vector<int> st(4, 1);
    int insert_idx = 4 - out_shape.num_dims;
    for (int i = insert_idx; i < 4; i++) {
      st[i] = out_shape.dims[i - insert_idx];
    }

    output_tensor->reshape(batch_n, st[1], st[2], st[3]);
    out_shape.dims[0] = batch_n;

    bm_device_mem_t output_dev_mem;
#ifdef USE_ARM
    output_dev_mem = *(bm_device_mem_t *)output_tensor->get_device_data();
#else
    output_tensor_device_[name]->reshape(batch_n, st[1], st[2], st[3]);
    output_dev_mem =
        *(bm_device_mem_t *)output_tensor_device_[name]->get_device_data();
#endif
    // output_dev_mem.size = output_tensor->get_size();
    output_tensors_[tensor_idx].dtype = net_info_->output_dtypes[tensor_idx];
    output_tensors_[tensor_idx].shape = out_shape;
    output_tensors_[tensor_idx].device_mem = output_dev_mem;
    output_tensors_[tensor_idx].st_mode = (bm_store_mode_t)store_mode_;
    LOG(INFO) << "add output:" << name << ",shape:" << batch_n << "," << st[1]
              << "," << st[2] << "," << st[3]
              << ",dev_addr:" << bm_mem_get_device_addr(output_dev_mem);
  }
}

void BM1688Net::update_output_tensors() {
  for (auto it = output_tensor_hash_.begin(); it != output_tensor_hash_.end();
       it++) {
    std::string name = it->first;
    std::shared_ptr<Tensor> output_tensor = it->second;
    std::vector<int> shapes = output_tensor->get_shape();

    int index = output_name_index_[name];
    // auto &out_shape = net_info_->stages[0].output_shapes[index];
    // for(int k = 0; k< out_shape.num_dims;k++){
    //  LOG(INFO)<<"update
    //  output:"<<name<<",dim"<<k<<",shape:"<<out_shape.dims[k];
    //}
    LOG(INFO) << "update output:" << name << ",shape:" << shapes[0] << ","
              << shapes[1] << "," << shapes[2] << "," << shapes[3];

#ifdef USE_ARM
    output_tensor->invalidate_device_mem();  // for soc mode
#else
    output_tensor->from_dev_mem(
        (const bm_device_mem_t *)output_tensor_device_[name]
            ->get_device_data());
#endif
  }
}

void BM1688Net::forward(bool syn) {
  LOG(INFO) << "input num:" << net_info_->input_num
            << ",output num:" << net_info_->output_num;
  for (int i = 0; i < net_info_->input_num; i++) {
    if (bm_mem_get_type(input_tensors_[i].device_mem) != BM_MEM_TYPE_DEVICE) {
      LOG(FATAL) << "input tensor is not device memory," << net_info_->name
                 << ",input:" << i;
    }
  }
  LOG(INFO) << "start to check output tensor";
  for (int i = 0; i < net_info_->output_num; i++) {
    if (bm_mem_get_type(output_tensors_[i].device_mem) != BM_MEM_TYPE_DEVICE) {
      LOG(FATAL) << "output tensor is not device memory," << net_info_->name
                 << ",output:" << i;
    }
  }
  LOG(INFO) << "start to do inference";
  bool ret = bmrt_launch_tensor_ex(p_bmrt_, net_info_->name, input_tensors_,
                                   net_info_->input_num, output_tensors_,
                                   net_info_->output_num, true, false);
  if (!ret) {
    LOG(FATAL) << "failed to do inference" << net_info_->name;
  }
  // sync, wait for finishing inference
  bmrt_thread_sync(p_bmrt_);

  LOG(INFO) << "forward success";
}

void *BM1688Net::get_device_output_tensors() { return output_tensors_; }

}  // namespace nncompact

// #endif
