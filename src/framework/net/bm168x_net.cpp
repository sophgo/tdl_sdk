#include "net/bm168x_net.hpp"

#include <bmruntime_interface.h>
#include <bmruntime_legacy.h>

#include <cassert>
#include <sstream>

#include "memory/bm_memory_pool.hpp"
#include "utils/common_utils.hpp"
#include "utils/tdl_log.hpp"
ModelInstance::~ModelInstance() {
  for (auto kv : model_bmrts_) {
    if (kv.second) {
      bmrt_destroy(kv.second);
      LOGI("model %s destroyed", kv.first.c_str());
    }
  }
  model_bmrts_.clear();
}

void *ModelInstance::get_model_bmrt(const std::string &model_path,
                                    int device_id) {
  static ModelInstance inst;
  std::string str_model_flag =
      model_path + std::string("_device_") + std::to_string(device_id);
  LOGI("to get_model_bmrt: %s", str_model_flag.c_str());
  if (inst.model_bmrts_.count(str_model_flag) == 0) {
    bm_handle_t handle = BMContext::cnn_bm168x_handle(device_id);
    if (handle == nullptr) {
      LOGE("get handle failed,device_id:%d", device_id);
    }
    void *p_bmrt = bmrt_create(handle);
    bool flag = bmrt_load_bmodel(p_bmrt, model_path.c_str());
    if (!flag) {
      LOGE("model %s load failed", model_path.c_str());
    }
    inst.model_bmrts_[str_model_flag] = p_bmrt;
  }
  LOGI("get_model_bmrt: %s,%0x", str_model_flag.c_str(),
       inst.model_bmrts_[str_model_flag]);
  return inst.model_bmrts_[str_model_flag];
}

BM168xNet::BM168xNet(const NetParam &net_param) : BaseNet(net_param) {
  // p_bmrt_ = ModelInstance::get_model_bmrt(net_param.model_file_path,
  //                                         net_param.device_id);
}

BM168xNet::~BM168xNet() {
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
  LOGI("destroy 168xnet:%s", net_name_.c_str());
}

int32_t BM168xNet::setup() {
  if (p_bmrt_ != 0) {
    LOGI("%s has been setup , do not setup again", net_name_.c_str());
    return -1;
  }
  int device_id = net_param_.device_id;
  device_id_ = device_id;

  bm_handle_ = BMContext::cnn_bm168x_handle(device_id);
  if (bm_handle_ == nullptr) {
    LOGE("get handle failed,device_id:%d", device_id);
  }
  memory_pool_ = std::make_shared<BmMemoryPool>(bm_handle_);

  LOGI("toxxx get_model_bmrt: %s", net_param_.model_file_path.c_str());
  p_bmrt_ =
      ModelInstance::get_model_bmrt(net_param_.model_file_path, device_id);
  LOGI("getxxx_model_bmrt: %s,%0x", net_param_.model_file_path.c_str(),
       p_bmrt_);
  // if name was not set,use the name inside bmodel as default
  std::string net_name = net_param_.net_name;
  LOGI("net_name: %s", net_name.c_str());
  if (net_name == "") {
    const char **net_names = NULL;
    LOGI("to get net_num");
    int net_num = bmrt_get_network_number(p_bmrt_);
    LOGI("net_num: %d", net_num);
    bmrt_get_network_names(p_bmrt_, &net_names);
    if (net_num != 1) {
      std::stringstream ss;
      for (int i = 0; i < net_num; i++) ss << net_names[i] << ",";
      LOGE("no net_name has been config,found %d,names:%s", net_num,
           ss.str().c_str());
      return -1;
    } else {
      net_name = net_names[0];
      LOGI("net_name auto find %s", net_name.c_str());
    }
    free(net_names);
  }
  LOGI("start to setup %s on device:%d from:%s", net_name.c_str(), device_id,
       net_param_.model_file_path.c_str());
  net_info_ = bmrt_get_network_info(p_bmrt_, net_name.c_str());

  input_tensor_names_.clear();
  output_tensor_names_.clear();
  supported_batch_sizes_.clear();
  input_output_tensor_infos_.clear();
  for (int i = 0; i < net_info_->input_num; i++) {
    input_name_index_[net_info_->input_names[i]] = i;
    input_tensor_names_.push_back(net_info_->input_names[i]);
    LOGI("input %d,name:%s", i, net_info_->input_names[i]);
    auto &shape0 = net_info_->stages[0].input_shapes[i];
    if (shape0.num_dims != 4) {
      LOGE("input %s,dim error,got:%d expect 4", input_tensor_names_[i].c_str(),
           shape0.num_dims);
      return -1;
    }

    input_output_tensor_infos_[input_tensor_names_[i]] =
        extractTensorInfo(true, i);

    for (int j = 0; j < net_info_->stage_num; j++) {
      auto &bmrt_shape = net_info_->stages[j].input_shapes[i];
      supported_batch_sizes_[input_tensor_names_[i]].push_back(
          bmrt_shape.dims[0]);
    }
  }
  // TODO(fuquan.ke) fix me, add specified outputs only
  for (int i = 0; i < net_info_->output_num; i++) {
    output_name_index_[net_info_->output_names[i]] = i;
    // add_output(net_info_->output_names[i]);
    output_tensor_names_.push_back(net_info_->output_names[i]);
    input_output_tensor_infos_[output_tensor_names_[i]] =
        extractTensorInfo(false, i);
    // LOG(INFO) << "output " << i << ",name:" << net_info_->output_names[i];
  }

  for (auto &name : input_tensor_names_) {
    addInput(name);
  }
  for (auto &name : output_tensor_names_) {
    addOutput(name);
  }

  input_tensors_ = new bm_tensor_t[net_info_->input_num];
  output_tensors_ = new bm_tensor_t[net_info_->output_num];
  net_name_ = net_name;
  LOGI("%s is setup", net_name.c_str());
  return 0;
}
TensorInfo BM168xNet::extractTensorInfo(bool is_input, int idx) {
  bm_data_type_t *p_data_type = net_info_->input_dtypes;
  if (!is_input) {
    p_data_type = net_info_->output_dtypes;
  }
  bm_shape_t *p_shape = net_info_->stages[0].input_shapes;
  if (!is_input) {
    p_shape = net_info_->stages[0].output_shapes;
  }

  char const **names = net_info_->input_names;
  if (!is_input) {
    names = net_info_->output_names;
  }
  float *p_qscale = net_info_->input_scales;
  if (!is_input) {
    p_qscale = net_info_->output_scales;
  }
  int *p_zero_point = net_info_->input_zero_point;
  if (!is_input) {
    p_zero_point = net_info_->output_zero_point;
  }
  TensorInfo tensor_info;

  tensor_info.shape.resize(4, 1);
  int insert_idx = 4 - p_shape[idx].num_dims;
  for (int i = insert_idx; i < 4; i++) {
    tensor_info.shape[i] = p_shape[idx].dims[i - insert_idx];
  }

  if (p_shape[idx].num_dims != 4) {
    LOGW("tensor shape size not equal 4,size:%d", p_shape[idx].num_dims);
  }
  tensor_info.qscale = p_qscale[idx];
  tensor_info.zero_point = p_zero_point[idx];
  if (p_data_type[idx] == BM_FLOAT32) {
    tensor_info.data_type = TDLDataType::FP32;
  } else if (p_data_type[idx] == BM_FLOAT16) {
    tensor_info.data_type = TDLDataType::FP16;
  } else if (p_data_type[idx] == BM_BFLOAT16) {
    tensor_info.data_type = TDLDataType::BF16;
  } else if (p_data_type[idx] == BM_INT16) {
    tensor_info.data_type = TDLDataType::INT16;
  } else if (p_data_type[idx] == BM_UINT16) {
    tensor_info.data_type = TDLDataType::UINT16;
  } else if (p_data_type[idx] == BM_INT8) {
    tensor_info.data_type = TDLDataType::INT8;
  } else if (p_data_type[idx] == BM_UINT8) {
    tensor_info.data_type = TDLDataType::UINT8;
  } else {
    LOGE("unsupported data type:%d", p_data_type[idx]);
    assert(0);
  }
  uint32_t data_type_size = get_data_type_size(tensor_info.data_type);
  if (data_type_size != bmrt_data_type_size(p_data_type[idx])) {
    LOGE("data type size not equal,expect:%d,got:%d", data_type_size,
         bmrt_data_type_size(p_data_type[idx]));
  }
  tensor_info.tensor_elem = tensor_info.shape[0] * tensor_info.shape[1] *
                            tensor_info.shape[2] * tensor_info.shape[3];
  tensor_info.tensor_size =
      get_data_type_size(tensor_info.data_type) * tensor_info.tensor_elem;
  return tensor_info;
}
int32_t BM168xNet::addInput(const std::string &name) {
  if (input_tensor_hash_.find(name) != input_tensor_hash_.end()) {
    LOGI("Layer %s is already exist in net", name.c_str());
    return 0;
  }

  if (input_name_index_.count(name) == 0) {
    LOGE("input layer:%s dont existed in the model", name.c_str());
  }
  int input_blob_idx = input_name_index_[name];
  int element_bytes =
      bmrt_data_type_size(net_info_->input_dtypes[input_blob_idx]);
  input_tensor_hash_[name] =
      std::make_shared<BaseTensor>(element_bytes, memory_pool_);
  auto &shape = input_output_tensor_infos_[name].shape;
  input_tensor_hash_[name]->reshape(1, shape[1], shape[2], shape[3]);
  LOGI(
      "finish add "
      "input:%s,element_bytes:%d,shape:%d,%d,%d,%d,qscale:%f,zero_point:%d,"
      "dtype:%d",
      name.c_str(), element_bytes, shape[0], shape[1], shape[2], shape[3],
      input_output_tensor_infos_[name].qscale,
      input_output_tensor_infos_[name].zero_point,
      static_cast<int>(input_output_tensor_infos_[name].data_type));
  return 0;
}

int32_t BM168xNet::addOutput(const std::string &name) {
  if (output_tensor_hash_.find(name) != output_tensor_hash_.end()) {
    LOGI("Layer %s is already exist in net", name.c_str());
    return 0;
  }
  LOGI("start to add output :%s", name.c_str());

  if (output_name_index_.count(name) == 0) {
    LOGE("output name %s not existed in the model", name.c_str());
  }
  int output_blob_idx = output_name_index_[name];
  int element_bytes =
      bmrt_data_type_size(net_info_->output_dtypes[output_blob_idx]);
  output_tensor_hash_[name] =
      std::make_shared<BaseTensor>(element_bytes, memory_pool_);
  auto &shape = input_output_tensor_infos_[name].shape;
  output_tensor_hash_[name]->reshape(1, shape[1], shape[2], shape[3]);
  LOGI(
      "finish add output:%s,element_bytes:%d,shape:%d,%d,%d,%d,qscale:%f,"
      "zero_point:%d,dtype:%d",
      name.c_str(), element_bytes, shape[0], shape[1], shape[2], shape[3],
      input_output_tensor_infos_[name].qscale,
      input_output_tensor_infos_[name].zero_point,
      static_cast<int>(input_output_tensor_infos_[name].data_type));
  return 0;
}

int32_t BM168xNet::updateInputTensors() {
  int batch_n = 1;
  for (auto it = input_tensor_hash_.begin(); it != input_tensor_hash_.end();
       it++) {
    std::string name = it->first;
    std::shared_ptr<BaseTensor> input_tensor = it->second;
    input_tensor->flushCache();
    std::vector<int> input_shape = input_tensor->getShape();
    batch_n = input_shape[0];
    int tensor_idx = input_name_index_[name];
    int stage_index = -1;
    for (int i = 0; i < supported_batch_sizes_[name].size(); i++) {
      if (input_shape[0] == supported_batch_sizes_[name][i]) {
        stage_index = i;
        break;
      }
    }
    if (stage_index == -1) {
      LOGE("batch not supported,batch:%d", input_shape[0]);
      assert(0);
    }
    LOGI("to get stage:%d,stagenum:%d", stage_index, net_info_->stage_num);
    auto &bmrt_shape = net_info_->stages[stage_index].input_shapes[tensor_idx];
    if (bmrt_shape.num_dims != 4) {
      LOGW("input tensor shape size not equal 4,size:%d", bmrt_shape.num_dims);
      assert(0);
    }
    for (int i = 0; i < 4; i++) {
      LOGI("%d,tensor:%d,bmrt:%d", i, input_shape[i], bmrt_shape.dims[i]);
      if (input_shape[i] != bmrt_shape.dims[i]) LOGE("shape not equal");
    }

    MemoryBlock *memory_block = input_tensor->getMemoryBlock();
    bm_device_mem_t input_dev = *(bm_device_mem_t *)memory_block->handle;

    bmrt_shape.dims[0] = input_shape[0];
    input_tensors_[tensor_idx].dtype = net_info_->input_dtypes[tensor_idx];
    input_tensors_[tensor_idx].shape = bmrt_shape;  //  num_dims = 4;
    // memcpy(input_tensors_[tensor_idx].shape.dims,&input_shape[0],4*sizeof(int));
    input_tensors_[tensor_idx].device_mem = input_dev;
    input_tensors_[tensor_idx].st_mode = BM_STORE_1N;
    LOGI("add input:%s,shape:%d,%d,%d,%d,dev_addr:%d,dtype:%d", name.c_str(),
         batch_n, input_shape[1], input_shape[2], input_shape[3],
         bm_mem_get_device_addr(input_dev), input_tensors_[tensor_idx].dtype);
    updateTensorInfo(name, input_tensor);
  }

  // for bmrt version, the output buffer should be prepared at the same time
  for (auto it = output_tensor_hash_.begin(); it != output_tensor_hash_.end();
       it++) {
    std::string name = it->first;
    std::shared_ptr<BaseTensor> output_tensor = it->second;
    int tensor_idx = output_name_index_[name];
    auto out_shape = net_info_->stages[0].output_shapes[tensor_idx];

    std::vector<int> st(4, 1);
    int insert_idx = 4 - out_shape.num_dims;
    for (int i = insert_idx; i < 4; i++) {
      st[i] = out_shape.dims[i - insert_idx];
    }

    BaseTensor *tensor = dynamic_cast<BaseTensor *>(output_tensor.get());
    MemoryBlock *memory_block = tensor->getMemoryBlock();
    bm_device_mem_t output_dev_mem = *(bm_device_mem_t *)memory_block->handle;
    if (batch_n == 1 && out_shape.dims[0] > 1) {
      LOGW("special case,batch_n:%d,out_shape:%d", batch_n, out_shape.dims[0]);
      output_tensor->reshape(out_shape.dims[0], st[1], st[2], st[3]);

    } else {
      out_shape.dims[0] = batch_n;
      output_tensor->reshape(batch_n, st[1], st[2], st[3]);
    }

    // output_dev_mem.size = output_tensor->get_size();
    output_tensors_[tensor_idx].dtype = net_info_->output_dtypes[tensor_idx];
    output_tensors_[tensor_idx].shape = out_shape;
    output_tensors_[tensor_idx].device_mem = output_dev_mem;
    output_tensors_[tensor_idx].st_mode = (bm_store_mode_t)store_mode_;
    LOGI("add output:%s,shape:%d,%d,%d,%d,dev_addr:%d,dtype:%d", name.c_str(),
         batch_n, st[1], st[2], st[3], bm_mem_get_device_addr(output_dev_mem),
         output_tensors_[tensor_idx].dtype);
    updateTensorInfo(name, output_tensor);
  }
  return 0;
}
void BM168xNet::updateTensorInfo(const std::string &name,
                                 const std::shared_ptr<BaseTensor> &tensor) {
  input_output_tensor_infos_[name].sys_mem =
      (uint8_t *)tensor->getMemoryBlock()->virtualAddress;
  input_output_tensor_infos_[name].phy_addr =
      tensor->getMemoryBlock()->physicalAddress;
  input_output_tensor_infos_[name].shape = tensor->getShape();
  input_output_tensor_infos_[name].tensor_elem =
      tensor->getShape()[0] * tensor->getShape()[1] * tensor->getShape()[2] *
      tensor->getShape()[3];
  input_output_tensor_infos_[name].tensor_size =
      input_output_tensor_infos_[name].tensor_elem *
      get_data_type_size(input_output_tensor_infos_[name].data_type);
}
int32_t BM168xNet::updateOutputTensors() {
  for (auto it = output_tensor_hash_.begin(); it != output_tensor_hash_.end();
       it++) {
    std::string name = it->first;
    std::shared_ptr<BaseTensor> output_tensor = it->second;
    std::vector<int> shapes = output_tensor->getShape();

    int index = output_name_index_[name];
    // auto &out_shape = net_info_->stages[0].output_shapes[index];
    // for(int k = 0; k< out_shape.num_dims;k++){
    //  LOG(INFO)<<"update
    //  output:"<<name<<",dim"<<k<<",shape:"<<out_shape.dims[k];
    //}
    LOGI("update output:%s,shape:%d,%d,%d,%d", name.c_str(), shapes[0],
         shapes[1], shapes[2], shapes[3]);

    output_tensor->invalidateCache();
  }
  return 0;
}

int32_t BM168xNet::forward(bool sync) {
  LOGI("input num:%d,output num:%d", net_info_->input_num,
       net_info_->output_num);
  for (int i = 0; i < net_info_->input_num; i++) {
    if (bm_mem_get_type(input_tensors_[i].device_mem) != BM_MEM_TYPE_DEVICE) {
      LOGE("input tensor is not device memory,%s,input:%d", net_info_->name, i);
    }
  }
  LOGI("start to check output tensor");
  for (int i = 0; i < net_info_->output_num; i++) {
    if (bm_mem_get_type(output_tensors_[i].device_mem) != BM_MEM_TYPE_DEVICE) {
      LOGE("output tensor is not device memory,%s,output:%d", net_info_->name,
           i);
    }
  }
  LOGI("start to do inference");
  bool ret = bmrt_launch_tensor_ex(p_bmrt_, net_info_->name, input_tensors_,
                                   net_info_->input_num, output_tensors_,
                                   net_info_->output_num, true, false);
  if (!ret) {
    LOGE("failed to do inference,%s", net_info_->name);
  }
  // sync, wait for finishing inference
  bmrt_thread_sync(p_bmrt_);

  LOGI("forward success");
  return 0;
}
