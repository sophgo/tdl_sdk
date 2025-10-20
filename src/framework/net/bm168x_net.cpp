#include "net/bm168x_net.hpp"

#include <bmruntime_interface.h>
#include <bmruntime_legacy.h>

#include <cassert>
#include <cstring>
#include <sstream>

#include "memory/bm_memory_pool.hpp"
#include "utils/common_utils.hpp"
#include "utils/tdl_log.hpp"

BM168xNet::BM168xNet(const NetParam &net_param) : BaseNet(net_param) {}

BM168xNet::~BM168xNet() {
  if (input_tensors_) {
    delete[] input_tensors_;
    input_tensors_ = 0;
  }
  if (output_tensors_) {
    delete[] output_tensors_;
    output_tensors_ = 0;
  }
  if (p_bmrt_ != nullptr) {
    LOGI("destroy bmrt");
    bmrt_destroy(p_bmrt_);
    p_bmrt_ = nullptr;
  }
  net_info_ = nullptr;
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

  LOGI("to get_model_bmrt: %s", net_param_.model_file_path.c_str());
  p_bmrt_ = bmrt_create(bm_handle_);

  int32_t ret = loadModel();
  if (ret != 0) {
    LOGE("load model failed");
    return -1;
  }
  // if name was not set,use the name inside bmodel as default
  std::string net_name = net_param_.model_config.net_name;
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

int32_t BM168xNet::loadModel() {
  bool flag = false;
  LOGI("runtime_mem_addrs size:%d,runtime_mem_sizes size:%d",
       net_param_.runtime_mem_addrs.size(),
       net_param_.runtime_mem_sizes.size());
  if (net_param_.runtime_mem_addrs.size() == 5) {
    LOGI("runtime_mem_addrs=[0x%llx, 0x%llx, 0x%llx, 0x%llx, 0x%llx]",
         (unsigned long long)net_param_.runtime_mem_addrs[0],
         (unsigned long long)net_param_.runtime_mem_addrs[1],
         (unsigned long long)net_param_.runtime_mem_addrs[2],
         (unsigned long long)net_param_.runtime_mem_addrs[3],
         (unsigned long long)net_param_.runtime_mem_addrs[4]);
    LOGI("runtime_mem_sizes=[%u, %u, %u, %u, %u]",
         net_param_.runtime_mem_sizes[0], net_param_.runtime_mem_sizes[1],
         net_param_.runtime_mem_sizes[2], net_param_.runtime_mem_sizes[3],
         net_param_.runtime_mem_sizes[4]);
  }
  if (net_param_.model_file_path.empty()) {
    if (net_param_.model_buffer == nullptr ||
        net_param_.model_buffer_size == 0) {
      LOGE("model_buffer is nullptr or model_buffer_size is 0");
      return -1;
    }
    if (net_param_.runtime_mem_addrs.size() == 5 &&
        net_param_.runtime_mem_sizes.size() == 5) {
#ifndef __BM1684X__
      mem_info_t mem_info;
      if (bmrt_get_bmodel_data_info(net_param_.model_buffer,
                                    net_param_.model_buffer_size,
                                    &mem_info) == false) {
        LOGE("bmrt_get_bmodel_data_info failed");
        return -1;
      }
      if (updateMemoryInfo(net_param_.runtime_mem_addrs,
                           net_param_.runtime_mem_sizes, &mem_info) != 0) {
        LOGE("updateMemoryInfo failed");
        return -1;
      }
      flag = bmrt_load_bmodel_data_with_mem(p_bmrt_, net_param_.model_buffer,
                                            net_param_.model_buffer_size,
                                            &mem_info);
      if (flag == false) {
        LOGE("bmrt_load_bmodel_data_with_mem failed");
        return -1;
      }
      // Use the same mem_info for multi-thread pre-allocation
      flag = bmrt_pre_alloc_mem_multi_thread(p_bmrt_, 0, &mem_info);
      if (flag == false) {
        LOGE("bmrt_pre_alloc_mem_multi_thread failed");
        return -1;
      }
      use_runtime_memory_ = true;
#else
      LOGE("bm1684x not support load bmodel with mem");
      return -1;
#endif
    } else {
      flag = bmrt_load_bmodel_data(p_bmrt_, net_param_.model_buffer,
                                   net_param_.model_buffer_size);
    }

  } else {
    if (net_param_.runtime_mem_addrs.size() == 5 &&
        net_param_.runtime_mem_sizes.size() == 5) {
#ifndef __BM1684X__
      mem_info_t mem_info;
      if (bmrt_get_bmodel_info(net_param_.model_file_path.c_str(), &mem_info) ==
          false) {
        LOGE("bmrt_get_bmodel_info failed");
        return -1;
      }
      if (updateMemoryInfo(net_param_.runtime_mem_addrs,
                           net_param_.runtime_mem_sizes, &mem_info) != 0) {
        LOGE("updateMemoryInfo failed");
        return -1;
      }
      LOGI("to load bmodel with mem");
      flag = bmrt_load_bmodel_with_mem(
          p_bmrt_, net_param_.model_file_path.c_str(), &mem_info);
      if (flag == false) {
        LOGE("bmrt_load_bmodel_with_mem failed");
        return -1;
      }
      // Use the same mem_info for multi-thread pre-allocation
      LOGI("to pre-alloc mem for multi-thread");
      flag = bmrt_pre_alloc_mem_multi_thread(p_bmrt_, 0, &mem_info);
      if (flag == false) {
        LOGE("bmrt_pre_alloc_mem_multi_thread failed");
        return -1;
      }
      use_runtime_memory_ = true;
      LOGI("load bmodel with mem done,flag:%d", flag);
#else
      LOGE("bm1684x not support load bmodel with mem");
      return -1;
#endif
    } else {
      flag = bmrt_load_bmodel(p_bmrt_, net_param_.model_file_path.c_str());
    }
  }
  if (!flag) {
    LOGE("model load failed");
    return -1;
  }
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
  for (int i = 0; i < p_shape[idx].num_dims; i++) {
    tensor_info.shape[i] = p_shape[idx].dims[i];
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
  } else if (p_data_type[idx] == BM_INT32) {
    tensor_info.data_type = TDLDataType::INT32;
  } else {
    LOGE("unsupported data type:%d", p_data_type[idx]);
    assert(0);
  }
  uint32_t data_type_size = CommonUtils::getDataTypeSize(tensor_info.data_type);
  if (data_type_size != bmrt_data_type_size(p_data_type[idx])) {
    LOGE("data type size not equal,expect:%d,got:%d", data_type_size,
         bmrt_data_type_size(p_data_type[idx]));
  }
  tensor_info.tensor_elem = tensor_info.shape[0] * tensor_info.shape[1] *
                            tensor_info.shape[2] * tensor_info.shape[3];
  tensor_info.tensor_size =
      CommonUtils::getDataTypeSize(tensor_info.data_type) *
      tensor_info.tensor_elem;
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
  input_tensor_hash_[name]->reshape(shape[0], shape[1], shape[2], shape[3]);
  TensorInfo &tensor_info = input_output_tensor_infos_[name];
  tensor_info.sys_mem = reinterpret_cast<uint8_t *>(
      input_tensor_hash_[name]->getMemoryBlock()->virtualAddress);
  tensor_info.phy_addr =
      input_tensor_hash_[name]->getMemoryBlock()->physicalAddress;
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
  output_tensor_hash_[name]->reshape(shape[0], shape[1], shape[2], shape[3]);
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
    int num_bmrt_elems = 1;
    for (int i = 0; i < bmrt_shape.num_dims; i++) {
      num_bmrt_elems *= bmrt_shape.dims[i];
    }
    int num_input_elems = 1;
    for (int i = 0; i < input_shape.size(); i++) {
      num_input_elems *= input_shape[i];
    }
    if (num_bmrt_elems != num_input_elems) {
      LOGE("num_bmrt_elems:%d,num_input_elems:%d", num_bmrt_elems,
           num_input_elems);
      assert(0);
    }

    MemoryBlock *memory_block = input_tensor->getMemoryBlock();
    bm_device_mem_t dev =
        bm_mem_from_device(memory_block->physicalAddress, memory_block->size);

    bmrt_shape.dims[0] = input_shape[0];
    input_tensors_[tensor_idx].dtype = net_info_->input_dtypes[tensor_idx];
    input_tensors_[tensor_idx].shape = bmrt_shape;  //  num_dims = 4;
    // memcpy(input_tensors_[tensor_idx].shape.dims,&input_shape[0],4*sizeof(int));
    input_tensors_[tensor_idx].device_mem = dev;
    input_tensors_[tensor_idx].st_mode = BM_STORE_1N;
    LOGI("add input:%s,shape:%d,%d,%d,%d,dev_addr:0x%llx,dtype:%d",
         name.c_str(), batch_n, input_shape[1], input_shape[2], input_shape[3],
         (unsigned long long)memory_block->physicalAddress,
         input_tensors_[tensor_idx].dtype);
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
    for (int i = 0; i < out_shape.num_dims; i++) {
      st[i] = out_shape.dims[i];
    }

    BaseTensor *tensor = dynamic_cast<BaseTensor *>(output_tensor.get());
    MemoryBlock *memory_block = tensor->getMemoryBlock();
    bm_device_mem_t dev =
        bm_mem_from_device(memory_block->physicalAddress, memory_block->size);

    // Handle special case: when input batch_n=1 but output first dim > 1
    // This is common for models like simcc_pose where output shape is [1, 17,
    // 384]
    if (batch_n == 1 && out_shape.dims[0] > 1) {
      LOGW("special case,batch_n:%d,out_shape.dims[0]:%d", batch_n,
           out_shape.dims[0]);
      output_tensor->reshape(out_shape.dims[0], st[1], st[2], st[3]);
    } else {
      out_shape.dims[0] = batch_n;
      output_tensor->reshape(batch_n, st[1], st[2], st[3]);
    }

    // output_dev_mem.size = output_tensor->get_size();
    output_tensors_[tensor_idx].dtype = net_info_->output_dtypes[tensor_idx];
    output_tensors_[tensor_idx].shape = out_shape;
    output_tensors_[tensor_idx].device_mem = dev;
    output_tensors_[tensor_idx].st_mode = (bm_store_mode_t)store_mode_;
    LOGI("add output:%s,shape:%d,%d,%d,%d,dev_addr:0x%llx,dtype:%d",
         name.c_str(), batch_n, st[1], st[2], st[3],
         (unsigned long long)memory_block->physicalAddress,
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
      CommonUtils::getDataTypeSize(input_output_tensor_infos_[name].data_type);
}
int32_t BM168xNet::updateOutputTensors() {
  for (auto it = output_tensor_hash_.begin(); it != output_tensor_hash_.end();
       it++) {
    std::string name = it->first;
    std::shared_ptr<BaseTensor> output_tensor = it->second;
    std::vector<int> shapes = output_tensor->getShape();

    int index = output_name_index_[name];

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

int32_t BM168xNet::updateMemoryInfo(const std::vector<uint64_t> &mem_addrs,
                                    const std::vector<uint32_t> &mem_sizes,
                                    void *mem_info_ptr) {
#ifndef __BM1684X__
  mem_info_t *mem_info = (mem_info_t *)mem_info_ptr;
  if (mem_addrs.size() != 5 || mem_sizes.size() != 5) {
    LOGE("memory addrs size not equal to 5");
    return -1;
  }

  // Helper function to set memory address with validation
  auto set_addr = [&](memory_t &dst_mem, const uint64_t src_addr,
                      const uint32_t src_size,
                      const std::string &name) -> bool {
    if (dst_mem.size == 0) {
      LOGI("%s size is 0, skip", name.c_str());
      return true;
    }
    // User-provided memory (src_size) must be >= model requirement
    // (dst_mem.size)
    if (src_size < dst_mem.size) {
      LOGE("%s user_provided_size:%u < required_size:%u", name.c_str(),
           src_size, dst_mem.size);
      return false;
    }
    dst_mem.addr = src_addr;
    LOGI("%s set addr:0x%llx, size:%u", name.c_str(),
         (unsigned long long)dst_mem.addr, dst_mem.size);
    return true;
  };

  // Set all memory blocks in the same mem_info structure
  if (!set_addr(mem_info->instruction_mem, mem_addrs[0], mem_sizes[0],
                "instruction_mem")) {
    return -1;
  }
  if (!set_addr(mem_info->variable_instruction_mem, mem_addrs[1], mem_sizes[1],
                "variable_instruction_mem")) {
    return -1;
  }
  if (!set_addr(mem_info->neuron_mem, mem_addrs[2], mem_sizes[2],
                "neuron_mem")) {
    return -1;
  }
  if (!set_addr(mem_info->coeff_mem, mem_addrs[3], mem_sizes[3], "coeff_mem")) {
    return -1;
  }
  if (!set_addr(mem_info->io_mem, mem_addrs[4], mem_sizes[4], "io_mem")) {
    return -1;
  }

  // Set number to 1 for single-thread scenario
  if (mem_info->neuron_mem.size > 0) {
    mem_info->neuron_mem.number = 1;
  }
  if (mem_info->io_mem.size > 0) {
    mem_info->io_mem.number = 1;
  }

  LOGI("mem_info update done");
  return 0;
#else

  return -1;
#endif
}
int32_t BM168xNet::getModelMemInfo(const std::string &model_file,
                                   std::vector<uint64_t> &mem_addrs,
                                   std::vector<uint32_t> &mem_sizes) {
#ifndef __BM1684X__
  mem_info_t mem_info;
  if (bmrt_get_bmodel_info(model_file.c_str(), &mem_info) == false) {
    LOGE("bmrt_get_bmodel_info failed, model_file: %s", model_file.c_str());
    return -1;
  }

  mem_addrs.push_back(mem_info.instruction_mem.addr);
  mem_sizes.push_back(mem_info.instruction_mem.size);
  mem_addrs.push_back(mem_info.variable_instruction_mem.addr);
  mem_sizes.push_back(mem_info.variable_instruction_mem.size);
  mem_addrs.push_back(mem_info.neuron_mem.addr);
  mem_sizes.push_back(mem_info.neuron_mem.size);
  mem_addrs.push_back(mem_info.coeff_mem.addr);
  mem_sizes.push_back(mem_info.coeff_mem.size);
  mem_addrs.push_back(mem_info.io_mem.addr);
  mem_sizes.push_back(mem_info.io_mem.size);

  LOGI("instruction_mem.addr:0x%llx,instruction_mem.size:%u,number:%d",
       (unsigned long long)mem_info.instruction_mem.addr,
       mem_info.instruction_mem.size, mem_info.instruction_mem.number);
  LOGI(
      "variable_instruction_mem.addr:0x%llx,variable_instruction_mem.size:%u,"
      "number:%d",
      (unsigned long long)mem_info.variable_instruction_mem.addr,
      mem_info.variable_instruction_mem.size,
      mem_info.variable_instruction_mem.number);
  LOGI("neuron_mem.addr:0x%llx,neuron_mem.size:%u,number:%d",
       (unsigned long long)mem_info.neuron_mem.addr, mem_info.neuron_mem.size,
       mem_info.neuron_mem.number);
  LOGI("coeff_mem.addr:0x%llx,coeff_mem.size:%u,number:%d",
       (unsigned long long)mem_info.coeff_mem.addr, mem_info.coeff_mem.size,
       mem_info.coeff_mem.number);
  LOGI("io_mem.addr:0x%llx,io_mem.size:%u,number:%d",
       (unsigned long long)mem_info.io_mem.addr, mem_info.io_mem.size,
       mem_info.io_mem.number);

  LOGI("Summary: mem_sizes=[%u, %u, %u, %u, %u]", mem_info.instruction_mem.size,
       mem_info.variable_instruction_mem.size, mem_info.neuron_mem.size,
       mem_info.coeff_mem.size, mem_info.io_mem.size);
#else
  LOGE("bm1684x not support getModelMemInfo with model_file");
  return -1;
#endif
  return 0;
}

int32_t BM168xNet::getModelMemInfo(const void *model_buffer,
                                   const uint32_t model_buffer_size,
                                   std::vector<uint64_t> &mem_addrs,
                                   std::vector<uint32_t> &mem_sizes) {
#ifndef __BM1684X__
  mem_info_t mem_info;
  if (bmrt_get_bmodel_data_info(model_buffer, model_buffer_size, &mem_info) ==
      false) {
    LOGE(
        "bmrt_get_bmodel_data_info failed, model_buffer: %p, "
        "model_buffer_size: %d",
        model_buffer, model_buffer_size);
    return -1;
  }

  mem_addrs.push_back(mem_info.instruction_mem.addr);
  mem_sizes.push_back(mem_info.instruction_mem.size);
  mem_addrs.push_back(mem_info.variable_instruction_mem.addr);
  mem_sizes.push_back(mem_info.variable_instruction_mem.size);
  mem_addrs.push_back(mem_info.neuron_mem.addr);
  mem_sizes.push_back(mem_info.neuron_mem.size);
  mem_addrs.push_back(mem_info.coeff_mem.addr);
  mem_sizes.push_back(mem_info.coeff_mem.size);
  mem_addrs.push_back(mem_info.io_mem.addr);
  mem_sizes.push_back(mem_info.io_mem.size);
#else
  LOGE("bm1684x not support getModelMemInfo with model_buffer");
  return -1;
#endif
  return 0;
}