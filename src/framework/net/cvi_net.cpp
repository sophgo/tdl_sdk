#include "net/cvi_net.hpp"
#ifdef __CMODEL_CV181X__
#include "memory/cpu_memory_pool.hpp"
#else
#include "memory/cvi_memory_pool.hpp"
#endif
#include "utils/tdl_log.hpp"

CviNet::CviNet(const NetParam& param) : BaseNet(param) {}

CviNet::~CviNet() {
  if (model_handle_ != nullptr) {
    int32_t ret = CVI_NN_CleanupModel(model_handle_);
    if (ret != CVI_RC_SUCCESS) {  // NOLINT
      LOGE("CVI_NN_CleanupModel failed: %d\n", ret);
    }
  }
  model_handle_ = nullptr;
}

int32_t CviNet::setup() {
  LOGI("to setup CviNet,model_file_path: %s",
       net_param_.model_file_path.c_str());
  int ret = 0;
  if (net_param_.model_file_path.empty()) {
    if (net_param_.model_buffer == nullptr ||
        net_param_.model_buffer_size == 0) {
      LOGE("model_buffer is nullptr or model_buffer_size is 0");
      return -1;
    }
    ret = CVI_NN_RegisterModelFromBuffer(
        reinterpret_cast<const int8_t*>(net_param_.model_buffer),
        net_param_.model_buffer_size, &model_handle_);
  } else {
    if (net_param_.model_file_path.empty()) {
      LOGE("model_file_path is empty");
      return -1;
    }
    ret = CVI_NN_RegisterModel(net_param_.model_file_path.c_str(),
                               &model_handle_);
  }

  if (ret != 0) {
    LOGE("CVI_NN_RegisterModel failed");
    return ret;
  }
  LOGI("CVI_NN_RegisterModel success");
  ret = CVI_NN_SetConfig(model_handle_, OPTION_OUTPUT_ALL_TENSORS, 0);
  if (ret != 0) {
    LOGE("CVI_NN_SetConfig failed");
    return ret;
  }
  LOGI("CVI_NN_SetConfig success");
  CVI_TENSOR *input_tensors, *output_tensors;
  int32_t input_tensors_num, output_tensors_num;
  ret = CVI_NN_GetInputOutputTensors(model_handle_, &input_tensors,
                                     &input_tensors_num, &output_tensors,
                                     &output_tensors_num);
  if (ret != 0) {
    LOGE("CVI_NN_GetInputOutputTensors failed");
    return ret;
  }
  LOGI("CVI_NN_GetInputOutputTensors success");

  input_tensors_ = input_tensors;
  output_tensors_ = output_tensors;

#ifdef __CMODEL_CV181X__
  memory_pool_ = std::make_shared<CpuMemoryPool>();
#else
  memory_pool_ = std::make_shared<CviMemoryPool>();
#endif
  input_output_tensor_infos_.clear();
  setupTensorInfo(input_tensors, input_tensors_num, input_output_tensor_infos_);
  setupTensorInfo(output_tensors, output_tensors_num,
                  input_output_tensor_infos_);

  for (int i = 0; i < input_tensors_num; i++) {
    input_tensor_names_.push_back(CVI_NN_TensorName(&input_tensors[i]));
    addInput(input_tensor_names_[i]);
    supported_batch_sizes_[input_tensor_names_[i]] = {1};
  }
  for (int i = 0; i < output_tensors_num; i++) {
    output_tensor_names_.push_back(CVI_NN_TensorName(&output_tensors[i]));
    addOutput(output_tensor_names_[i]);
  }

  return 0;
}

void CviNet::setupTensorInfo(CVI_TENSOR* cvi_tensor, int32_t num_tensors,
                             std::map<std::string, TensorInfo>& tensor_info) {
  for (int32_t i = 0; i < num_tensors; i++) {
    TensorInfo tinfo;
    tinfo.tensor_handle = cvi_tensor + i;
    std::string tensor_name = CVI_NN_TensorName(cvi_tensor + i);
    CVI_SHAPE shape = CVI_NN_TensorShape(cvi_tensor + i);
    for (size_t j = 0; j < shape.dim_size; j++) {
      tinfo.shape.push_back(shape.dim[j]);
    }
    tinfo.sys_mem = (uint8_t*)CVI_NN_TensorPtr(cvi_tensor + i);
    assert(tinfo.sys_mem == cvi_tensor[i].sys_mem);
    tinfo.phy_addr =
        cvi_tensor[i].paddr;  // CVI_NN_TensorPhyAddr(cvi_tensor + i);
    tinfo.tensor_elem = CVI_NN_TensorCount(cvi_tensor + i);
    tinfo.tensor_size = CVI_NN_TensorSize(cvi_tensor + i);
    tinfo.qscale = CVI_NN_TensorQuantScale(cvi_tensor + i);
    tinfo.zero_point = CVI_NN_TensorQuantZeroPoint(cvi_tensor + i);
    if (cvi_tensor[i].fmt == CVI_FMT_UINT8) {
      tinfo.data_type = TDLDataType::UINT8;
    } else if (cvi_tensor[i].fmt == CVI_FMT_INT8) {
      tinfo.data_type = TDLDataType::INT8;
    } else if (cvi_tensor[i].fmt == CVI_FMT_FP32) {
      tinfo.data_type = TDLDataType::FP32;
    } else if (cvi_tensor[i].fmt == CVI_FMT_BF16) {
      tinfo.data_type = TDLDataType::BF16;
    } else if (cvi_tensor[i].fmt == CVI_FMT_INT16) {
      tinfo.data_type = TDLDataType::INT16;
    } else if (cvi_tensor[i].fmt == CVI_FMT_UINT16) {
      tinfo.data_type = TDLDataType::UINT16;
    } else if (cvi_tensor[i].fmt == CVI_FMT_INT32) {
      tinfo.data_type = TDLDataType::INT32;
    } else if (cvi_tensor[i].fmt == CVI_FMT_UINT32) {
      tinfo.data_type = TDLDataType::UINT32;
    } else {
      LOGE("unknown tensor format %d", cvi_tensor[i].fmt);
    }
    if (tinfo.data_type == TDLDataType::FP32 && tinfo.qscale == 0) {
      LOGI("tensor_name:%s,qscale is 0,set to 1", tensor_name.c_str());
      tinfo.qscale = 1;
    }
    LOGI(
        "tensor_name:%s,shape:%d,%d,%d,%d,qscale:%f,zero_point:%d,dtype:%d,sys_"
        "mem:%p,phy_addr:%"
        "llu",
        tensor_name.c_str(), tinfo.shape[0], tinfo.shape[1], tinfo.shape[2],
        tinfo.shape[3], tinfo.qscale, tinfo.zero_point,
        static_cast<int>(tinfo.data_type), tinfo.sys_mem, tinfo.phy_addr);
    tensor_info[tensor_name] = tinfo;
  }
}

int32_t CviNet::forward(bool sync) {
  int ret = CVI_NN_Forward(model_handle_, (CVI_TENSOR*)input_tensors_,
                           (int)input_tensor_names_.size(),
                           (CVI_TENSOR*)output_tensors_,
                           (int)output_tensor_names_.size());
  if (ret != 0) {
    LOGE("CVI_NN_Forward failed,net_name:%s",
         net_param_.model_file_path.c_str());
    return ret;
  }
  return 0;
}

int32_t CviNet::addInput(const std::string& name) {
  if (input_tensor_hash_.find(name) != input_tensor_hash_.end()) {
    LOGI("Layer %s is already exist in net", name.c_str());
    return 0;
  }

  TensorInfo& tinfo = input_output_tensor_infos_[name];
  int element_bytes = tinfo.tensor_size / tinfo.tensor_elem;
  input_tensor_hash_[name] =
      std::make_shared<BaseTensor>(element_bytes, memory_pool_);
  auto& shape = input_output_tensor_infos_[name].shape;
  input_tensor_hash_[name]->shareMemory(tinfo.sys_mem, tinfo.phy_addr,
                                        element_bytes, shape);
  LOGI(
      "finish add "
      "input:%s,reset input "
      "memory,element_bytes:%d,shape:%d,%d,%d,%d,qscale:%f,zero_point:%d,"
      "dtype:%d,sys_mem:%p,phy_addr:%llu",
      name.c_str(), element_bytes, shape[0], shape[1], shape[2], shape[3],
      input_output_tensor_infos_[name].qscale,
      input_output_tensor_infos_[name].zero_point,
      static_cast<int>(input_output_tensor_infos_[name].data_type),
      tinfo.sys_mem, tinfo.phy_addr);

  return 0;
}

int32_t CviNet::addOutput(const std::string& name) {
  if (output_tensor_hash_.find(name) != output_tensor_hash_.end()) {
    LOGI("Layer %s is already exist in net", name.c_str());
    return 0;
  }
  LOGI("start to add output :%s", name.c_str());

  TensorInfo& tinfo = input_output_tensor_infos_[name];
  int element_bytes = tinfo.tensor_size / tinfo.tensor_elem;
  output_tensor_hash_[name] =
      std::make_shared<BaseTensor>(element_bytes, memory_pool_);
  auto& shape = input_output_tensor_infos_[name].shape;
  output_tensor_hash_[name]->shareMemory(tinfo.sys_mem, tinfo.phy_addr,
                                         element_bytes, shape);

  LOGI(
      "finish add output:%s,use share "
      "memory,element_bytes:%d,shape:%d,%d,%d,%d,qscale:%f,"
      "zero_point:%d,dtype:%d",
      name.c_str(), element_bytes, shape[0], shape[1], shape[2], shape[3],
      input_output_tensor_infos_[name].qscale,
      input_output_tensor_infos_[name].zero_point,
      static_cast<int>(input_output_tensor_infos_[name].data_type));
  return 0;
}

std::shared_ptr<BaseTensor> CviNet::getInputTensor(const std::string& name) {
  return input_tensor_hash_[name];
}

std::shared_ptr<BaseTensor> CviNet::getOutputTensor(const std::string& name) {
  return output_tensor_hash_[name];
}

int32_t CviNet::updateInputTensors() { return 0; }

int32_t CviNet::updateOutputTensors() { return 0; }