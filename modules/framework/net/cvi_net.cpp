#include "net/cvi_net.hpp"

#include <cviruntime.h>

#include "cvi_tdl_log.hpp"
#include "net/cvi_tensor.hpp"

CviNet::CviNet(const NetParam& param) : BaseNet(param) {}

CviNet::~CviNet() {}

int32_t CviNet::setup() {
  int ret = CVI_NN_RegisterModel(filepath, &model_handle_);
  if (ret != 0) {
    LOGE("CVI_NN_RegisterModel failed");
    return ret;
  }

  ret = CVI_NN_SetConfig(model_handle_, OPTION_OUTPUT_ALL_TENSORS, 0);
  if (ret != 0) {
    LOGE("CVI_NN_SetConfig failed");
    return ret;
  }
  CVI_TENSOR *input_tensors, *output_tensors;
  int32_t input_tensors_num, output_tensors_num;
  ret = CVI_NN_GetInputOutputTensors(model_handle_, &input_tensors,
                                     &input_tensors_num, &output_tensors,
                                     &output_tensors_num);
  if (ret != 0) {
    LOGE("CVI_NN_GetInputOutputTensors failed");
    return ret;
  }

  input_tensors_ = input_tensors;
  output_tensors_ = output_tensors;

  setupTensorInfo(input_tensors, input_tensors_num, input_tensor_infos_);
  setupTensorInfo(output_tensors, output_tensors_num, output_tensor_infos_);

  for (int i = 0; i < input_tensors_num; i++) {
    input_tensor_names_.push_back(CVI_NN_TensorName(input_tensors[i]));
    addInput(input_tensor_names_[i]);
  }
  for (int i = 0; i < output_tensors_num; i++) {
    output_tensor_names_.push_back(CVI_NN_TensorName(output_tensors[i]));
    addOutput(output_tensor_names_[i]);
  }

  supported_batch_sizes_ = {1};
  return 0;
}

void CviNet::setupTensorInfo(void* cvi_tensor, int32_t num_tensors,
                             std::map<std::string, TensorInfo>& tensor_info) {
  tensor_info.clear();
  CVI_TENSOR* tensor = (CVI_TENSOR*)cvi_tensor;
  for (int32_t i = 0; i < num_tensors; i++) {
    TensorInfo tinfo;
    tinfo.tensor_handle = tensor + i;
    std::string tensor_name = CVI_NN_TensorName(tinfo.tensor_handle);
    CVI_SHAPE shape = CVI_NN_TensorShape(tinfo.tensor_handle);
    for (int j = 0; j < shape.dim_size; j++) {
      tinfo.shape.push_back(shape.dim[j]);
    }
    tinfo.sys_mem = CVI_NN_TensorPtr(tinfo.tensor_handle);
    tinfo.phy_addr = CVI_NN_TensorPhyAddr(tinfo.tensor_handle);
    tinfo.tensor_elem = CVI_NN_TensorCount(tinfo.tensor_handle);
    tinfo.tensor_size = CVI_NN_TensorSize(tinfo.tensor_handle);
    tinfo.qscale = CVI_NN_TensorQuantScale(tinfo.tensor_handle);
    tinfo.zero_point = CVI_NN_TensorZeroPoint(tinfo.tensor_handle);
    if (tensor[i].fmt == CVI_FMT_UINT8) {
      tinfo.data_type = ImagePixDataType::UINT8;
    } else if (tensor[i].fmt == CVI_FMT_INT8) {
      tinfo.data_type = ImagePixDataType::INT8;
    } else if (tensor[i].fmt == CVI_FMT_FLOAT32) {
      tinfo.data_type = ImagePixDataType::FP32;
    } else if (tensor[i].fmt == CVI_FMT_BF16) {
      tinfo.data_type = ImagePixDataType::BF16;
    } else if (tensor[i].fmt == CVI_FMT_INT16) {
      tinfo.data_type = ImagePixDataType::INT16;
    } else if (tensor[i].fmt == CVI_FMT_UINT16) {
      tinfo.data_type = ImagePixDataType::UINT16;
    } else if (tensor[i].fmt == CVI_FMT_INT32) {
      tinfo.data_type = ImagePixDataType::INT32;
    } else if (tensor[i].fmt == CVI_FMT_UINT32) {
      tinfo.data_type = ImagePixDataType::UINT32;
    } else {
      LOGE("unknown tensor format %d", tensor[i].fmt);
    }
    tensor_info[tensor_name] = tinfo;
  }
}

int32_t CviNet::forward(bool sync) {
  int ret = CVI_NN_Forward(model_handle_, (CVI_TENSOR*)input_tensors_,
                           (int)input_tensor_names_.size(),
                           (CVI_TENSOR*)output_tensors_,
                           (int)output_tensor_names_.size());
  if (ret != 0) {
    LOGE("CVI_NN_Forward failed,net_name:%s", net_param_.name.c_str());
    return ret;
  }
  return 0;
}

int32_t CviNet::addInput(const std::string& name) {
  auto tensor = std::make_shared<CviTensor>();
  TensorInfo& tinfo = input_tensor_infos_[name];
  int element_size = tinfo.tensor_size / tinfo.tensor_elem;
  tensor->shareMemory(tinfo.sys_mem, tinfo.phy_addr, element_size, tinfo.shape);
  if (net_param_.share_input_mem && tensor->getWidth() % 64 == 0) {
    // release input tensor memory
    CVI_NN_SetTensorPhysicalAddr((CVI_TENSOR*)tinfo.tensor_handle, 0);
    tensor->shareMemory(nullptr, 0, element_size, tinfo.shape);
    LOGI("input tensor %s share memory", name.c_str());
  }
  input_tensor_hash_[name] = tensor;
  return 0;
}

int32_t CviNet::addOutput(const std::string& name) {
  auto tensor = std::make_shared<CviTensor>();
  TensorInfo& tinfo = output_tensor_infos_[name];
  int element_size = tinfo.tensor_size / tinfo.tensor_elem;
  tensor->shareMemory(tinfo.sys_mem, tinfo.phy_addr, element_size, tinfo.shape);
  // if (net_param_.share_output_mem && tensor->getWidth() % 64 == 0) {
  //   CVI_NN_SetTensorPhysicalAddr((CVI_TENSOR*)tinfo.tensor_handle, 0);
  //   tensor->shareMemory(nullptr, 0, element_size, tinfo.shape);
  //   LOGI("output tensor %s share memory", name.c_str());
  // }
  output_tensor_hash_[name] = tensor;
  return 0;
}

std::shared_ptr<BaseTensor> CviNet::getInputTensor(const std::string& name) {
  return input_tensor_hash_[name];
}

std::shared_ptr<BaseTensor> CviNet::getOutputTensor(const std::string& name) {
  return output_tensor_hash_[name];
}

int32_t CviNet::setInputTensorPhyAddr(const std::string& tensor_name,
                                      uint64_t phy_addr) {
  int idx = std::find(input_tensor_names_.begin(), input_tensor_names_.end(),
                      tensor_name) -
            input_tensor_names_.begin();
  if (idx < 0 || idx >= input_tensor_names_.size()) {
    LOGE("input tensor %s not found", tensor_name.c_str());
    return -1;
  }
  TensorInfo& tinfo = input_tensor_infos_[tensor_name];
  CVI_NN_SetTensorPhysicalAddr((CVI_TENSOR*)tinfo.tensor_handle, phy_addr);
  return 0;
}
