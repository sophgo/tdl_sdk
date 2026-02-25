#include "feature_extract/feature_extraction.hpp"
#include <numeric>
#include "utils/detection_helper.hpp"
#include "utils/tdl_log.hpp"

FeatureExtraction::FeatureExtraction() {}

FeatureExtraction::~FeatureExtraction() {}

// Callback after model is opened, no extra handling required.
int FeatureExtraction::onModelOpened() { return 0; }

int32_t FeatureExtraction::outputParse(
    const std::vector<std::shared_ptr<BaseImage>> &images,
    std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) {
  if (net_param_.model_config.std.size() == 0 ||
      net_param_.model_config.std[0] == 0) {
    LOGE("model_config is not set");
    // assert(false); // 建议注释掉 assert 防止崩在通用模型上
    return -1;
  }

  std::string input_tensor_name = net_->getInputNames()[0];
  TensorInfo input_tensor = net_->getTensorInfo(input_tensor_name);

  LOGI("outputParse,batch size:%d,input shape:%d,%d,%d,%d", (int)images.size(),
       input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2],
       input_tensor.shape[3]);

  const auto &output_layers = net_->getOutputNames();
  TensorInfo output_info = net_->getTensorInfo(output_layers[0]);
  std::shared_ptr<BaseTensor> output_tensor =
      net_->getOutputTensor(output_layers[0]);

  auto output_shape = output_info.shape;

  if (output_shape.size() < 2) {
    LOGE("invalid output shape size:%zu", output_shape.size());
    return -1;
  }

  // 计算输出特征长度 (兼容多维 shape)
  int batch_elem_num = 1;
  for (size_t i = 1; i < output_shape.size(); ++i) {
    batch_elem_num *= output_shape[i];
  }

  uint32_t batch =
      std::min<uint32_t>(input_tensor.shape[0], (uint32_t)output_shape[0]);

  for (uint32_t b = 0; b < batch; b++) {
    std::shared_ptr<ModelFeatureInfo> feature_meta =
        std::make_shared<ModelFeatureInfo>();
    feature_meta->embedding_num = batch_elem_num;
    feature_meta->embedding_type = output_info.data_type;

    if (output_info.data_type == TDLDataType::FP32) {
      feature_meta->embedding =
          (uint8_t *)malloc(batch_elem_num * sizeof(float));
      memcpy(feature_meta->embedding, output_tensor->getBatchPtr<float>(b),
             batch_elem_num * sizeof(float));

    } else if (output_info.data_type == TDLDataType::INT8) {
      feature_meta->embedding =
          (uint8_t *)malloc(batch_elem_num * sizeof(int8_t));
      memcpy(feature_meta->embedding, output_tensor->getBatchPtr<int8_t>(b),
             batch_elem_num * sizeof(int8_t));

    } else if (output_info.data_type == TDLDataType::UINT8) {
      feature_meta->embedding =
          (uint8_t *)malloc(batch_elem_num * sizeof(uint8_t));
      memcpy(feature_meta->embedding, output_tensor->getBatchPtr<uint8_t>(b),
             batch_elem_num * sizeof(uint8_t));
    } else {
      LOGE("not supported data type:%d", (int)output_info.data_type);
    }

    out_datas.push_back(feature_meta);
  }
  return 0;
}