#include "feature_extract/clip_image.hpp"

#include "utils/tdl_log.hpp"

template <typename T>
void parse_feature_info(T *data, int num_elem, float qscale, float *features) {
  for (int i = 0; i < num_elem; i++) {
    features[i] = data[i] * qscale;
  }
}
Clip_Image::Clip_Image() {
  net_param_.model_config.std = {1.0 / 0.0145984266, 1.0 / 0.0150077685,
                                 1.0 / 0.0142200657};
  net_param_.model_config.mean = {1.7922625 / 0.0145984266,
                                  1.7465649 / 0.0150077685,
                                  1.4802198 / 0.0142200657};
}

Clip_Image::~Clip_Image() {}

int Clip_Image::onModelOpened() {
  if (net_->getOutputNames().size() != 1) {
    LOGE("Clip_Image only expected 1 output branch!\n");
    return -1;
  }

  return 0;
}

int32_t Clip_Image::outputParse(
    const std::vector<std::shared_ptr<BaseImage>> &images,
    std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) {
  std::string input_tensor_name = net_->getInputNames()[0];
  TensorInfo input_tensor = net_->getTensorInfo(input_tensor_name);
  uint32_t input_width = input_tensor.shape[3];
  uint32_t input_height = input_tensor.shape[2];
  float input_width_f = float(input_width);
  float input_height_f = float(input_height);
  LOGI("outputParse,batch size:%d,input shape:%d,%d,%d,%d", images.size(),
       input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2],
       input_tensor.shape[3]);

  const auto &output_layers = net_->getOutputNames();
  size_t num_output = output_layers.size();
  TensorInfo output_info = net_->getTensorInfo(output_layers[0]);
  std::shared_ptr<BaseTensor> output_tensor =
      net_->getOutputTensor(output_layers[0]);
  auto output_shape = output_info.shape;
  int batch_elem_num = output_shape[3] * output_shape[2] * output_shape[1];

  for (uint32_t b = 0; b < (uint32_t)input_tensor.shape[0]; b++) {
    std::shared_ptr<ModelFeatureInfo> feature_meta =
        std::make_shared<ModelFeatureInfo>();
    feature_meta->embedding_num = batch_elem_num;
    feature_meta->embedding_type = TDLDataType::FP32;
    feature_meta->embedding = (uint8_t *)malloc(batch_elem_num * sizeof(float));

    float *feature = (float *)feature_meta->embedding;
    if (output_info.data_type == TDLDataType::INT8) {
      parse_feature_info<int8_t>(output_tensor->getBatchPtr<int8_t>(b),
                                 batch_elem_num, output_info.qscale, feature);
    } else if (output_info.data_type == TDLDataType::UINT8) {
      parse_feature_info<uint8_t>(output_tensor->getBatchPtr<uint8_t>(b),
                                  batch_elem_num, output_info.qscale, feature);
    } else if (output_info.data_type == TDLDataType::FP32) {
      parse_feature_info<float>(output_tensor->getBatchPtr<float>(b),
                                batch_elem_num, output_info.qscale, feature);
    } else {
      LOGE("not supported data type:%d", (int)output_info.data_type);
    }

    out_datas.push_back(feature_meta);
  }
  return 0;
}
