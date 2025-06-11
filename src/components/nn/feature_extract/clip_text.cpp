#include "feature_extract/clip_text.hpp"

#include "utils/tdl_log.hpp"

template <typename T>
void parse_feature_info(T *data, int num_elem, float qscale, float *features) {
  for (int i = 0; i < num_elem; i++) {
    features[i] = data[i] * qscale;
  }
}

Clip_Text::Clip_Text() {
  net_param_.model_config.std = {255.0, 255.0, 255.0};
  net_param_.model_config.mean = {0.0, 0.0, 0.0};
}

Clip_Text::~Clip_Text() {}

int Clip_Text::onModelOpened() {
  if (net_->getOutputNames().size() != 1) {
    LOGE("Clip_Tex only expected 1 output branch!\n");
    return -1;
  }

  return 0;
}
int32_t Clip_Text::inference(
    const std::vector<std::shared_ptr<BaseImage>> &images,
    std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas,
    const std::map<std::string, float> &parameters) {
  for (auto &image : images) {
    // int32_t *temp_buffer = (int32_t *)image->getVirtualAddress()[0];
    int32_t *temp_buffer =
        reinterpret_cast<int32_t *>(image->getVirtualAddress()[0]);
    std::string input_layer = net_->getInputNames()[0];

    const TensorInfo &tinfo = net_->getTensorInfo(input_layer);
    int32_t *input_ptr = (int32_t *)tinfo.sys_mem;

    memcpy(input_ptr, temp_buffer, 77 * sizeof(int32_t));

    net_->updateInputTensors();
    net_->forward();
    net_->updateOutputTensors();
    std::vector<std::shared_ptr<ModelOutputInfo>> batch_results;

    std::vector<std::shared_ptr<BaseImage>> batch_images = {image};
    outputParse(batch_images, batch_results);

    out_datas.insert(out_datas.end(), batch_results.begin(),
                     batch_results.end());
  }

  return 0;
}

int32_t Clip_Text::outputParse(
    const std::vector<std::shared_ptr<BaseImage>> &images,
    std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) {
  std::string input_tensor_name = net_->getInputNames()[0];
  TensorInfo input_tensor = net_->getTensorInfo(input_tensor_name);
  LOGI("outputParse,batch size:%d,input shape:%d,%d", images.size(),
       input_tensor.shape[0], input_tensor.shape[1]);

  const auto &output_layers = net_->getOutputNames();
  // size_t num_output = output_layers.size();
  TensorInfo output_info = net_->getTensorInfo(output_layers[0]);
  std::shared_ptr<BaseTensor> output_tensor =
      net_->getOutputTensor(output_layers[0]);
  auto output_shape = output_info.shape;
  int batch_elem_num = output_shape[1] * output_shape[2] * output_shape[3];

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
