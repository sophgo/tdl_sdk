#include "feature_extract/feature_extraction.hpp"

#include "utils/detection_helper.hpp"
#include "utils/tdl_log.hpp"

template <typename T>
void parse_feature_info(T *data, int num_elem, float qscale, float *features) {
  for (int i = 0; i < num_elem; i++) {
    features[i] = data[i] * qscale;
  }
}

FeatureExtraction::FeatureExtraction() {
  net_param_.pre_params.scale[0] = 1.0;
  net_param_.pre_params.scale[1] = 1.0;
  net_param_.pre_params.scale[2] = 1.0;
  net_param_.pre_params.dstImageFormat = ImageFormat::RGB_PLANAR;
  net_param_.pre_params.keepAspectRatio = false;
}

FeatureExtraction::~FeatureExtraction() {}

int FeatureExtraction::onModelOpened() { return 0; }

int32_t FeatureExtraction::outputParse(
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