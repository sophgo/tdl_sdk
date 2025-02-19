#include "feature_extract/feature_extraction.hpp"

#include "core/cvi_tdl_types_mem_internal.h"
#include "cvi_tdl_log.hpp"
#include "utils/detection_helper.hpp"

template <typename T>
void parse_feature_info(T *data, int num_elem, float qscale,
                        std::vector<float> &features) {
  for (int i = 0; i < num_elem; i++) {
    features.push_back(data[i] * qscale);
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
    std::vector<void *> &out_datas) {
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
    std::vector<float> feature;
    if (output_info.data_type == ImagePixDataType::INT8) {
      parse_feature_info<int8_t>(output_tensor->getBatchPtr<int8_t>(b),
                                 batch_elem_num, output_info.qscale, feature);
    } else if (output_info.data_type == ImagePixDataType::UINT8) {
      parse_feature_info<uint8_t>(output_tensor->getBatchPtr<uint8_t>(b),
                                  batch_elem_num, output_info.qscale, feature);
    } else if (output_info.data_type == ImagePixDataType::FP32) {
      parse_feature_info<float>(output_tensor->getBatchPtr<float>(b),
                                batch_elem_num, output_info.qscale, feature);
    } else {
      LOGE("not supported data type:%d", (int)output_info.data_type);
    }
    cvtdl_feature_t *feature_meta =
        (cvtdl_feature_t *)malloc(sizeof(cvtdl_feature_t));
    feature_meta->size = feature.size() * sizeof(float);
    feature_meta->type = TYPE_FLOAT;
    feature_meta->ptr = (int8_t *)malloc(feature_meta->size);
    memcpy(feature_meta->ptr, feature.data(), feature_meta->size);
    out_datas.push_back(feature_meta);
  }
  return 0;
}