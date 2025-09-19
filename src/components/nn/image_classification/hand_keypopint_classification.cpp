#include "image_classification/hand_keypopint_classification.hpp"

#include <numeric>

#include "utils/tdl_log.hpp"
#define topK 2

HandKeypointClassification::HandKeypointClassification() {}

HandKeypointClassification::~HandKeypointClassification() {}

int HandKeypointClassification::onModelOpened() {
  if (net_->getOutputNames().size() != 1) {
    LOGE("HandKeypointClassification only expected 1 output branch!\n");
    return -1;
  }

  return 0;
}

int32_t HandKeypointClassification::inference(
    const std::vector<std::shared_ptr<BaseImage>> &images,
    std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas,
    const std::map<std::string, float> &parameters) {
  for (auto &image : images) {
    std::string input_layer = net_->getInputNames()[0];

    const TensorInfo &tinfo = net_->getTensorInfo(input_layer);
    float *temp_buffer =
        reinterpret_cast<float *>(image->getVirtualAddress()[0]);

#if defined(__CV181X__) || defined(__CMODEL_CV181X__)

    int8_t *input_ptr = (int8_t *)tinfo.sys_mem;
    for (int i = 0; i < 42; i++) {
      float temp_float = tinfo.qscale * temp_buffer[i];
      if (temp_float < -128)
        input_ptr[i] = -128;
      else if (temp_float > 127)
        input_ptr[i] = 128;
      else
        input_ptr[i] = (int8_t)std::round(temp_float);
    }
#else

    uint8_t *input_ptr = (uint8_t *)tinfo.sys_mem;
    for (int i = 0; i < 42; i++) {
      float temp_float = tinfo.qscale * temp_buffer[i];
      if (temp_float < 0)
        input_ptr[i] = 0;
      else if (temp_float > 255)
        input_ptr[i] = 255;
      else
        input_ptr[i] = (uint8_t)std::round(temp_float);
    }

#endif

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

int32_t HandKeypointClassification::outputParse(
    const std::vector<std::shared_ptr<BaseImage>> &images,
    std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) {
  std::string output_name = net_->getOutputNames()[0];
  TensorInfo oinfo = net_->getTensorInfo(output_name);

  std::shared_ptr<BaseTensor> output_tensor =
      net_->getOutputTensor(output_name);

  int num_cls = std::max(
      {oinfo.shape[0], oinfo.shape[1], oinfo.shape[2], oinfo.shape[3]});

  for (size_t b = 0; b < images.size(); b++) {
    std::shared_ptr<ModelClassificationInfo> cls_meta =
        std::make_shared<ModelClassificationInfo>();

    int max_index = -1;
    float max_score = -1.0f;  // 用float存储分数，便于统一处理

    if (oinfo.data_type == TDLDataType::INT8) {
      std::cout << "INT8" << std::endl;
      int8_t *out_data = output_tensor->getBatchPtr<int8_t>(b);
      max_score = static_cast<float>(out_data[0]);
      max_index = 0;
      for (int k = 1; k < num_cls; k++) {
        if (out_data[k] > out_data[max_index]) {
          max_score = static_cast<float>(out_data[k]);
          max_index = k;
        }
      }
    } else if (oinfo.data_type == TDLDataType::UINT8) {
      uint8_t *out_data = output_tensor->getBatchPtr<uint8_t>(b);
      max_score = static_cast<float>(out_data[0]);
      max_index = 0;
      for (int k = 1; k < num_cls; k++) {
        if (out_data[k] > out_data[max_index]) {
          max_score = static_cast<float>(out_data[k]);
          max_index = k;
        }
      }
    } else if (oinfo.data_type == TDLDataType::FP32) {
      float *out_data = output_tensor->getBatchPtr<float>(b);
      max_score = out_data[0];
      max_index = 0;
      for (int k = 1; k < num_cls; k++) {
        if (out_data[k] > max_score) {
          max_score = out_data[k];
          max_index = k;
        }
      }
    } else {
      LOGE("unsupported data type:%d\n", static_cast<int>(oinfo.data_type));
      return -1;
    }

    cls_meta->topk_class_ids.push_back(max_index);
    cls_meta->topk_scores.push_back(max_score);

    out_datas.push_back(cls_meta);
  }
  return 0;
}
