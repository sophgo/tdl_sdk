#include "image_classification/rgb_image_classification.hpp"

#include <numeric>

#include "utils/tdl_log.hpp"
#define topK 2

std::vector<int> top_indices(std::vector<float> &vec, int topk) {
  std::vector<int> topKIndex;

  std::vector<size_t> vec_index(vec.size());
  std::iota(vec_index.begin(), vec_index.end(), 0);

  std::sort(vec_index.begin(), vec_index.end(),
            [&vec](size_t index_1, size_t index_2) {
              return vec[index_1] > vec[index_2];
            });

  int k_num = std::min<int>(vec.size(), topk);

  for (int i = 0; i < k_num; ++i) {
    topKIndex.emplace_back(vec_index[i]);
  }

  return topKIndex;
}

template <typename T>
void parse_output(T *ptr_out, const int num_cls, float qscale,
                  std::shared_ptr<ModelClassificationInfo> cls_meta) {
  std::vector<float> scores;
  for (int i = 0; i < num_cls; i++) {
    scores.push_back(ptr_out[i] * qscale);
  }
  int max_top = std::min<int>(num_cls, topK);
  std::vector<int> topKIndex = top_indices(scores, max_top);

  for (int i = 0; i < max_top; i++) {
    cls_meta->topk_class_ids.push_back(topKIndex[i]);
    cls_meta->topk_scores.push_back(scores[topKIndex[i]]);
  }

  float max_score = *std::max_element(cls_meta->topk_scores.begin(),
                                      cls_meta->topk_scores.end());
  float sum = 0.0f;
  std::vector<float> softmax_scores;
  softmax_scores.reserve(cls_meta->topk_scores.size());

  for (const auto &score : cls_meta->topk_scores) {
    float exp_score = std::exp(score - max_score);
    softmax_scores.push_back(exp_score);
    sum += exp_score;
  }
  for (size_t i = 0; i < softmax_scores.size(); ++i) {
    cls_meta->topk_scores[i] = softmax_scores[i] / sum;
  }
}

RgbImageClassification::RgbImageClassification() : BaseModel() {
  float mean[3] = {0, 0, 0};
  float std[3] = {255, 255, 255};

  for (int i = 0; i < 3; i++) {
    net_param_.pre_params.mean[i] = mean[i] / std[i];
    net_param_.pre_params.scale[i] = 1.0 / std[i];
  }

  net_param_.pre_params.dstImageFormat = ImageFormat::RGB_PLANAR;

  net_param_.pre_params.keepAspectRatio = true;
}

int RgbImageClassification::onModelOpened() {
  if (net_->getOutputNames().size() != 1) {
    LOGE("ImageClassification only expected 1 output branch!\n");
    return -1;
  }

  return 0;
}

RgbImageClassification::~RgbImageClassification() {}

int32_t RgbImageClassification::outputParse(
    const std::vector<std::shared_ptr<BaseImage>> &images,
    std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) {
  std::string output_name = net_->getOutputNames()[0];
  TensorInfo oinfo = net_->getTensorInfo(output_name);

  std::shared_ptr<BaseTensor> output_tensor =
      net_->getOutputTensor(output_name);

  for (size_t b = 0; b < images.size(); b++) {
    std::shared_ptr<ModelClassificationInfo> cls_meta =
        std::make_shared<ModelClassificationInfo>();
    if (oinfo.data_type == TDLDataType::INT8) {
      parse_output<int8_t>(output_tensor->getBatchPtr<int8_t>(b),
                           oinfo.tensor_elem, oinfo.qscale, cls_meta);
    } else if (oinfo.data_type == TDLDataType::UINT8) {
      parse_output<uint8_t>(output_tensor->getBatchPtr<uint8_t>(b),
                            oinfo.tensor_elem, oinfo.qscale, cls_meta);
    } else if (oinfo.data_type == TDLDataType::FP32) {
      parse_output<float>(output_tensor->getBatchPtr<float>(b),
                          oinfo.tensor_elem, oinfo.qscale, cls_meta);
    } else {
      LOGE("unsupported data type: %d", oinfo.data_type);
    }
    out_datas.push_back(cls_meta);
  }
}