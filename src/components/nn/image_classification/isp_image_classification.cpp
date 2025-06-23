#include "image_classification/isp_image_classification.hpp"
#include <numeric>
#include "utils/tdl_log.hpp"
#define TopK 5

static std::vector<int> top_indices(std::vector<float> &vec, int topk) {
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
  float max_score = *std::max_element(scores.begin(), scores.end());
  float sum = 0.0f;
  std::vector<float> softmax_scores;
  softmax_scores.reserve(scores.size());

  for (const auto &score : scores) {
    float exp_score = std::exp(score - max_score);
    softmax_scores.push_back(exp_score);
    sum += exp_score;
  }
  for (size_t i = 0; i < softmax_scores.size(); ++i) {
    scores[i] = softmax_scores[i] / sum;
  }

  int max_top = std::min<int>(num_cls, TopK);
  std::vector<int> topKIndex = top_indices(scores, max_top);

  for (int i = 0; i < max_top; i++) {
    cls_meta->topk_class_ids.push_back(topKIndex[i]);
    cls_meta->topk_scores.push_back(scores[topKIndex[i]]);
  }
}

IspImageClassification::IspImageClassification() : BaseModel() {
  net_param_.model_config.mean = {123.675, 116.28, 103.52};
  net_param_.model_config.std = {58.395, 57.12, 57.375};
  net_param_.model_config.rgb_order = "gray";
  keep_aspect_ratio_ = true;
}

IspImageClassification::~IspImageClassification() {}

int IspImageClassification::onModelOpened() {
  // if (net_->getOutputNames().size() != 1) {
  //  LOGE("ImageClassification only expected 1 output branch!\n");
  //  return -1;
  //}

  return 0;
}

int32_t IspImageClassification::inference(
    const std::vector<std::shared_ptr<BaseImage>> &images,
    std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas,
    const std::map<std::string, float> &parameters) {
  if (images.empty()) {
    LOGE("Input images is empty");
    return -1;
  }

  float awb[3];  // rgain, ggain, bgain
  float ccm[9];  // rgb[3][3]
  float blc[1];

  if (parameters.count("awb[0]")) awb[0] = parameters.at("awb[0]");
  if (parameters.count("awb[1]")) awb[1] = parameters.at("awb[1]");
  if (parameters.count("awb[2]")) awb[2] = parameters.at("awb[2]");
  if (parameters.count("ccm[0]")) ccm[0] = parameters.at("ccm[0]");
  if (parameters.count("ccm[1]")) awb[1] = parameters.at("ccm[1]");
  if (parameters.count("ccm[2]")) awb[2] = parameters.at("ccm[2]");
  if (parameters.count("ccm[3]")) ccm[3] = parameters.at("ccm[3]");
  if (parameters.count("ccm[4]")) awb[4] = parameters.at("ccm[4]");
  if (parameters.count("ccm[5]")) awb[5] = parameters.at("ccm[5]");
  if (parameters.count("ccm[6]")) ccm[6] = parameters.at("ccm[6]");
  if (parameters.count("ccm[7]")) awb[7] = parameters.at("ccm[7]");
  if (parameters.count("ccm[8]")) awb[8] = parameters.at("ccm[8]");
  if (parameters.count("blc")) blc[0] = parameters.at("blc");

  size_t input_num = net_->getInputNames().size();
  std::string input_awb = net_->getInputNames()[1];
  const TensorInfo &tinfot_awb = net_->getTensorInfo(input_awb);
  int32_t *input_ptr = (int32_t *)tinfot_awb.sys_mem;
  memcpy(input_ptr, awb, 3 * sizeof(float));

  if (input_num == 3) {  // compatible with old models
    std::string input_blc = net_->getInputNames()[2];  // blc
    const TensorInfo &tinfot_blc = net_->getTensorInfo(input_blc);
    input_ptr = (int32_t *)tinfot_blc.sys_mem;
    memcpy(input_ptr, blc, sizeof(float));
  } else if (input_num == 4) {  // compatible with old models
    std::string input_ccm = net_->getInputNames()[2];  // ccm
    const TensorInfo &tinfot_ccm = net_->getTensorInfo(input_ccm);
    input_ptr = (int32_t *)tinfot_ccm.sys_mem;
    memcpy(input_ptr, ccm, 9 * sizeof(float));

    std::string input_blc = net_->getInputNames()[3];  // blc
    const TensorInfo &tinfot_blc = net_->getTensorInfo(input_blc);
    input_ptr = (int32_t *)tinfot_blc.sys_mem;
    memcpy(input_ptr, blc, sizeof(float));
  }

  model_timer_.TicToc("runstart");

  for (auto &image : images) {
    std::string input_layer_name = net_->getInputNames()[0];

    net_->getInputTensor(input_layer_name)->copyFromImage(image, 0);
    model_timer_.TicToc("preprocess");

    net_->updateInputTensors();
    net_->forward();
    model_timer_.TicToc("tpu");
    net_->updateOutputTensors();
    std::shared_ptr<ModelClassificationInfo> result =
        std::make_shared<ModelClassificationInfo>();

    outputParse(result);
    model_timer_.TicToc("post");

    out_datas.push_back(result);
  }

  return 0;
}

int32_t IspImageClassification::outputParse(
    std::shared_ptr<ModelClassificationInfo> &out_data) {
  std::string output_name = net_->getOutputNames()[1];
  TensorInfo oinfo = net_->getTensorInfo(output_name);

  std::shared_ptr<BaseTensor> output_tensor =
      net_->getOutputTensor(output_name);

  if (oinfo.data_type == TDLDataType::INT8) {
    parse_output<int8_t>(output_tensor->getBatchPtr<int8_t>(0),
                         oinfo.tensor_elem, oinfo.qscale, out_data);
  } else if (oinfo.data_type == TDLDataType::UINT8) {
    parse_output<uint8_t>(output_tensor->getBatchPtr<uint8_t>(0),
                          oinfo.tensor_elem, oinfo.qscale, out_data);
  } else if (oinfo.data_type == TDLDataType::FP32) {
    parse_output<float>(output_tensor->getBatchPtr<float>(0), oinfo.tensor_elem,
                        oinfo.qscale, out_data);
  } else {
    LOGE("unsupported data type: %d", oinfo.data_type);
  }

  return 0;
}
