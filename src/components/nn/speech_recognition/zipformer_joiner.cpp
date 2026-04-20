#include "speech_recognition/zipformer_joiner.hpp"
#include <numeric>
#include "utils/tdl_log.hpp"

ZipformerJoiner::ZipformerJoiner() {}

ZipformerJoiner::~ZipformerJoiner() {}

int32_t ZipformerJoiner::setupNetwork(NetParam &net_param) {
  net_ = NetFactory::createNet(net_param, net_param.platform);
  int32_t ret = net_->setup();
  if (ret != 0) {
    std::cout << "Net setup failed" << std::endl;
    assert(false);
    return ret;
  }
  return 0;
}

int32_t ZipformerJoiner::onModelOpened() { return 0; }

int32_t ZipformerJoiner::inference(
    const std::shared_ptr<BaseImage> &image,
    std::shared_ptr<ModelOutputInfo> &out_data,
    const std::map<std::string, float> &parameters) {
  int img_width = image->getWidth();
  int img_height = image->getHeight();
  float *temp_buffer = (float *)image->getVirtualAddress()[0];

  std::vector<std::string> input_layers = net_->getInputNames();
  for (int i = 0; i < input_layers.size(); i++) {
    const TensorInfo &tinfo = net_->getTensorInfo(input_layers[i]);
    float *input_ptr = (float *)tinfo.sys_mem;

    if (i == 0) {
      feature_size_ =
          tinfo.shape[0] * tinfo.shape[1] * tinfo.shape[2] * tinfo.shape[3];
    }

    if (feature_size_ * 2 != img_width * img_height) {
      LOGE(
          "ZipformerJoiner input feature size %d not match half of image size "
          "%d * %d\n",
          feature_size_, img_width, img_height);
      return -1;
    }

    memcpy(input_ptr, temp_buffer + i * feature_size_,
           feature_size_ * sizeof(float));
  }

  net_->updateInputTensors();
  net_->forward();
  net_->updateOutputTensors();

  outputParse(image, out_data);

  return 0;
}

int32_t ZipformerJoiner::outputParse(
    const std::shared_ptr<BaseImage> &image,
    std::shared_ptr<ModelOutputInfo> &out_data) {
  std::shared_ptr<ModelASRInfo> asr_meta =
      std::static_pointer_cast<ModelASRInfo>(out_data);

  std::vector<std::string> output_layers = net_->getOutputNames();

  const TensorInfo &tinfo = net_->getTensorInfo(output_layers[0]);
  int data_size =
      tinfo.shape[0] * tinfo.shape[1] * tinfo.shape[2] * tinfo.shape[3];

  std::shared_ptr<BaseTensor> output_tensor =
      net_->getOutputTensor(output_layers[0]);
  float *output_ptr = output_tensor->getBatchPtr<float>(0);

  auto it = std::max_element(output_ptr, output_ptr + data_size);
  asr_meta->pred_index = static_cast<int32_t>(it - output_ptr);

  return 0;
}
