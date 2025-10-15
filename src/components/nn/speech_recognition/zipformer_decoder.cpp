#include "speech_recognition/zipformer_decoder.hpp"
#include <numeric>
#include "utils/tdl_log.hpp"

ZipformerDecoder::ZipformerDecoder() {}

ZipformerDecoder::~ZipformerDecoder() {}

int32_t ZipformerDecoder::setupNetwork(NetParam &net_param) {
  net_ = NetFactory::createNet(net_param, net_param.platform);
  int32_t ret = net_->setup();
  if (ret != 0) {
    std::cout << "Net setup failed" << std::endl;
    assert(false);
    return ret;
  }
  return 0;
}

int32_t ZipformerDecoder::onModelOpened() { return 0; }

int32_t ZipformerDecoder::inference(
    const std::shared_ptr<BaseImage> &image,
    std::shared_ptr<ModelOutputInfo> &out_data,
    const std::map<std::string, float> &parameters) {
  std::shared_ptr<ModelASRInfo> asr_meta =
      std::static_pointer_cast<ModelASRInfo>(out_data);

  int img_width = image->getWidth();
  int img_height = image->getHeight();
  int32_t *temp_buffer = (int32_t *)image->getVirtualAddress()[0];

  std::vector<std::string> input_layers = net_->getInputNames();
  const TensorInfo &tinfo = net_->getTensorInfo(input_layers[0]);
  int input_size =
      tinfo.shape[0] * tinfo.shape[1] * tinfo.shape[2] * tinfo.shape[3];

  if (input_size != img_width * img_height) {
    LOGE("input size not match, expect %d, but get %d", input_size,
         img_width * img_height);
    return -1;
  }

  int32_t *input_ptr = (int32_t *)tinfo.sys_mem;

  memcpy(input_ptr, temp_buffer, input_size * sizeof(int32_t));

  net_->updateInputTensors();
  net_->forward();
  net_->updateOutputTensors();

  outputParse(image, out_data);

  return 0;
}

int32_t ZipformerDecoder::outputParse(
    const std::shared_ptr<BaseImage> &image,
    std::shared_ptr<ModelOutputInfo> &out_data) {
  std::shared_ptr<ModelASRInfo> asr_meta =
      std::static_pointer_cast<ModelASRInfo>(out_data);

  std::vector<std::string> output_layers = net_->getOutputNames();

  const TensorInfo &tinfo = net_->getTensorInfo(output_layers[0]);
  int data_size =
      tinfo.shape[0] * tinfo.shape[1] * tinfo.shape[2] * tinfo.shape[3];

  if (!asr_meta->decoder_feature) {
    asr_meta->decoder_feature = (float *)malloc(data_size * sizeof(float));
    asr_meta->decoder_feature_size = data_size;
  }

  std::shared_ptr<BaseTensor> output_tensor =
      net_->getOutputTensor(output_layers[0]);
  float *output_ptr = output_tensor->getBatchPtr<float>(0);

  memcpy(asr_meta->decoder_feature, output_ptr, data_size * sizeof(float));

  return 0;
}
