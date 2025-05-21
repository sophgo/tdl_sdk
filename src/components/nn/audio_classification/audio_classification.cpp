#include "audio_classification/audio_classification.hpp"
#include <numeric>
#include "utils/tdl_log.hpp"
#define SCALE_FACTOR_FOR_INT16 32768.0

AudioClassification::AudioClassification()
    : AudioClassification(std::make_pair(256, 0)) {}

AudioClassification::AudioClassification(std::pair<int, int> sound_pair) {
  win_len_ = 1024;
  num_fft_ = 1024;
  hop_len_ = sound_pair.first;
  sample_rate_ = 16000;
  time_len_ = 3;  // 3 second
  num_mel_ = 40;
  fmin_ = 0;
  fmax_ = sample_rate_ / 2;
  fix_ = sound_pair.second;
}

AudioClassification::~AudioClassification() { delete mp_extractor_; }

int32_t AudioClassification::setupNetwork(NetParam &net_param) {
  net_ = NetFactory::createNet(net_param, net_param.platform);
  int32_t ret = net_->setup();
  if (ret != 0) {
    std::cout << "Net setup failed" << std::endl;
    assert(false);
    return ret;
  }
  return 0;
}

int32_t AudioClassification::onModelOpened() {
  std::string input_layer = net_->getInputNames()[0];
  const TensorInfo &tinfo = net_->getTensorInfo(input_layer);
  int32_t image_width = tinfo.shape[2];
  int32_t image_height = tinfo.shape[3];
  bool htk = false;
  auto &parameters = net_param_.model_config.custom_config_i;

  if (parameters.find("hop_len") != parameters.end()) {
    hop_len_ = static_cast<int>(parameters.at("hop_len"));
    LOGI("hop_len:%d", hop_len_);
  }

  if (parameters.find("fix") != parameters.end()) {
    fix_ = static_cast<int>(parameters.at("fix"));
    LOGI("fix:%d", fix_);
  }
  if (image_width == 251 && hop_len_ == 128) {  // sr16k * 2s, hop_len = 128
    sample_rate_ = 16000;
    time_len_ = 2;
  } else if (image_width == 63 || hop_len_ == 128) {  // sr8k * 2s
    sample_rate_ = 8000;
    time_len_ = 2;
  } else if (image_width == 94) {  // sr8k * 3s
    sample_rate_ = 8000;
    time_len_ = 3;
  } else if (image_width == 126) {  // sr16k * 2s
    sample_rate_ = 16000;
    time_len_ = 2;
  } else if (image_width == 188) {  // sr16k * 3s
    sample_rate_ = 16000;
    time_len_ = 3;
  }

  fmax_ = sample_rate_ / 2;
  int num_frames = time_len_ * sample_rate_;

  mp_extractor_ = new melspec::MelFeatureExtract(num_frames, sample_rate_,
                                                 num_fft_, hop_len_, num_mel_,
                                                 fmin_, fmax_, "reflect", htk);
  LOGI("model input width:%d,height:%d,sample_rate:%d,time_len:%d\n",
       image_width, image_height, sample_rate_, time_len_);
  return 0;
}

int32_t AudioClassification::inference(
    const std::vector<std::shared_ptr<BaseImage>> &images,
    std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas,
    const std::map<std::string, float> &parameters) {
  for (auto &image : images) {
    int img_width = image->getWidth() / 2;  // unit: 16 bits
    int img_height = image->getHeight();

    // save audio to image array
    short *temp_buffer = (short *)image->getVirtualAddress()[0];
    normalizeSound(temp_buffer, img_width * img_height);
    mp_extractor_->update_data(temp_buffer, img_width * img_height);

    std::string input_layer = net_->getInputNames()[0];

    const TensorInfo &tinfo = net_->getTensorInfo(input_layer);
    int8_t *input_ptr = (int8_t *)tinfo.sys_mem;

    if (parameters.count("pack_idx") && parameters.count("pack_len")) {
      mp_extractor_->melspectrogram_pack_optimize(
          temp_buffer, img_width * img_height, (int)parameters.at("pack_len"),
          (int)parameters.at("pack_idx"), input_ptr, int(tinfo.tensor_elem),
          tinfo.qscale, fix_);
    } else {
      mp_extractor_->melspectrogram_optimze(temp_buffer, img_width * img_height,
                                            input_ptr, int(tinfo.tensor_elem),
                                            tinfo.qscale, fix_);
    }

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

int32_t AudioClassification::outputParse(
    const std::vector<std::shared_ptr<BaseImage>> &images,
    std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) {
  std::string output_layer = net_->getOutputNames()[0];
  const TensorInfo &tinfo = net_->getTensorInfo(output_layer);
  float *output_ptr = (float *)tinfo.sys_mem;
  float score;
  int32_t index = getTopK(output_ptr, tinfo.tensor_elem, &score);
  std::shared_ptr<ModelClassificationInfo> output_info =
      std::make_shared<ModelClassificationInfo>();
  output_info->topk_scores.push_back(score);
  output_info->topk_class_ids.push_back(index);

  if (type_mapping_.find(index) != type_mapping_.end()) {
    output_info->topk_object_types.push_back(type_mapping_[index]);
  }
  out_datas.push_back(output_info);
  return 0;
}

int32_t AudioClassification::getTopK(float *result, size_t count,
                                     float *score) {
  int idx = -1;
  float max_e = -10000;
  float cur_e;

  float sum_e = 0.;
  for (size_t i = 0; i < count; i++) {
    cur_e = std::exp(result[i]);
    if (i != 0 && cur_e > max_e) {
      max_e = cur_e;
      idx = i;
    }
    sum_e = float(sum_e) + float(cur_e);
    // std::cout << i << ": " << cur_e << "\t";
  }

  float max = max_e / sum_e;
  if (idx != 0 && max < model_threshold_) {
    idx = 0;
    *score = std::exp(result[0]) / sum_e;
  } else {
    *score = max;
  }
  return idx;
}

void AudioClassification::normalizeSound(short *audio_data, int n) {
  // std::cout << "before:" << audio_data[0];
  std::vector<double> audio_abs(n);
  for (int i = 0; i < n; i++) {
    audio_abs[i] = std::abs(static_cast<double>(audio_data[i]));
  }
  std::vector<double> top_data;
  std::make_heap(audio_abs.begin(), audio_abs.end());
  if (top_num <= 0) {
    std::cerr << "When top_num<=0, the volume adaptive algorithm will fail. "
                 "Current top_num="
              << top_num << std::endl;
    return;
  }
  for (int i = 0; i < top_num; i++) {
    top_data.push_back(audio_abs.front());
    std::pop_heap(audio_abs.begin(), audio_abs.end());
    audio_abs.pop_back();
  }
  double top_mean =
      std::accumulate(top_data.begin(), top_data.end(), 0.0) / top_num;
  if (top_mean == 0) {
    std::cout
        << "The average of the top data is zero, cannot scale the audio data."
        << std::endl;
  } else {
    double r = max_rate * SCALE_FACTOR_FOR_INT16 / double(top_mean);
    double tmp = 0;
    for (int i = 0; i < n; i++) {
      tmp = audio_data[i] * r;
      audio_data[i] = short(tmp);
    }
  }
  // std::cout << ", after:" << audio_data[0];
}
