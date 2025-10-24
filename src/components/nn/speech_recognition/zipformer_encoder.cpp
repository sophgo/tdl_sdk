#include "speech_recognition/zipformer_encoder.hpp"
#include <fstream>
#include <numeric>
#include "utils/tdl_log.hpp"

#define ZIPFORMER_SAMPLE_RATE 16000
#define ZIPFORMER_FEATURE_SIZE 512
#define ZIPFORMER_TOKENS_SIZE 6257

static void replace_substr(std::string &str, const std::string &from,
                           const std::string &to) {
  if (from.empty()) return;  // Prevent infinite loop if 'from' is empty
  size_t start_pos = 0;
  while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length();  // Advance position by length of the replacement
  }
}

ZipformerEncoder::ZipformerEncoder() {
  knf::FbankOptions fbank_opts;
  fbank_opts.frame_opts.dither = 0;
  fbank_opts.frame_opts.snip_edges = false;
  fbank_opts.frame_opts.samp_freq = ZIPFORMER_SAMPLE_RATE;
  fbank_opts.mel_opts.num_bins = 80;
  fbank_opts.mel_opts.high_freq = -400;

  fbank_extractor_ = new knf::OnlineFbank(fbank_opts);

  decoder_input_data_ = ImageFactory::createImage(2, 1, ImageFormat::GRAY,
                                                  TDLDataType::INT32, true);
  joiner_input_data_ =
      ImageFactory::createImage(ZIPFORMER_FEATURE_SIZE * 2, 1,
                                ImageFormat::GRAY, TDLDataType::FP32, true);
}

ZipformerEncoder::~ZipformerEncoder() {
  free(int32_chached_inputs_);
  int32_chached_inputs_ = nullptr;
  free(float_chached_inputs_);
  float_chached_inputs_ = nullptr;
  delete fbank_extractor_;
}

int32_t ZipformerEncoder::setupNetwork(NetParam &net_param) {
  net_ = NetFactory::createNet(net_param, net_param.platform);
  int32_t ret = net_->setup();
  if (ret != 0) {
    std::cout << "Net setup failed" << std::endl;
    assert(false);
    return ret;
  }
  return 0;
}

int32_t ZipformerEncoder::setTokensPath(std::string tokens_path) {
  std::ifstream tokens_file(tokens_path);
  if (!tokens_file.is_open()) {
    LOGE("ZipformerEncoder setTokensPath failed, tokens_path: %s\n",
         tokens_path.c_str());
    return -1;
  }
  std::string line;
  while (std::getline(tokens_file, line)) {
    // 截取第一个空格前的子串
    size_t sp_pos = line.find(' ');
    if (sp_pos != std::string::npos) {
      line = line.substr(0, sp_pos);
    }
    tokens_.push_back(line);
  }
  tokens_file.close();

  if (tokens_.size() != ZIPFORMER_TOKENS_SIZE) {
    LOGE(
        "ZipformerEncoder setTokensPath failed, tokens_path: %s, tokens size "
        "is %d, expect %d\n",
        tokens_path.c_str(), tokens_.size(), ZIPFORMER_TOKENS_SIZE);
    return -1;
  }
  return 0;
}

int32_t ZipformerEncoder::setModel(std::shared_ptr<BaseModel> decoder_model,
                                   std::shared_ptr<BaseModel> joiner_model) {
  decoder_model_ = decoder_model;
  joiner_model_ = joiner_model;
  return 0;
}

int32_t ZipformerEncoder::onModelOpened() {
  std::vector<std::string> input_layers = net_->getInputNames();

  TDLDataType data_type;

  float_cached_offset_.clear();
  int32_cached_offset_.clear();
  for (int i = 0; i < input_layers.size(); i++) {
    const TensorInfo &tinfo = net_->getTensorInfo(input_layers[i]);

    LOGI("input[%d]:%s, shape:[%d,%d,%d,%d]\n", i, input_layers[i].c_str(),
         tinfo.shape[0], tinfo.shape[1], tinfo.shape[2], tinfo.shape[3]);
    if (i == 0) {  // sound feature input
      float_cached_offset_.push_back(0);
      int32_cached_offset_.push_back(0);
      data_type = tinfo.data_type;
      segment_size_ = tinfo.shape[1];
      num_mel_ = tinfo.shape[2];

      if (segment_size_ == 103) {
        frame_offset_ = 96;  // model-m
      } else {
        frame_offset_ = 64;  // model-s
      }
      LOGI("ZipformerEncoder segment_size_: %d, num_mel_: %d\n", segment_size_,
           num_mel_);
      continue;
    }
    int cached_size =
        tinfo.shape[0] * tinfo.shape[1] * tinfo.shape[2] * tinfo.shape[3];

    if (i <= 5) {
      int32_cached_offset_.push_back(
          int32_cached_offset_.back() +
          cached_size);  // int32_cached_offset_ size = input int32 cached num +
                         // 1
    } else {
      float_cached_offset_.push_back(
          float_cached_offset_.back() +
          cached_size);  // float_cached_offset_ size = input float cached num +
                         // 1
    }
  }

  if (data_type != TDLDataType::FP32) {
    LOGE("ZipformerEncoder only support float32 input now!\n");
    return -1;
  } else {
    int32_chached_inputs_ =
        (int32_t *)malloc(int32_cached_offset_.back() * sizeof(int32_t));
    memset(int32_chached_inputs_, 0,
           int32_cached_offset_.back() * sizeof(int32_t));

    float_chached_inputs_ =
        (float *)malloc(float_cached_offset_.back() * sizeof(float));
    memset(float_chached_inputs_, 0,
           float_cached_offset_.back() * sizeof(float));
  }

  return 0;
}

int32_t ZipformerEncoder::prepareInput() {
  std::vector<std::string> input_layers = net_->getInputNames();

  for (int i = 0; i < input_layers.size(); i++) {
    const TensorInfo &tinfo = net_->getTensorInfo(input_layers[i]);

    if (i == 0) {
      float *input_ptr = (float *)tinfo.sys_mem;

      for (int j = 0; j < segment_size_; j++) {
        const float *frame =
            fbank_extractor_->GetFrame(num_processed_frames_ + j);
        memcpy(input_ptr + j * num_mel_, frame, num_mel_ * sizeof(float));
      }
      num_processed_frames_ += frame_offset_;

    } else if (i <= 5) {
      int32_t *input_ptr = (int32_t *)tinfo.sys_mem;
      memcpy(input_ptr, int32_chached_inputs_ + int32_cached_offset_[i - 1],
             (int32_cached_offset_[i] - int32_cached_offset_[i - 1]) *
                 sizeof(int32_t));
    } else {
      float *input_ptr = (float *)tinfo.sys_mem;
      memcpy(input_ptr, float_chached_inputs_ + float_cached_offset_[i - 6],
             (float_cached_offset_[i - 5] - float_cached_offset_[i - 6]) *
                 sizeof(float));
    }
  }

  return 0;
}

int32_t ZipformerEncoder::greedy_search(
    float *data_ptr, std::shared_ptr<ModelOutputInfo> &out_data) {
  std::shared_ptr<ModelASRInfo> asr_meta =
      std::static_pointer_cast<ModelASRInfo>(out_data);

  if (!init_decoder_output_) {
    int32_t *decoder_input_buffer =
        (int32_t *)decoder_input_data_->getVirtualAddress()[0];

    memset(decoder_input_buffer, 0,
           decoder_input_data_->getWidth() * decoder_input_data_->getHeight() *
               sizeof(int32_t));
    decoder_model_->inference(decoder_input_data_, out_data);
    init_decoder_output_ = true;
  }

  for (int i = 0; i < feature_num_; i++) {
    float *joiner_input_buffer =
        (float *)joiner_input_data_->getVirtualAddress()[0];
    memcpy(joiner_input_buffer, data_ptr + i * ZIPFORMER_FEATURE_SIZE,
           ZIPFORMER_FEATURE_SIZE * sizeof(float));
    memcpy(joiner_input_buffer + ZIPFORMER_FEATURE_SIZE,
           asr_meta->decoder_feature, ZIPFORMER_FEATURE_SIZE * sizeof(float));

    joiner_model_->inference(joiner_input_data_, out_data);

    if (asr_meta->pred_index != 0) {
      int32_t *decoder_input_buffer =
          (int32_t *)decoder_input_data_->getVirtualAddress()[0];

      decoder_input_buffer[0] = hyp_.back();
      decoder_input_buffer[1] = asr_meta->pred_index;
      hyp_.push_back(asr_meta->pred_index);

      decoder_model_->inference(decoder_input_data_, out_data);
    }
  }
  return 0;
}

int32_t ZipformerEncoder::inference(
    const std::shared_ptr<BaseImage> &image,
    std::shared_ptr<ModelOutputInfo> &out_data,
    const std::map<std::string, float> &parameters) {
  std::shared_ptr<ModelASRInfo> asr_meta =
      std::static_pointer_cast<ModelASRInfo>(out_data);

  int img_width = image->getWidth() / 2;  // unit: 16 bits
  int img_height = image->getHeight();
  short *temp_buffer = (short *)image->getVirtualAddress()[0];

  LOGI("input data size: %d", img_width * img_height);

  std::vector<float> float_buffer(img_width * img_height);
  for (int i = 0; i < float_buffer.size(); i++) {
    float_buffer[i] = (float)(temp_buffer[i]) / 32768.0;
  }

  fbank_extractor_->AcceptWaveform(ZIPFORMER_SAMPLE_RATE, float_buffer.data(),
                                   float_buffer.size());

  int32_t num_frames = fbank_extractor_->NumFramesReady();

  int used_count = 0;

  while (num_frames - num_processed_frames_ >= segment_size_) {
    prepareInput();

    net_->updateInputTensors();
    net_->forward();
    net_->updateOutputTensors();
    outputParse(image, out_data);

    std::vector<std::string> output_layers = net_->getOutputNames();
    std::shared_ptr<BaseTensor> output_tensor =
        net_->getOutputTensor(output_layers[0]);
    float *output_ptr = output_tensor->getBatchPtr<float>(0);

    greedy_search(output_ptr, out_data);

    used_count++;
  }

  if (asr_meta->text_info) {
    delete[] asr_meta->text_info;
    asr_meta->text_info = nullptr;
    asr_meta->text_length = 0;
  }

  std::string text;
  for (int i = 1; i < hyp_.size(); i++) {
    std::string tmp = tokens_[hyp_[i]];
    replace_substr(tmp, "▁", " ");
    text += tmp;
  }

  if (!text.empty()) {
    // 多申请 1 字节给 '\0'
    asr_meta->text_info = new char[text.size() + 1];
    strcpy(asr_meta->text_info, text.c_str());
    asr_meta->text_length = text.size();  // 不含 '\0' 的长度
  }

  hyp_ = {hyp_.back()};

  fbank_extractor_->Pop(used_count * frame_offset_);  // release used frames

  if (asr_meta->input_finished) {  // for evaluating next sound file

    hyp_ = {0};
    init_decoder_output_ = false;

    memset(int32_chached_inputs_, 0,
           int32_cached_offset_.back() * sizeof(int32_t));
    memset(float_chached_inputs_, 0,
           float_cached_offset_.back() * sizeof(float));

    fbank_extractor_->Pop(num_frames -
                          num_processed_frames_);  // release remaining frames
    num_processed_frames_ = num_frames;
  }

  return 0;
}

int32_t ZipformerEncoder::outputParse(
    const std::shared_ptr<BaseImage> &image,
    std::shared_ptr<ModelOutputInfo> &out_data) {
  std::vector<std::string> output_layers = net_->getOutputNames();

  for (int i = 0; i < output_layers.size(); i++) {
    const TensorInfo &tinfo = net_->getTensorInfo(output_layers[i]);
    int data_size =
        tinfo.shape[0] * tinfo.shape[1] * tinfo.shape[2] * tinfo.shape[3];

    std::shared_ptr<BaseTensor> output_tensor =
        net_->getOutputTensor(output_layers[i]);
    float *output_ptr = output_tensor->getBatchPtr<float>(0);

    if (i == 0) {
      feature_num_ = tinfo.shape[1];
      LOGI("feature_num_: %d\n", feature_num_);
    } else if (i <= 5) {  // update int32_t cached input

      if (data_size != int32_cached_offset_[i] - int32_cached_offset_[i - 1]) {
        LOGE(
            "ZipformerEncoder output size %d not equal to int32_cached_offset_ "
            "size %d\n",
            data_size, int32_cached_offset_[i] - int32_cached_offset_[i - 1]);
        return -1;
      }

      for (int j = 0; j < data_size; j++) {
        int32_chached_inputs_[(i - 1) * data_size + j] =
            (int32_t)(output_ptr[j]);  // output data type != input data type
                                       // cause of tpu-milr bug
      }
    } else {  // update float cached input

      if (data_size !=
          float_cached_offset_[i - 5] - float_cached_offset_[i - 6]) {
        LOGE(
            "ZipformerEncoder output size %d not equal to float_cached_offset_ "
            "size %d\n",
            data_size,
            float_cached_offset_[i - 5] - float_cached_offset_[i - 6]);
        return -1;
      }

      memcpy(float_chached_inputs_ + float_cached_offset_[i - 6], output_ptr,
             (float_cached_offset_[i - 5] - float_cached_offset_[i - 6]) *
                 sizeof(float));
    }
  }

  return 0;
}
