/*
 * aeroear_classification.cpp — AeroEar acoustic drone detection model
 * implementation
 *
 * Implements the full AeroEar pipeline: preprocessing + TPU encoder + CPU GRU
 * + detection threshold. Callers use the standard TDL_Classification API.
 * Input: 5s 48kHz int16 PCM. Output: 0=background, 1=drone.
 */

#include "aeroear_classification.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include "net/base_net.hpp"
#include "utils/tdl_log.hpp"

/* Number of long-window frames (1.0s) used for threshold detection. */
#define AEROEAR_DETECTION_WINDOW 30

/* Run all 153 GRU steps per batch. The 109 zero-padded front frames flush
 * old drone state; the 47 recent-audio tail frames set the new state.
 * This allows clearing within 1 batch (~6s) instead of 7 batches (~34s). */
#define AEROEAR_GRU_STEPS_PER_BATCH AEROEAR_TPU_T_LONG /* 153 */

/* Constructor: 5s @ 48kHz = 240000 samples. */
AeroEarClassification::AeroEarClassification() : BaseModel() {
  /* 5s @ 48kHz = 240000 samples */
  time_len_ = 5;
  sample_rate_ = AEROEAR_SR;
}

AeroEarClassification::~AeroEarClassification() = default;

/* Create network object. */
int32_t AeroEarClassification::setupNetwork(NetParam &net_param) {
  net_ = NetFactory::createNet(net_param, net_param.platform);
  int32_t ret = net_->setup();
  if (ret != 0) {
    LOGE("AeroEar: Net setup failed, ret=%d", ret);
    return ret;
  }
  return 0;
}

/* Post-model-load init: get I/O tensor info, init preprocessing context. */
int32_t AeroEarClassification::onModelOpened() {
  /* Validate input tensors */
  const auto &input_names = net_->getInputNames();
  if (input_names.size() < 2) {
    LOGE("AeroEar: expected 2 inputs (mel_short, mel_long), got %zu",
         input_names.size());
    return -1;
  }
  LOGI("AeroEar: input[0]=%s, input[1]=%s", input_names[0].c_str(),
       input_names[1].c_str());

  /* Validate output tensors */
  const auto &output_names = net_->getOutputNames();
  if (output_names.empty()) {
    LOGE("AeroEar: no output tensors found");
    return -1;
  }
  LOGI("AeroEar: output[0]=%s", output_names[0].c_str());

  /* Validate input tensor shapes: mel_short (64, 621), mel_long (64, 153) */
  const TensorInfo &tinfo_short = net_->getTensorInfo(input_names[0]);
  const TensorInfo &tinfo_long = net_->getTensorInfo(input_names[1]);
  LOGI("AeroEar: input_short shape = [%d,%d,%d,%d], elem=%u",
       tinfo_short.shape.size() > 0 ? tinfo_short.shape[0] : 0,
       tinfo_short.shape.size() > 1 ? tinfo_short.shape[1] : 0,
       tinfo_short.shape.size() > 2 ? tinfo_short.shape[2] : 0,
       tinfo_short.shape.size() > 3 ? tinfo_short.shape[3] : 0,
       tinfo_short.tensor_elem);
  LOGI("AeroEar: input_long shape = [%d,%d,%d,%d], elem=%u",
       tinfo_long.shape.size() > 0 ? tinfo_long.shape[0] : 0,
       tinfo_long.shape.size() > 1 ? tinfo_long.shape[1] : 0,
       tinfo_long.shape.size() > 2 ? tinfo_long.shape[2] : 0,
       tinfo_long.shape.size() > 3 ? tinfo_long.shape[3] : 0,
       tinfo_long.tensor_elem);

  /* Allocate work buffers */
  mel_short_all_.resize(AEROEAR_SHORT_FRAMES * AEROEAR_N_MELS);
  mel_long_all_.resize(AEROEAR_LONG_FRAMES * AEROEAR_N_MELS);
  mel_short_tpu_.resize(AEROEAR_TPU_T_SHORT * AEROEAR_N_MELS);
  proj_out_.resize(AEROEAR_TPU_T_LONG * AEROEAR_GRU_INPUT);
  snr_out_.resize(AEROEAR_TPU_T_LONG);
  act_out_.resize(AEROEAR_TPU_T_LONG);

  /* Init preprocessing context (Hann windows + FFT trig tables) */
  if (!ctx_inited_) {
    aeroear_context_init(&ctx_);
    ctx_inited_ = true;
  }

  gru_inited_ = false;

  LOGI("AeroEar: model opened successfully");
  return 0;
}

/*
 * Compute dual-window log-mel spectrogram.
 *
 * Input:  int16 PCM 48kHz mono
 * Output: mel_short_tpu (621, 64) row-major — TPU short input
 *         mel_long_all  (153, 64) row-major — TPU long input
 *
 * Short window: n_fft=1536, hop=384, 622 frames
 * Long window:  n_fft=6144, hop=1536, 153 frames
 * TPU short: first 621 frames (tpu-mlir already trimmed the last)
 */
void AeroEarClassification::computeLogMel(const int16_t *pcm, size_t n_samples,
                                          float *mel_short_tpu,
                                          float *mel_long_all) {
  /* int16 -> float32 */
  std::vector<float> pcm_float(n_samples);
  for (size_t i = 0; i < n_samples; i++) {
    pcm_float[i] = pcm[i] / 32768.0f;
  }

  /* Work buffer: long window needs n_fft + 3*n_freq = 6144 + 3*3073 = 15363 */
  int max_work = AEROEAR_LONG_NFFT + 3 * AEROEAR_LONG_NFREQ;
  std::vector<float> work(max_work);

  float *mel_short_all = this->mel_short_all_.data();
  float *mel_long_all_buf = this->mel_long_all_.data();

  /* Long window log-mel: one frame per 1536 samples */
  for (int f = 0; f < AEROEAR_LONG_FRAMES &&
                  (f * AEROEAR_LONG_HOP + AEROEAR_LONG_NFFT) <= (int)n_samples;
       f++) {
    float *frame = mel_long_all_buf + f * AEROEAR_N_MELS;
    aeroear_log_mel_frame(&ctx_, pcm_float.data() + f * AEROEAR_LONG_HOP,
                          AEROEAR_LONG_NFFT, frame, work.data());
  }

  /* Short window log-mel: one frame per 384 samples */
  for (int f = 0;
       f < AEROEAR_SHORT_FRAMES &&
       (f * AEROEAR_SHORT_HOP + AEROEAR_SHORT_NFFT) <= (int)n_samples;
       f++) {
    float *frame = mel_short_all + f * AEROEAR_N_MELS;
    aeroear_log_mel_frame(&ctx_, pcm_float.data() + f * AEROEAR_SHORT_HOP,
                          AEROEAR_SHORT_NFFT, frame, work.data());
  }

  /* Slice first 621 frames (T, C) row-major for TPU input */
  for (int m = 0; m < AEROEAR_N_MELS; m++) {
    for (int t = 0; t < AEROEAR_TPU_T_SHORT; t++) {
      mel_short_tpu[t * AEROEAR_N_MELS + m] =
          mel_short_all[t * AEROEAR_N_MELS + m];
    }
  }

  /* mel_long_all is already (T, C) row-major; caller is responsible for usage
   */
}

/*
 * TPU encoder inference: CNN + proj -> 256-dim features.
 *
 * Input:  mel_short_tpu (621, 64) row-major (T, C)
 *         mel_long_all  (153, 64) row-major (T, C)
 * Output: proj (153, 256) row-major
 *
 * Layout conversion: (T, C) -> (C, T) because cvimodel uses NCHW layout.
 */
int32_t AeroEarClassification::encoderInference(const float *mel_short_tpu,
                                                const float *mel_long_all,
                                                float *proj_out) {
  const auto &input_names = net_->getInputNames();
  const auto &output_names = net_->getOutputNames();

  const TensorInfo &tinfo_short = net_->getTensorInfo(input_names[0]);
  const TensorInfo &tinfo_long = net_->getTensorInfo(input_names[1]);

  float *ptr_short = reinterpret_cast<float *>(tinfo_short.sys_mem);
  float *ptr_long = reinterpret_cast<float *>(tinfo_long.sys_mem);

  int C = AEROEAR_N_MELS;
  int T_s = AEROEAR_TPU_T_SHORT;
  int T_l = AEROEAR_TPU_T_LONG;

  /* (T, C) -> (C, T) layout conversion, write to sys_mem */
  for (int c = 0; c < C; c++) {
    for (int t = 0; t < T_s; t++) {
      ptr_short[c * T_s + t] = mel_short_tpu[t * C + c];
    }
  }
  for (int c = 0; c < C; c++) {
    for (int t = 0; t < T_l; t++) {
      ptr_long[c * T_l + t] = mel_long_all[t * C + c];
    }
  }

  /* Flush input tensors to TPU */
  net_->updateInputTensors();

  /* TPU forward inference */
  model_timer_.TicToc("aeroear_tpu");
  int32_t ret = net_->forward();
  model_timer_.TicToc("aeroear_tpu");
  if (ret != 0) {
    LOGE("AeroEar: TPU forward failed, ret=%d", ret);
    return -1;
  }

  /* Read output tensors */
  net_->updateOutputTensors();
  const TensorInfo &tinfo_out = net_->getTensorInfo(output_names[0]);
  float *proj_ptr = reinterpret_cast<float *>(tinfo_out.sys_mem);
  size_t proj_count = tinfo_out.tensor_elem;
  std::memcpy(proj_out, proj_ptr, proj_count * sizeof(float));

  return 0;
}

/*
 * Main inference entry point.
 *
 * Input: list of BaseImage, each is 5s 48kHz int16 PCM audio.
 * Pipeline:
 *   1. int16 -> float32
 *   2. Dual-window log-mel spectrogram
 *   3. TPU encoder inference -> proj (153, 256)
 *   4. CPU GRU 153 steps -> snr, act
 *   5. Threshold detection
 *   6. Output classification result
 */
int32_t AeroEarClassification::inference(
    const std::vector<std::shared_ptr<BaseImage>> &images,
    std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas,
    const std::map<std::string, float> &parameters) {
  for (auto &image : images) {
    /* Get audio data: width = bytes, height = 1 */
    int img_width = image->getWidth();
    int img_height = image->getHeight();
    int num_samples = img_width / 2; /* int16 -> sample count */

    if (num_samples < AEROEAR_CLIP_SAMPLES) {
      LOGW("AeroEar: audio too short (%d samples, need %d)", num_samples,
           AEROEAR_CLIP_SAMPLES);
      continue;
    }

    const int16_t *pcm =
        reinterpret_cast<const int16_t *>(image->getVirtualAddress()[0]);

    /* Step 1-2: dual-window log-mel spectrogram */
    model_timer_.TicToc("aeroear_preprocess");
    computeLogMel(pcm, num_samples, mel_short_tpu_.data(),
                  mel_long_all_.data());
    model_timer_.TicToc("aeroear_preprocess");

    /* Step 3: TPU encoder inference */
    int32_t ret = encoderInference(mel_short_tpu_.data(), mel_long_all_.data(),
                                   proj_out_.data());
    if (ret != 0) {
      LOGE("AeroEar: encoder inference failed");
      continue;
    }

    /* Step 4: CPU GRU — run all 153 steps every batch.
     * Zero-padded front frames flush old drone state; recent-audio tail
     * frames set the new state. Hidden state persists across calls. */
    model_timer_.TicToc("aeroear_gru");
    if (!gru_inited_) {
      aeroear_gru_init(&gru_);
      gru_inited_ = true;
    }
    for (int t = 0; t < AEROEAR_GRU_STEPS_PER_BATCH; t++) {
      aeroear_gru_step(&gru_, proj_out_.data() + t * AEROEAR_GRU_INPUT,
                       &snr_out_[t], &act_out_[t]);
    }
    LOGI("[AeroEar] GRU ran 0..%d steps", AEROEAR_GRU_STEPS_PER_BATCH - 1);
    model_timer_.TicToc("aeroear_gru");

    /* Step 5: threshold detection over the last 30 frames (1.0s). */
    detection_frames_ = 0;
    detection_segments_ = 0;
    int in_detection = 0;
    float snr_sum = 0.0f;
    float act_sum = 0.0f;

    int start_t = AEROEAR_TPU_T_LONG - AEROEAR_DETECTION_WINDOW;

    for (int t = start_t; t < AEROEAR_TPU_T_LONG; t++) {
      int detected =
          (snr_out_[t] > model_threshold_) && (act_out_[t] > model_threshold_);
      if (detected) {
        detection_frames_++;
        if (!in_detection) {
          detection_segments_++;
          in_detection = 1;
        }
      } else {
        in_detection = 0;
      }
      snr_sum += snr_out_[t];
      act_sum += act_out_[t];
    }

    avg_snr_ = snr_sum / AEROEAR_DETECTION_WINDOW;
    avg_act_ = act_sum / AEROEAR_DETECTION_WINDOW;

    /* Log output (consistent with ipcamera implementation) */
    if (detection_frames_ > 0) {
      LOGW(
          "AeroEar: drone detected! %d/%d frames, %d segments, "
          "avg SNR %.1f dB, avg Act %.3f",
          detection_frames_, AEROEAR_DETECTION_WINDOW, detection_segments_,
          (double)avg_snr_, (double)avg_act_);
    } else {
      LOGI("AeroEar: no drone, avg SNR %.1f dB, avg Act %.3f", (double)avg_snr_,
           (double)avg_act_);
    }

    /* Step 6: output parsing */
    model_timer_.TicToc("aeroear_post");
    std::vector<std::shared_ptr<BaseImage>> batch_images = {image};
    outputParse(batch_images, out_datas);
    model_timer_.TicToc("aeroear_post");
  }

  return 0;
}

/*
 * Output parsing: convert detection results to classification output.
 * class_id: 0 = background, 1 = drone
 * score: detection frame ratio (as confidence)
 */
int32_t AeroEarClassification::outputParse(
    const std::vector<std::shared_ptr<BaseImage>> &images,
    std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) {
  auto output_info = std::make_shared<ModelClassificationInfo>();

  if (detection_frames_ > 0) {
    output_info->topk_class_ids.push_back(1); /* drone */
    output_info->topk_scores.push_back(static_cast<float>(detection_frames_) /
                                       AEROEAR_DETECTION_WINDOW);
    if (type_mapping_.find(1) != type_mapping_.end()) {
      output_info->topk_object_types.push_back(type_mapping_[1]);
    }
  } else {
    output_info->topk_class_ids.push_back(0); /* background */
    output_info->topk_scores.push_back(1.0f);
    if (type_mapping_.find(0) != type_mapping_.end()) {
      output_info->topk_object_types.push_back(type_mapping_[0]);
    }
  }

  out_datas.push_back(output_info);
  return 0;
}

/* Get detection stats from the last inference. */
void AeroEarClassification::getDetectionStats(float &avg_snr, float &avg_act,
                                              int &detection_frames,
                                              int &detection_segments) const {
  avg_snr = avg_snr_;
  avg_act = avg_act_;
  detection_frames = detection_frames_;
  detection_segments = detection_segments_;
}