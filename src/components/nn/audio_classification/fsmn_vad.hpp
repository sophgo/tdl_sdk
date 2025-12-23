#ifndef FSMN_VAD_ENCODER_HPP
#define FSMN_VAD_ENCODER_HPP

#include "feature-fbank.h"
#include "model/base_model.hpp"
#include "online-feature.h"
#include "utils/e2e_vad.hpp"
#include "utils/tdl_log.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#define VAD_SAMPLE_RATE 16000
#define VAD_NUM_BIN 80
#define VAD_LFR_M 5
#define VAD_LFR_N 1

class FsmnVad : public BaseModel {
 public:
  FsmnVad();
  virtual ~FsmnVad();

  int32_t setupNetwork(NetParam& net_param);
  virtual int32_t inference(
      const std::shared_ptr<BaseImage>& image,
      std::shared_ptr<ModelOutputInfo>& out_data,
      const std::map<std::string, float>& parameters = {}) override;
  virtual int32_t outputParse(const std::shared_ptr<BaseImage>& image,
                              std::shared_ptr<ModelOutputInfo>& out_data,
                              std::vector<std::vector<int>>& segment);
  virtual int32_t onModelOpened() override;

 private:
  int32_t processInput(std::vector<float>& wav_data, bool is_final = false);

  void FbankKaldi(float sample_rate, std::vector<std::vector<float>>& vad_feats,
                  std::vector<float>& waves);

  int OnlineLfrCmvn(std::vector<std::vector<float>>& vad_feats,
                    bool input_finished);

  void ExtractFeatsOnline(float sample_rate,
                          std::vector<std::vector<float>>& vad_feats,
                          std::vector<float>& waves, bool input_finished);

  static int32_t compute_frame_num(int sample_length, int frame_sample_length,
                                   int frame_shift_sample_length);

  void InitCmvn();

  void Reset();

  void ResetCache();

  void allocateFsmnCacheDynamic();
  void updateOutputCaches();
  static size_t numElementsFromShape(const std::vector<int>& shape);
  static bool containsIgnoreCase(const std::string& s, const std::string& pat);

 private:
  // VAD 参数
  E2EVadModel vad_scorer = E2EVadModel();
  int vad_sample_rate_ = VAD_SAMPLE_RATE;
  float vad_speech_noise_thres_ = 0.6f;
  int vad_max_len_ = 16000;         // ms
  int vad_silence_duration_ = 800;  // ms

  // 帧级参数
  float frame_length_ms_ = 25.0f;
  float frame_shift_ms_ = 10.0f;
  int frame_sample_length = 0;
  int frame_shift_sample_length = 0;

  // FBank 配置
  knf::FbankOptions fbank_opts_;

  // CMVN
  bool cmvn_inited_ = false;
  std::vector<float> means_list_;
  std::vector<float> vars_list_;

  // LFR 参数
  int lfr_m = VAD_LFR_M;  // 5
  int lfr_n = VAD_LFR_N;  // 1

  // 运行时缓存：输入、LFR 剩余、Reserve 波形（对齐）
  std::vector<float> input_cache_;
  std::vector<std::vector<float>> lfr_splice_cache_;
  std::vector<float> reserve_waveforms_;

  // FSMN 缓存
  int fsmn_layers_ = 4;  // 层数
  int proj_dim_ = 128;   // 投影维
  int lorder_ = 20;      // 时序深度（lorder-1 作为缓存）
  std::vector<int> cache_offsets_;
  float* fsmn_cache_ = nullptr;

  // 动态 I/O 映射
  std::string feature_input_name_;
  int feature_input_index_ = -1;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;

  struct CacheSlot {
    std::string in_name;         // 输入缓存名
    std::string out_name;        // 输出缓存名
    int layer_idx = -1;          // 层序号（用于 debug）
    int size = 0;                // 此层缓存元素数
    int offset = 0;              // 在 fsmn_cache_ 中的偏移
    std::vector<int> out_shape;  // 输出缓存形状
  };
  std::vector<CacheSlot> cache_slots_;
  bool dynamic_cache_layout_ready_ = false;

  // 网络输入形状
  int segment_size_ = 0;  // T（例如 100）
  int feature_dim_ = 0;   // C（例如 400 = 80 * 5）

  // 评分对齐
  std::vector<float> last_vad_wave_for_scorer_;
  int last_valid_prob_frames_ = 0;
  bool last_is_final_ = false;

  // 流式分块控制
  bool first_chunk_ = true;
  size_t sample_offset_ = 0;

  // 流式模式下，避免每次 inference() 都重置 vad_scorer
  bool vad_stream_inited_ = false;
};

#endif  // FSMN_VAD_ENCODER_HPP
