#include "audio_classification/fsmn_vad.hpp"
#include "utils/tdl_log.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

bool FsmnVad::containsIgnoreCase(const std::string& s, const std::string& pat) {
  auto tolower_str = [](const std::string& x) {
    std::string y(x);
    std::transform(y.begin(), y.end(), y.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return y;
  };
  std::string ss = tolower_str(s);
  std::string pp = tolower_str(pat);
  return ss.find(pp) != std::string::npos;
}

size_t FsmnVad::numElementsFromShape(const std::vector<int>& shape) {
  if (shape.empty()) return 0;
  size_t n = 1;
  for (int v : shape) {
    if (v <= 0) return 0;
    n *= static_cast<size_t>(v);
  }
  return n;
}

int32_t FsmnVad::compute_frame_num(int sample_length, int frame_sample_length,
                                   int frame_shift_sample_length) {
  if (sample_length < frame_sample_length) {
    return 0;
  }
  int frame_num =
      (sample_length - frame_sample_length) / frame_shift_sample_length + 1;
  return (frame_num >= 1) ? frame_num : 0;
}

FsmnVad::FsmnVad() {
  frame_sample_length =
      static_cast<int>(frame_length_ms_ * vad_sample_rate_ / 1000);
  frame_shift_sample_length =
      static_cast<int>(frame_shift_ms_ * vad_sample_rate_ / 1000);

  knf::FbankOptions fbank_opts;
  fbank_opts.frame_opts.dither = 0.0f;
  fbank_opts.frame_opts.snip_edges = true;
  fbank_opts.frame_opts.samp_freq = static_cast<float>(vad_sample_rate_);
  fbank_opts.frame_opts.window_type = "hamming";
  fbank_opts.frame_opts.frame_shift_ms = frame_shift_ms_;
  fbank_opts.frame_opts.frame_length_ms = frame_length_ms_;
  fbank_opts.frame_opts.preemph_coeff = 0.9700000286102295f;
  fbank_opts.frame_opts.blackman_coeff = 0.41999998688697815f;
  fbank_opts.mel_opts.num_bins = VAD_NUM_BIN;  // 80
  fbank_opts.mel_opts.high_freq = 0.0f;
  fbank_opts.mel_opts.low_freq = 20.0f;
  fbank_opts.energy_floor = 0.0f;
  fbank_opts.mel_opts.debug_mel = false;
  fbank_opts_ = fbank_opts;

  InitCmvn();

  int total_cache_size = 0;
  cache_offsets_.clear();
  cache_offsets_.push_back(0);
  for (int i = 0; i < fsmn_layers_; ++i) {
    int layer_size = proj_dim_ * (lorder_ - 1);
    total_cache_size += layer_size;
    cache_offsets_.push_back(total_cache_size);
  }
  if (total_cache_size > 0) {
    fsmn_cache_ =
        static_cast<float*>(std::malloc(total_cache_size * sizeof(float)));
    std::memset(fsmn_cache_, 0, total_cache_size * sizeof(float));
  }

  // 初始化 LFR 与输入缓存状态
  lfr_splice_cache_.clear();
  input_cache_.clear();
  reserve_waveforms_.clear();
  last_vad_wave_for_scorer_.clear();
  last_valid_prob_frames_ = 0;

  first_chunk_ = true;
  sample_offset_ = 0;
  last_is_final_ = false;
}

FsmnVad::~FsmnVad() {
  if (fsmn_cache_) {
    std::free(fsmn_cache_);
    fsmn_cache_ = nullptr;
  }
  first_chunk_ = true;
  sample_offset_ = 0;
}

int32_t FsmnVad::setupNetwork(NetParam& net_param) {
  net_ = NetFactory::createNet(net_param, net_param.platform);
  if (!net_) {
    LOGE("FSMN-VAD: Failed to create net");
    return -1;
  }
  return net_->setup();
}

void FsmnVad::allocateFsmnCacheDynamic() {
  int total_cache_size = 0;
  cache_offsets_.clear();
  cache_offsets_.push_back(0);

  bool shape_ok = true;
  for (size_t i = 0; i < cache_slots_.size(); ++i) {
    const auto& cs = cache_slots_[i];
    size_t elems = numElementsFromShape(cs.out_shape);
    if (elems == 0) {
      shape_ok = false;
      break;
    }
    total_cache_size += static_cast<int>(elems);
    cache_offsets_.push_back(total_cache_size);
  }
  if (!shape_ok || total_cache_size <= 0) {
    LOGI(
        "FSMN-VAD: dynamic cache layout not available, fallback to hardcoded "
        "[%d layers, proj=%d, lorder=%d]",
        fsmn_layers_, proj_dim_, lorder_);
    // 回退已在构造函数中完成，不再重建
    dynamic_cache_layout_ready_ = false;
    return;
  }

  // 重建缓存
  if (fsmn_cache_) {
    std::free(fsmn_cache_);
    fsmn_cache_ = nullptr;
  }
  fsmn_cache_ =
      static_cast<float*>(std::malloc(total_cache_size * sizeof(float)));
  std::memset(fsmn_cache_, 0, total_cache_size * sizeof(float));

  // 设置每层 offset/size
  for (size_t i = 0; i < cache_slots_.size(); ++i) {
    cache_slots_[i].offset = cache_offsets_[i];
    cache_slots_[i].size = cache_offsets_[i + 1] - cache_offsets_[i];
    cache_slots_[i].layer_idx = static_cast<int>(i);
  }
  dynamic_cache_layout_ready_ = true;

  LOGI("FSMN-VAD: dynamic cache layout built, total=%d elems, layers=%zu",
       total_cache_size, cache_slots_.size());
}

int32_t FsmnVad::onModelOpened() {
  if (!net_) return -1;

  input_names_ = net_->getInputNames();
  output_names_ = net_->getOutputNames();

  feature_input_index_ = -1;
  feature_input_name_.clear();
  for (size_t i = 0; i < input_names_.size(); ++i) {
    const auto& name = input_names_[i];
    const TensorInfo& tinfo = net_->getTensorInfo(name);
    if (tinfo.shape.size() >= 2) {
      int T = tinfo.shape[1];
      int C = tinfo.shape[2];
      if (C == VAD_NUM_BIN * VAD_LFR_M) {
        feature_input_index_ = static_cast<int>(i);
        feature_input_name_ = name;
        segment_size_ = T;
        feature_dim_ = C;
        break;
      }
    }
  }

  cache_slots_.clear();

  // 收集可能的 cache 输出
  std::vector<std::string> cache_out_names;
  for (const auto& on : output_names_) {
    if (containsIgnoreCase(on, "cache")) {
      cache_out_names.push_back(on);
    }
  }
  // 同时收集输入侧的 cache 名称
  std::vector<std::string> cache_in_names;
  for (size_t i = 0; i < input_names_.size(); ++i) {
    if ((int)i == feature_input_index_) continue;
    const auto& in = input_names_[i];
    if (containsIgnoreCase(in, "cache")) {
      cache_in_names.push_back(in);
    }
  }

  // 按出现顺序一一匹配
  size_t num_layers = std::min(cache_in_names.size(), cache_out_names.size());
  for (size_t li = 0; li < num_layers; ++li) {
    CacheSlot cs;
    cs.in_name = cache_in_names[li];
    cs.out_name = cache_out_names[li];
    const TensorInfo& otinfo = net_->getTensorInfo(cs.out_name);
    cs.out_shape = otinfo.shape;
    cache_slots_.push_back(cs);
    LOGI("FSMN-VAD: cache layer %zu: in=%s, out=%s, out_shape=[%s]", li,
         cs.in_name.c_str(), cs.out_name.c_str(),
         [&] {
           std::ostringstream oss;
           for (size_t k = 0; k < cs.out_shape.size(); ++k) {
             if (k) oss << ",";
             oss << cs.out_shape[k];
           }
           return oss.str();
         }()
             .c_str());
  }

  allocateFsmnCacheDynamic();

  return 0;
}

void FsmnVad::InitCmvn() {
  if (cmvn_inited_) return;

  static const float base_means[80] = {
      -8.311879, -8.600912, -9.615928, -10.43595, -11.21292, -11.88333,
      -12.36243, -12.63706, -12.8818,  -12.83066, -12.89103, -12.95666,
      -13.19763, -13.40598, -13.49113, -13.5546,  -13.55639, -13.51915,
      -13.68284, -13.53289, -13.42107, -13.65519, -13.50713, -13.75251,
      -13.76715, -13.87408, -13.73109, -13.70412, -13.56073, -13.53488,
      -13.54895, -13.56228, -13.59408, -13.62047, -13.64198, -13.66109,
      -13.62669, -13.58297, -13.57387, -13.4739,  -13.53063, -13.48348,
      -13.61047, -13.64716, -13.71546, -13.79184, -13.90614, -14.03098,
      -14.18205, -14.35881, -14.48419, -14.60172, -14.70591, -14.83362,
      -14.92122, -15.00622, -15.05122, -15.03119, -14.99028, -14.92302,
      -14.86927, -14.82691, -14.7972,  -14.76909, -14.71356, -14.61277,
      -14.51696, -14.42252, -14.36405, -14.30451, -14.23161, -14.19851,
      -14.16633, -14.15649, -14.10504, -13.99518, -13.79562, -13.3996,
      -12.7767,  -11.71208};

  static const float base_vars[80] = {
      0.155775,  0.154484,  0.1527379, 0.1518718, 0.1506028, 0.1489256,
      0.147067,  0.1447061, 0.1436307, 0.1443568, 0.1451849, 0.1455157,
      0.1452821, 0.1445717, 0.1439195, 0.1435867, 0.1436018, 0.1438781,
      0.1442086, 0.1448844, 0.1454756, 0.145663,  0.146268,  0.1467386,
      0.1472724, 0.147664,  0.1480913, 0.1483739, 0.1488841, 0.1493636,
      0.1497088, 0.1500379, 0.1502916, 0.1505389, 0.1506787, 0.1507102,
      0.1505992, 0.1505445, 0.1505938, 0.1508133, 0.1509569, 0.1512396,
      0.1514625, 0.1516195, 0.1516156, 0.1515561, 0.1514966, 0.1513976,
      0.1512612, 0.151076,  0.1510596, 0.1510431, 0.151077,  0.1511168,
      0.1511917, 0.151023,  0.1508045, 0.1505885, 0.1503493, 0.1502373,
      0.1501726, 0.1500762, 0.1500065, 0.1499782, 0.150057,  0.1502658,
      0.150469,  0.1505335, 0.1505505, 0.1505328, 0.1504275, 0.1502438,
      0.1499674, 0.1497118, 0.1494661, 0.1493102, 0.1493681, 0.1495501,
      0.1499738, 0.1509654};

  means_list_.clear();
  vars_list_.clear();

  // 展开至 80 * lfr_m（外部 LFR：m=5）
  for (int r = 0; r < lfr_m; ++r) {
    means_list_.insert(means_list_.end(), std::begin(base_means),
                       std::end(base_means));
    vars_list_.insert(vars_list_.end(), std::begin(base_vars),
                      std::end(base_vars));
  }
  cmvn_inited_ = true;
}

void FsmnVad::FbankKaldi(float sample_rate,
                         std::vector<std::vector<float>>& vad_feats,
                         std::vector<float>& waves) {
  // 1) 合并输入缓存
  if (!input_cache_.empty()) {
    waves.insert(waves.begin(), input_cache_.begin(), input_cache_.end());
  }

  // 2) 计算可形成的帧数
  int frame_number =
      compute_frame_num(static_cast<int>(waves.size()), frame_sample_length,
                        frame_shift_sample_length);

  // 3) 尾部回灌到输入缓存
  input_cache_.clear();
  if (frame_number == 0) {
    input_cache_ = waves;  // 全部回灌
    return;
  } else {
    int tail_start = frame_number * frame_shift_sample_length;
    if (tail_start >= 0 && tail_start <= static_cast<int>(waves.size())) {
      input_cache_.insert(input_cache_.begin(), waves.begin() + tail_start,
                          waves.end());
    }
  }

  // 4) 删除未对齐尾部：仅保留可对齐处理部分
  int aligned_end =
      (frame_number - 1) * frame_shift_sample_length + frame_sample_length;
  if (aligned_end >= 0 && aligned_end < static_cast<int>(waves.size())) {
    waves.erase(waves.begin() + aligned_end, waves.end());
  }

  // 5) 对齐段做 fbank
  knf::OnlineFbank fbank(fbank_opts_);
  std::vector<float> buf(waves.size());
  for (int32_t i = 0; i < static_cast<int32_t>(waves.size()); ++i) {
    buf[i] = waves[i] * 32768.0f;
  }
  fbank.AcceptWaveform(sample_rate, buf.data(), buf.size());

  int32_t frames = fbank.NumFramesReady();
  for (int32_t i = 0; i < frames; ++i) {
    const float* frame = fbank.GetFrame(i);
    std::vector<float> frame_vector(frame,
                                    frame + fbank_opts_.mel_opts.num_bins);
    vad_feats.emplace_back(std::move(frame_vector));
  }
}

int FsmnVad::OnlineLfrCmvn(std::vector<std::vector<float>>& vad_feats,
                           bool input_finished) {
  std::vector<std::vector<float>> out_feats;
  int T = static_cast<int>(vad_feats.size());
  int T_lrf = static_cast<int>(
      std::ceil((T - (lfr_m - 1) / 2) / static_cast<float>(lfr_n)));
  int lfr_splice_frame_idxs = T_lrf;

  std::vector<float> p;
  for (int i = 0; i < T_lrf; ++i) {
    int remain = T - i * lfr_n;
    if (lfr_m <= remain) {
      for (int j = 0; j < lfr_m; ++j) {
        const auto& frm = vad_feats[i * lfr_n + j];
        p.insert(p.end(), frm.begin(), frm.end());
      }
      out_feats.emplace_back(p);
      p.clear();
    } else {
      if (input_finished) {
        int avail = std::max(0, remain);
        int num_padding = lfr_m - avail;
        for (int j = 0; j < avail; ++j) {
          const auto& frm = vad_feats[i * lfr_n + j];
          p.insert(p.end(), frm.begin(), frm.end());
        }
        for (int j = 0; j < num_padding; ++j) {
          const auto& frm = vad_feats.back();
          p.insert(p.end(), frm.begin(), frm.end());
        }
        out_feats.emplace_back(p);
        p.clear();
      } else {
        lfr_splice_frame_idxs = i;
        break;
      }
    }
  }

  lfr_splice_frame_idxs = std::min(T - 1, lfr_splice_frame_idxs * lfr_n);

  // 更新剩余帧缓存
  lfr_splice_cache_.clear();
  lfr_splice_cache_.insert(lfr_splice_cache_.begin(),
                           vad_feats.begin() + lfr_splice_frame_idxs,
                           vad_feats.end());

  // CMVN
  for (auto& out_feat : out_feats) {
    for (int j = 0; j < static_cast<int>(means_list_.size()); ++j) {
      out_feat[j] = (out_feat[j] + means_list_[j]) * vars_list_[j];
    }
  }

  vad_feats = std::move(out_feats);
  return lfr_splice_frame_idxs;
}

void FsmnVad::ExtractFeatsOnline(float sample_rate,
                                 std::vector<std::vector<float>>& vad_feats,
                                 std::vector<float>& waves,
                                 bool input_finished) {
  // fbank 提取（含输入缓存与对齐截断）
  FbankKaldi(sample_rate, vad_feats, waves);

  if (!vad_feats.empty()) {
    // 在 LFR/CMVN 前，合并历史保留波形，用于后续对齐计算
    if (!reserve_waveforms_.empty()) {
      waves.insert(waves.begin(), reserve_waveforms_.begin(),
                   reserve_waveforms_.end());
    }
    // 首次补左半上下文
    if (lfr_splice_cache_.empty()) {
      for (int i = 0; i < (lfr_m - 1) / 2; ++i) {
        lfr_splice_cache_.emplace_back(vad_feats[0]);
      }
    }

    // 是否足够做一次 LFR 合并
    if (static_cast<int>(vad_feats.size() + lfr_splice_cache_.size()) >=
        lfr_m) {
      // 拼上历史左侧上下文
      vad_feats.insert(vad_feats.begin(), lfr_splice_cache_.begin(),
                       lfr_splice_cache_.end());

      // 计算从 waves 中可形成的 fbank 帧数
      int frame_from_waves =
          compute_frame_num(static_cast<int>(waves.size()), frame_sample_length,
                            frame_shift_sample_length);

      // 首次 chunk 用于补偿左侧上下文
      int minus_frame = reserve_waveforms_.empty() ? (lfr_m - 1) / 2 : 0;

      // LFR+CMVN
      int lfr_splice_frame_idxs = OnlineLfrCmvn(vad_feats, input_finished);

      // 本次需保留的起始帧（对齐到帧移）
      int reserve_frame_idx = std::abs(lfr_splice_frame_idxs - minus_frame);

      // 维护保留波形
      reserve_waveforms_.clear();
      if (frame_from_waves > 0 && reserve_frame_idx <= frame_from_waves) {
        reserve_waveforms_.insert(
            reserve_waveforms_.begin(),
            waves.begin() + reserve_frame_idx * frame_shift_sample_length,
            waves.begin() + frame_from_waves * frame_shift_sample_length);
      }

      // 将 waves 裁剪到本次已对齐处理的长度（丢弃未对齐的尾部）
      int sample_length = (frame_from_waves - 1) * frame_shift_sample_length +
                          frame_sample_length;
      if (sample_length > 0 &&
          sample_length <= static_cast<int>(waves.size())) {
        waves.erase(waves.begin() + sample_length, waves.end());
      }
    } else {
      // 不足以做 LFR：仅缓存并等待下一段
      reserve_waveforms_.clear();
      if (!waves.empty()) {
        int start =
            std::max(0, frame_sample_length - frame_shift_sample_length);
        if (start < static_cast<int>(waves.size())) {
          reserve_waveforms_.insert(reserve_waveforms_.begin(),
                                    waves.begin() + start, waves.end());
        }
      }
      lfr_splice_cache_.insert(lfr_splice_cache_.end(), vad_feats.begin(),
                               vad_feats.end());
      vad_feats.clear();
    }
  } else {
    // fbank 未产生帧：若是流末尾，拿缓存做最后一次 LFR/CMVN
    if (input_finished) {
      if (!reserve_waveforms_.empty()) {
        waves = reserve_waveforms_;
      }
      vad_feats = lfr_splice_cache_;
      if (vad_feats.empty()) {
        LOGE("FSMN-VAD: vad_feats's size is 0 at final");
      } else {
        OnlineLfrCmvn(vad_feats, input_finished);
      }
    }
  }

  // 重置状态与缓存
  if (input_finished) {
    Reset();
    ResetCache();
  }
}

void FsmnVad::Reset() {
  input_cache_.clear();
  lfr_splice_cache_.clear();
  reserve_waveforms_.clear();
  last_vad_wave_for_scorer_.clear();
  last_valid_prob_frames_ = 0;
  last_is_final_ = false;
}

void FsmnVad::ResetCache() {
  if (!cache_offsets_.empty() && fsmn_cache_) {
    std::memset(fsmn_cache_, 0, cache_offsets_.back() * sizeof(float));
  }
}

int32_t FsmnVad::processInput(std::vector<float>& wav_data, bool is_final) {
  last_is_final_ = is_final;

  std::vector<std::vector<float>> vad_feats;
  std::vector<float> waves = wav_data;
  ExtractFeatsOnline(static_cast<float>(vad_sample_rate_), vad_feats, waves,
                     is_final);

  // 没有可前传的特征（不足 LFR 或仅缓存），直接返回
  if (vad_feats.empty()) {
    last_vad_wave_for_scorer_.clear();
    last_valid_prob_frames_ = 0;
    return 0;
  }

  // 保存与概率对齐的波形
  last_vad_wave_for_scorer_ = std::move(waves);

  if (!net_) return -1;
  auto input_names = net_->getInputNames();
  // 写入每个输入
  for (size_t i = 0; i < input_names.size(); ++i) {
    const std::string& in_name = input_names[i];
    const TensorInfo& tinfo = net_->getTensorInfo(in_name);
    float* input_ptr = reinterpret_cast<float*>(tinfo.sys_mem);
    size_t elems = numElementsFromShape(tinfo.shape);
    if (!input_ptr || elems == 0) {
      LOGI("FSMN-VAD: skip input %s (null or zero elems)", in_name.c_str());
      continue;
    }

    if ((int)i == feature_input_index_) {
      // 特征输入
      int num_frames =
          std::min(segment_size_, static_cast<int>(vad_feats.size()));
      last_valid_prob_frames_ = num_frames;
      // [T, C] 顺序拷贝
      for (int j = 0; j < num_frames; ++j) {
        std::memcpy(input_ptr + j * feature_dim_, vad_feats[j].data(),
                    static_cast<size_t>(feature_dim_) * sizeof(float));
      }
      // padding 清零
      if (num_frames < segment_size_) {
        std::memset(input_ptr + num_frames * feature_dim_, 0,
                    static_cast<size_t>(segment_size_ - num_frames) *
                        feature_dim_ * sizeof(float));
      }
    } else {
      // 是否为缓存输入
      auto it = std::find_if(
          cache_slots_.begin(), cache_slots_.end(),
          [&](const CacheSlot& cs) { return cs.in_name == in_name; });
      if (it != cache_slots_.end()) {
        if (fsmn_cache_ && it->size > 0 &&
            it->offset + it->size <= cache_offsets_.back()) {
          std::memcpy(input_ptr, fsmn_cache_ + it->offset,
                      static_cast<size_t>(it->size) * sizeof(float));
        } else if (fsmn_cache_ && !dynamic_cache_layout_ready_) {
          int layer_idx = it->layer_idx >= 0 ? it->layer_idx : 0;
          if (layer_idx < static_cast<int>(cache_offsets_.size()) - 1) {
            int offset = cache_offsets_[layer_idx];
            int size = cache_offsets_[layer_idx + 1] - offset;
            std::memcpy(input_ptr, fsmn_cache_ + offset,
                        static_cast<size_t>(size) * sizeof(float));
          } else {
            std::memset(input_ptr, 0, elems * sizeof(float));
          }
        } else {
          std::memset(input_ptr, 0, elems * sizeof(float));
        }
      } else {
        // 未识别的辅助输入（如 seq_len/offset/mask），统一清零，避免脏数据
        std::memset(input_ptr, 0, elems * sizeof(float));
      }
    }
  }

  return 0;
}

void FsmnVad::updateOutputCaches() {
  if (!net_) return;
  std::vector<std::string> output_names = net_->getOutputNames();
  if (output_names.empty()) return;

  // 按 cache_slots_ 的 out_name 将输出缓存回灌
  for (auto& cs : cache_slots_) {
    if (cs.out_name.empty()) continue;
    std::shared_ptr<BaseTensor> cache_tensor =
        net_->getOutputTensor(cs.out_name);
    if (!cache_tensor) continue;
    float* cache_ptr = cache_tensor->getBatchPtr<float>(0);
    size_t cache_size = cache_tensor->getNumElements();
    if (!cache_ptr || cache_size == 0) continue;

    if (fsmn_cache_ && cs.size > 0 &&
        static_cast<size_t>(cs.size) == cache_size) {
      std::memcpy(fsmn_cache_ + cs.offset, cache_ptr,
                  cache_size * sizeof(float));
    } else if (fsmn_cache_ && !dynamic_cache_layout_ready_) {
      LOGI(
          "FSMN-VAD: cache slot mismatch on %s (size=%zu), keep fallback cache",
          cs.out_name.c_str(), cache_size);
    }
  }
}

int32_t FsmnVad::inference(const std::shared_ptr<BaseImage>& image,
                           std::shared_ptr<ModelOutputInfo>& out_data,
                           const std::map<std::string, float>& parameters) {
  if (!image || !net_) return -1;
  out_data = std::make_shared<ModelVADInfo>();
  int num_samples = image->getWidth() / 2;  // 16-bit
  auto va = image->getVirtualAddress();
  if (va.empty() || !va[0] || num_samples <= 0) {
    LOGE("FSMN-VAD: invalid image/pcm");
    return -1;
  }
  short* pcm = reinterpret_cast<short*>(va[0]);
  std::vector<float> current_wav(num_samples);
  for (int i = 0; i < num_samples; ++i) {
    current_wav[i] = static_cast<float>(pcm[i]) / 32768.0f;
  }

  bool is_final =
      parameters.count("is_final") ? (parameters.at("is_final") > 0) : true;
  LOGI("FsmnVad::inference, is_final: %d, samples: %d", is_final ? 1 : 0,
       num_samples);
  LOGI(
      "FsmnVad::inference, vad_speech_noise_thres: %.3f, "
      "vad_silence_duration_ms: %d, vad_max_len_ms: %d",
      vad_speech_noise_thres_, vad_silence_duration_, vad_max_len_);

  size_t speech_length = current_wav.size();
  int step = vad_sample_rate_;

  if (!vad_stream_inited_) {
    first_chunk_ = true;
    std::vector<std::vector<float>> empty_scores;
    std::vector<float> empty_waveform;
    vad_scorer(empty_scores, empty_waveform, true, true, vad_silence_duration_,
               vad_max_len_, vad_speech_noise_thres_, vad_sample_rate_);
    vad_stream_inited_ = true;
  }

  while (sample_offset_ < speech_length) {
    size_t current_step = (first_chunk_) ? static_cast<size_t>(step * 1.04)
                                         : static_cast<size_t>(step);
    bool chunk_final = false;

    if (sample_offset_ + current_step >= speech_length - 1) {
      current_step = speech_length - sample_offset_;
      // 注意：buffer 结束 != 流结束。只有外部 is_final=1 时才触发 flush/reset。
      chunk_final = is_final;
    }

    size_t end = sample_offset_ + current_step;
    if (first_chunk_) {
      first_chunk_ = false;
    }

    std::vector<float> audio_chunk(current_wav.begin() + sample_offset_,
                                   current_wav.begin() + end);

    processInput(audio_chunk, chunk_final);

    // 若本次没有有效帧（不足 LFR 或仅缓存），跳过前传
    if (last_valid_prob_frames_ <= 0 || last_vad_wave_for_scorer_.empty()) {
      sample_offset_ = end;
      continue;
    }

    net_->updateInputTensors();
    net_->forward();
    net_->updateOutputTensors();

    // 更新 FSMN 输出缓存
    updateOutputCaches();
    std::vector<std::vector<int>> segment;

    outputParse(image, out_data, segment);
    sample_offset_ = end;
  }
  // 循环结束后：sample_offset_ 是单次 buffer 内部索引，始终归零。
  // 仅当 is_final=1 时结束本次流式会话，允许下次重新 reset。
  if (is_final) {
    first_chunk_ = true;
    vad_stream_inited_ = false;
  }
  sample_offset_ = 0;
  std::shared_ptr<ModelVADInfo> vad_meta =
      std::static_pointer_cast<ModelVADInfo>(out_data);

  if (!vad_meta->segments.empty()) {
    vad_meta->has_segments = true;
  }

  if (vad_meta && !vad_meta->segments.empty()) {
    for (const auto& batch_segments : vad_meta->segments) {
      for (const auto& seg : batch_segments) {
        if (seg.size() < 2) continue;
        const int start_ms = seg[0];
        const int end_ms = seg[1];
        if (start_ms >= 0 && end_ms < 0) {
          vad_meta->start_event = true;
        }
        if (end_ms >= 0) {
          vad_meta->end_event = true;  // 收到终点事件
        }
      }
    }
  }

  return 0;
}

int32_t FsmnVad::outputParse(const std::shared_ptr<BaseImage>& image,
                             std::shared_ptr<ModelOutputInfo>& out_data,
                             std::vector<std::vector<int>>& segment) {
  if (!net_) return -1;

  std::shared_ptr<ModelVADInfo> vad_meta =
      std::static_pointer_cast<ModelVADInfo>(out_data);
  std::vector<std::string> output_names = net_->getOutputNames();
  if (output_names.empty()) {
    LOGE("FSMN-VAD: No output tensors available");
    return -1;
  }
  std::string prob_name = output_names[0];
  for (const auto& name : output_names) {
    if (containsIgnoreCase(name, "softmax")) {
      prob_name = name;
      break;
    }
  }

  std::shared_ptr<BaseTensor> prob_tensor = net_->getOutputTensor(prob_name);
  if (!prob_tensor) {
    LOGE("FSMN-VAD: Failed to get output tensor: %s", prob_name.c_str());
    return -1;
  }

  float* prob_ptr = prob_tensor->getBatchPtr<float>(0);
  if (!prob_ptr) {
    LOGE("FSMN-VAD: Failed to get output tensor data for: %s",
         prob_name.c_str());
    return -1;
  }

  // 解析输出维度
  const TensorInfo tinfo = net_->getTensorInfo(prob_name);
  int time_dim = tinfo.shape[1];
  int class_dim = tinfo.shape[2];

  const size_t num_elems = prob_tensor->getNumElements();

  if (time_dim <= 0 || class_dim <= 0) {
    if (segment_size_ > 0) {
      time_dim = segment_size_;
      if (num_elems % time_dim == 0) {
        class_dim = static_cast<int>(num_elems / time_dim);
      } else {
        time_dim = 1;
        class_dim = static_cast<int>(num_elems);
      }
    } else {
      time_dim = 1;
      class_dim = static_cast<int>(num_elems);
    }
  }

  if (time_dim <= 0 || class_dim <= 0) {
    LOGE("FSMN-VAD: Invalid output dims: T=%d, C=%d, elems=%zu", time_dim,
         class_dim, num_elems);
    return -1;
  }

  // 截取有效概率帧数，避免 padding 残段
  int valid_T = std::min(time_dim, last_valid_prob_frames_);
  if (valid_T <= 0 || last_vad_wave_for_scorer_.empty()) {
    // 没有有效数据，直接返回
    return 0;
  }

  std::vector<std::vector<float>> scores;
  scores.reserve(valid_T);
  for (int t = 0; t < valid_T; ++t) {
    const float* row = prob_ptr + t * class_dim;
    scores.emplace_back(row, row + class_dim);
  }

  // 使用与概率对齐的波形
  std::vector<float>& waveform = last_vad_wave_for_scorer_;

  bool is_final = last_is_final_;

  bool online_mode = !is_final;

  segment =
      vad_scorer(scores, waveform, is_final, online_mode, vad_silence_duration_,
                 vad_max_len_, vad_speech_noise_thres_, vad_sample_rate_);

  if (!segment.empty()) {
    vad_meta->segments.push_back(segment);
  }

  return 0;
}
