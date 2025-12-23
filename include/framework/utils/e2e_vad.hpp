#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

// 移除namespace funasr，直接定义枚举和类
enum class VadStateMachine {
  kVadInStateStartPointNotDetected = 1,
  kVadInStateInSpeechSegment = 2,
  kVadInStateEndPointDetected = 3
};

enum class FrameState {
  kFrameStateInvalid = -1,
  kFrameStateSpeech = 1,
  kFrameStateSil = 0
};

// final voice/unvoice state per frame
enum class AudioChangeState {
  kChangeStateSpeech2Speech = 0,
  kChangeStateSpeech2Sil = 1,
  kChangeStateSil2Sil = 2,
  kChangeStateSil2Speech = 3,
  kChangeStateNoBegin = 4,
  kChangeStateInvalid = 5
};

enum class VadDetectMode {
  kVadSingleUtteranceDetectMode = 0,
  kVadMutipleUtteranceDetectMode = 1
};

class VADXOptions {
 public:
  int sample_rate;
  int detect_mode;
  int snr_mode;
  int max_end_silence_time;
  int max_start_silence_time;
  bool do_start_point_detection;
  bool do_end_point_detection;
  int window_size_ms;
  int sil_to_speech_time_thres;
  int speech_to_sil_time_thres;
  float speech_2_noise_ratio;
  int do_extend;
  int lookback_time_start_point;
  int lookahead_time_end_point;
  int max_single_segment_time;
  int nn_eval_block_size;
  int dcd_block_size;
  float snr_thres;
  int noise_frame_num_used_for_snr;
  float decibel_thres;
  float speech_noise_thres;
  float fe_prior_thres;
  int silence_pdf_num;
  std::vector<int> sil_pdf_ids;
  float speech_noise_thresh_low;
  float speech_noise_thresh_high;
  bool output_frame_probs;
  int frame_in_ms;
  int frame_length_ms;

  explicit VADXOptions(
      int sr = 16000,
      int dm = static_cast<int>(VadDetectMode::kVadMutipleUtteranceDetectMode),
      int sm = 0, int mset = 800, int msst = 3000, bool dspd = true,
      bool depd = true, int wsm = 200, int ststh = 150, int sttsh = 150,
      float s2nr = 1.0, int de = 1, int lbtps = 200, int latsp = 100,
      int mss = 15000, int nebs = 8, int dbs = 4, float st = -100.0,
      int nfnus = 100, float dt = -100.0, float snt = 0.9, float fept = 1e-4,
      int spn = 1, std::vector<int> spids = {0}, float sntl = -0.1,
      float snth = 0.3, bool ofp = false, int fim = 10, int flm = 25);
};

class E2EVadSpeechBufWithDoa {
 public:
  int start_ms;
  int end_ms;
  std::vector<float> buffer;
  bool contain_seg_start_point;
  bool contain_seg_end_point;
  int doa;

  E2EVadSpeechBufWithDoa();
  void Reset();
};

class E2EVadFrameProb {
 public:
  double noise_prob;
  double speech_prob;
  double score;
  int frame_id;
  int frm_state;

  E2EVadFrameProb();
};

class WindowDetector {
 public:
  int window_size_ms;
  int sil_to_speech_time;
  int speech_to_sil_time;
  int frame_size_ms;
  int win_size_frame;
  int win_sum;
  std::vector<int> win_state;
  int cur_win_pos;
  FrameState pre_frame_state;
  FrameState cur_frame_state;
  int sil_to_speech_frmcnt_thres;
  int speech_to_sil_frmcnt_thres;
  int voice_last_frame_count;
  int noise_last_frame_count;
  int hydre_frame_count;

  WindowDetector(int window_size_ms, int sil_to_speech_time,
                 int speech_to_sil_time, int frame_size_ms);
  void Reset();
  int GetWinSize();
  AudioChangeState DetectOneFrame(FrameState frameState, int frame_count);
  int FrameSizeMs();
};

class E2EVadModel {
 public:
  E2EVadModel();

  std::vector<std::vector<int>> operator()(
      const std::vector<std::vector<float>> &score,
      const std::vector<float> &waveform, bool is_final = false,
      bool online = false, int max_end_sil = 800,
      int max_single_segment_time = 15000, float speech_noise_thres = 0.8,
      int sample_rate = 16000);

 private:
  VADXOptions vad_opts;
  WindowDetector windows_detector;
  bool is_final;
  int data_buf_start_frame;
  int frm_cnt;
  int latest_confirmed_speech_frame;
  int lastest_confirmed_silence_frame;
  int continous_silence_frame_count;
  VadStateMachine vad_state_machine;
  int confirmed_start_frame;
  int confirmed_end_frame;
  int number_end_time_detected;
  int sil_frame;
  std::vector<int> sil_pdf_ids;
  float noise_average_decibel;
  bool pre_end_silence_detected;
  bool next_seg;
  std::vector<E2EVadSpeechBufWithDoa> output_data_buf;
  int output_data_buf_offset;
  std::vector<E2EVadFrameProb> frame_probs;
  int max_end_sil_frame_cnt_thresh;
  float speech_noise_thres;
  std::vector<std::vector<float>> scores;
  int idx_pre_chunk;
  bool max_time_out;
  std::vector<float> decibel;
  int data_buf_size;
  int data_buf_all_size;
  std::vector<float> waveform;

  void AllResetDetection();
  void ResetDetection();
  void ComputeDecibel();
  void ComputeScores(const std::vector<std::vector<float>> &scores);
  void PopDataBufTillFrame(int frame_idx);
  void PopDataToOutputBuf(int start_frm, int frm_cnt,
                          bool first_frm_is_start_point,
                          bool last_frm_is_end_point,
                          bool end_point_is_sent_end);
  void OnSilenceDetected(int valid_frame);
  void OnVoiceDetected(int valid_frame);
  void OnVoiceStart(int start_frame, bool fake_result = false);
  void OnVoiceEnd(int end_frame, bool fake_result, bool is_last_frame);
  void MaybeOnVoiceEndIfLastFrame(bool is_final_frame, int cur_frm_idx);
  int GetLatency();
  int LatencyFrmNumAtStartPoint();
  FrameState GetFrameState(int t);
  int DetectCommonFrames();
  int DetectLastFrames();
  void DetectOneFrame(FrameState cur_frm_state, int cur_frm_idx,
                      bool is_final_frame);
};