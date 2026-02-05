#include "audio/audio_player.hpp"
#include <cstring>
#include <iostream>
#include "audio/audio_system.hpp"
#include "cvi_audio.h"
#include "cvi_sys.h"

AudioPlayer::AudioPlayer() {}

AudioPlayer::~AudioPlayer() { Deinit(); }

int AudioPlayer::Init(const Config& config) {
  if (initialized_) return 0;
  config_ = config;

  // Use AudioSystem to manage CVI_AUDIO_INIT
  if (AudioSystem::GetInstance().RequestInit() != 0) {
    return -1;
  }

  // AO (Audio Output) Config
  AIO_ATTR_S AudoutAttr;
  memset(&AudoutAttr, 0, sizeof(AudoutAttr));
  AudoutAttr.u32ChnCnt = config_.channels;
  AudoutAttr.enSamplerate = (AUDIO_SAMPLE_RATE_E)config_.sample_rate;
  AudoutAttr.enSoundmode =
      (config_.channels == 1) ? AUDIO_SOUND_MODE_MONO : AUDIO_SOUND_MODE_STEREO;
  AudoutAttr.enWorkmode = AIO_MODE_I2S_MASTER;
  AudoutAttr.u32EXFlag = 0;
  AudoutAttr.u32FrmNum = 10;
  AudoutAttr.enBitwidth = AUDIO_BIT_WIDTH_16;
  AudoutAttr.u32PtNumPerFrm = config_.period_size;
  AudoutAttr.u32ClkSel = 0;
  AudoutAttr.enI2sType = AIO_I2STYPE_INNERCODEC;

  int s32Ret = CVI_AO_SetPubAttr(0, &AudoutAttr);
  if (s32Ret != 0) {
    std::cerr << "[AudioPlayer] CVI_AO_SetPubAttr failed: " << s32Ret
              << std::endl;
    return -1;
  }

  CVI_AO_Enable(0);
  CVI_AO_EnableChn(0, 0);
  CVI_AO_SetVolume(0, config_.volume);

  initialized_ = true;
  return 0;
}

int AudioPlayer::Deinit() {
  if (!initialized_) return 0;

  CVI_AO_DisableChn(0, 0);
  CVI_AO_Disable(0);

  AudioSystem::GetInstance().Release();

  initialized_ = false;
  return 0;
}

int AudioPlayer::SendFrame(const std::vector<uint8_t>& data, int timeout_ms) {
  return SendFrame(data.data(), data.size(), timeout_ms);
}

int AudioPlayer::SendFrame(const uint8_t* data, size_t size, int timeout_ms) {
  if (!initialized_) return -1;
  if (size == 0) return 0;

  AUDIO_FRAME_S stFrame;
  memset(&stFrame, 0, sizeof(stFrame));

  stFrame.u64VirAddr[0] = (uint8_t*)
      data;  // API usually takes non-const but treats as const for send
  stFrame.u32Len = config_.period_size;  // Or calculated from size?
  // CVI_AO_SendFrame expects u32Len to be number of sample points per frame?
  // DoubaoClient sets stFrame.u32Len = AUDIO_PERIOD_SIZE;
  // And sends `frame_bytes` which is AUDIO_PERIOD_SIZE * 2.
  // So u32Len is sample count.

  // Check if size matches
  size_t expected_size = config_.period_size * 2 * config_.channels;
  if (size != expected_size) {
    // Warning? Or calculate u32Len?
    // Let's trust the user to send correct frame size matching period_size,
    // or we calculate u32Len from size.
    stFrame.u32Len = size / (2 * config_.channels);
  }

  stFrame.u64TimeStamp = 0;
  stFrame.enSoundmode =
      (config_.channels == 1) ? AUDIO_SOUND_MODE_MONO : AUDIO_SOUND_MODE_STEREO;
  stFrame.enBitwidth = AUDIO_BIT_WIDTH_16;

  CVI_S32 ret = CVI_AO_SendFrame(0, 0, &stFrame, timeout_ms);
  if (ret != 0) {
    return -1;
  }
  return 0;
}
