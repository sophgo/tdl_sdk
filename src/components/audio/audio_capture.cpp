#include "audio/audio_capture.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
#include "audio/audio_system.hpp"
#include "cvi_audio.h"
#include "cvi_sys.h"

AudioCapture::AudioCapture() {}

AudioCapture::~AudioCapture() { Deinit(); }

int AudioCapture::Init(const Config& config) {
  if (initialized_) return 0;
  config_ = config;

  // Use AudioSystem to manage CVI_AUDIO_INIT
  if (AudioSystem::GetInstance().RequestInit() != 0) {
    return -1;
  }

  AIO_ATTR_S AudinAttr;
  memset(&AudinAttr, 0, sizeof(AudinAttr));
  AudinAttr.enSamplerate = (AUDIO_SAMPLE_RATE_E)config_.sample_rate;
  AudinAttr.u32ChnCnt = config_.channels;
  AudinAttr.enSoundmode =
      (config_.channels == 1) ? AUDIO_SOUND_MODE_MONO : AUDIO_SOUND_MODE_STEREO;
  AudinAttr.enBitwidth =
      AUDIO_BIT_WIDTH_16;  // Assuming 16-bit for now as per DoubaoClient
  AudinAttr.enWorkmode = AIO_MODE_I2S_MASTER;
  AudinAttr.u32EXFlag = 0;
  AudinAttr.u32FrmNum = 10;  // Buffer depth
  AudinAttr.u32PtNumPerFrm = config_.period_size;
  AudinAttr.u32ClkSel = 0;
  AudinAttr.enI2sType = AIO_I2STYPE_INNERCODEC;

  int s32Ret = CVI_AI_SetPubAttr(0, &AudinAttr);
  if (s32Ret != 0) {
    std::cerr << "[AudioCapture] CVI_AI_SetPubAttr failed: " << s32Ret
              << std::endl;
    return -1;
  }

  CVI_AI_Enable(0);
  CVI_AI_EnableChn(0, 0);  // Dev 0, Chn 0
  CVI_AI_SetVolume(0, config_.volume);

  initialized_ = true;
  return 0;
}

int AudioCapture::Deinit() {
  if (!initialized_) return 0;

  CVI_AI_DisableChn(0, 0);
  CVI_AI_Disable(0);

  AudioSystem::GetInstance().Release();

  initialized_ = false;
  return 0;
}

int AudioCapture::GetFrame(std::vector<uint8_t>& buffer) {
  if (!initialized_) return -1;

  AUDIO_FRAME_S stFrame;
  AEC_FRAME_S stAecFrm;
  memset(&stFrame, 0, sizeof(stFrame));
  memset(&stAecFrm, 0, sizeof(stAecFrm));

  // Non-blocking call? DoubaoClient used -1 (block?)
  // CVI_AI_GetFrame(0, 0, &stFrame, &stAecFrm, -1);
  // Let's use -1 as per DoubaoClient
  CVI_S32 s32Ret = CVI_AI_GetFrame(0, 0, &stFrame, &stAecFrm, -1);
  if (s32Ret == 0) {
    int frame_size = config_.period_size * 2;  // 16-bit = 2 bytes * period_size
    if (config_.channels == 2) frame_size *= 2;

    if (buffer.size() != (size_t)frame_size) {
      buffer.resize(frame_size);
    }

    // Copy data
    // Assuming stFrame.u64VirAddr[0] is valid
    memcpy(buffer.data(), (uint8_t*)stFrame.u64VirAddr[0], frame_size);

    CVI_AI_ReleaseFrame(0, 0, &stFrame, &stAecFrm);
    return 0;
  }

  return -1;
}
