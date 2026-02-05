#pragma once

class AudioSystem {
 public:
  static AudioSystem& GetInstance();

  // Returns 0 on success
  int RequestInit();
  int Release();

 private:
  AudioSystem() = default;
  ~AudioSystem() = default;
  AudioSystem(const AudioSystem&) = delete;
  AudioSystem& operator=(const AudioSystem&) = delete;

  int ref_count_ = 0;
};
