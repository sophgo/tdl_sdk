#pragma once
#include <cstdint>
#include <vector>

class AudioCapture {
 public:
  struct Config {
    int sample_rate = 16000;
    int channels = 1;
    int period_size = 640;
    int volume = 15;
  };

  AudioCapture();
  ~AudioCapture();

  int Init(const Config& config);
  int Deinit();

  // Returns 0 on success, -1 on failure/no data
  int GetFrame(std::vector<uint8_t>& buffer);

 private:
  Config config_;
  bool initialized_ = false;
};
