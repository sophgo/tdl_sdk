#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

class AudioPlayer {
 public:
  struct Config {
    int sample_rate = 16000;
    int channels = 1;
    int period_size = 640;
    int volume = 30;
  };

  AudioPlayer();
  ~AudioPlayer();

  int Init(const Config& config);
  int Deinit();

  // Returns 0 on success, -1 on failure
  // blocks up to timeout_ms
  int SendFrame(const uint8_t* data, size_t size, int timeout_ms = 1000);
  int SendFrame(const std::vector<uint8_t>& data, int timeout_ms = 1000);

 private:
  Config config_;
  bool initialized_ = false;
};
