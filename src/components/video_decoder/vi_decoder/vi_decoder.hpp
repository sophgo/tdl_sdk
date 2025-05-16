#ifndef VI_DECODER_HPP
#define VI_DECODER_HPP

#include <sys/prctl.h>
#include <atomic>
#include <thread>
#include "components/video_decoder/video_decoder_type.hpp"

class ViDecoder : public VideoDecoder {
 private:
  bool isInitialized = false;
  std::vector<std::unique_ptr<MemoryBlock>> memory_blocks_;
  std::shared_ptr<BaseMemoryPool> memory_pool_ = nullptr;

  int initialize();
  int deinitialize();

 public:
  ViDecoder();
  ~ViDecoder();

  int32_t init(const std::string &path,
               const std::map<std::string, int> &config = {}) override;
  int32_t read(std::shared_ptr<BaseImage> &image, int vi_chn = 0) override;
  int32_t release(int vi_chn = 0) override;
};

#endif  // VI_DECODER_HPP