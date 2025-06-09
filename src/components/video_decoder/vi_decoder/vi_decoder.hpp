#ifndef VI_DECODER_HPP
#define VI_DECODER_HPP

#include <sys/prctl.h>
#include <atomic>
#include <thread>
#include "cvi_comm_vi.h"
#include "cvi_comm_video.h"
#include "cvi_sns_ctrl.h"
#include "cvi_sys.h"
#include "cvi_vb.h"
#include "cvi_vi.h"
#include "video_decoder/video_decoder_type.hpp"

class ViDecoder : public VideoDecoder {
 private:
  bool isInitialized = false;
  std::vector<std::unique_ptr<MemoryBlock>> memory_blocks_;
  std::shared_ptr<BaseMemoryPool> memory_pool_ = nullptr;
  bool isMapped_ = false;
  void *addr_ = nullptr;
  uint32_t image_size_ = 0;

  int initialize();
  int deinitialize();

 public:
  ViDecoder();
  ~ViDecoder();

  int32_t init(const std::string &path,
               const std::map<std::string, int> &config = {}) override;
  int32_t read(std::shared_ptr<BaseImage> &image, int vi_chn = 0) override;
  int32_t read(VIDEO_FRAME_INFO_S *frame, int vi_chn);
  int32_t release(int vi_chn = 0) override;
  int32_t release(int vi_chn, VIDEO_FRAME_INFO_S *frame);
};

#endif  // VI_DECODER_HPP