#ifndef VI_DECODER_HPP
#define VI_DECODER_HPP

#include <sys/prctl.h>
#include <atomic>
#include <thread>
#include "cvi_comm_vi.h"
#include "cvi_comm_video.h"
#include "cvi_comm_vpss.h"
#include "cvi_sns_ctrl.h"
#include "cvi_sys.h"
#include "cvi_vb.h"
#include "cvi_vi.h"
#include "cvi_vpss.h"
#include "video_decoder/video_decoder_type.hpp"

class ViDecoder : public VideoDecoder {
 public:
  ViDecoder();
  ~ViDecoder();

  int32_t init(const std::string &path,
               const std::map<std::string, int32_t> &config = {}) override;
  int32_t initialize(int32_t w = 1920, int32_t h = 1080,
                     ImageFormat image_fmt = ImageFormat::YUV420SP_VU,
                     int32_t vb_buffer_num = 3) override;
  int32_t read(std::shared_ptr<BaseImage> &image, int32_t vi_chn = 0) override;
  int32_t release(int32_t vi_chn = 0) override;

 private:
  bool isInitialized = false;
  std::vector<std::unique_ptr<MemoryBlock>> memory_blocks_;
  std::shared_ptr<BaseMemoryPool> memory_pool_ = nullptr;
  bool isMapped_ = false;
  void *addr_[3] = {nullptr};
  uint32_t image_length_[3] = {0};

  int32_t deinitialize();
};

#endif  // VI_DECODER_HPP