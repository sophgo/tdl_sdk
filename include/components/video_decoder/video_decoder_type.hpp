#ifndef VIDEO_DECODER_TYPE_HPP
#define VIDEO_DECODER_TYPE_HPP
#include <map>
#include "framework/image/base_image.hpp"

enum class VideoDecoderType {
  UNKNOWN,
  VI,
  OPENCV,
  IMAGE_FOLDER

};

class VideoDecoder {
 public:
  VideoDecoder() {}
  virtual ~VideoDecoder() {}

  virtual int32_t init(const std::string &path,
                       const std::map<std::string, int> &config = {}) = 0;
  virtual int32_t read(std::shared_ptr<BaseImage> &image) = 0;
  uint64_t getFrameId() { return frame_id_; }

 protected:
  VideoDecoderType type_ = VideoDecoderType::UNKNOWN;
  std::string path_ = "";
  uint64_t frame_id_ = 0;
};

class VideoDecoderFactory {
 public:
  static std::shared_ptr<VideoDecoder> createVideoDecoder(
      VideoDecoderType type);
};

#endif