#include "opencv_decoder/opencv_decoder.hpp"
#include "framework/image/opencv_image.hpp"

OpencvDecoder::OpencvDecoder() { type_ = VideoDecoderType::OPENCV; }

OpencvDecoder::~OpencvDecoder() { capture_.release(); }

int32_t OpencvDecoder::init(const std::string &path,
                            const std::map<std::string, int> &config) {
  path_ = path;
  bool use_yuv = false;
  if (config.find("use_yuv") != config.end()) {
    use_yuv = (bool)config.at("use_yuv");
  }
  if (use_yuv) {
    capture_.open(path_);  //, cv::CAP_YUV);
  } else {
    capture_.open(path_);
  }
  if (!capture_.isOpened()) {
    return -1;
  }
  return 0;
}

int32_t OpencvDecoder::read(std::shared_ptr<BaseImage> &image, int vi_chn) {
  cv::Mat frame;
  capture_ >> frame;
  if (frame.empty()) {
    return -1;
  }

  // TODO(fuquan.ke) need to check image format
  image = std::make_shared<OpenCVImage>(frame, ImageFormat::BGR_PACKED);
  frame_id_++;
  return 0;
}
