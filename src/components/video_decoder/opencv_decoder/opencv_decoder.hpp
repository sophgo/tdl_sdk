#ifndef OPENCV_DECODER_HPP
#define OPENCV_DECODER_HPP

#include "opencv2/opencv.hpp"
#include "video_decoder/video_decoder_type.hpp"

class OpencvDecoder : public VideoDecoder {
 public:
  OpencvDecoder();
  ~OpencvDecoder();

  int32_t init(const std::string &path,
               const std::map<std::string, int> &config = {}) override;
  int32_t read(std::shared_ptr<BaseImage> &image, int vi_chn = 0) override;

 private:
  cv::VideoCapture capture_;
};

#endif
