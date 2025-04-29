#include "components/video_decoder/video_decoder_type.hpp"
#include "image_folder/image_folder_decoder.hpp"

#include "framework/utils/common_utils.hpp"
#include "framework/utils/tdl_log.hpp"
#if defined(__BM168X__) || defined(__BM1684__) || defined(__BM1684X__)
#include "opencv_decoder/opencv_decoder.hpp"
#else
// #include "components/video_decoder/vi_decoder/vi_decoder.hpp"
#endif

std::shared_ptr<VideoDecoder> VideoDecoderFactory::createVideoDecoder(
    VideoDecoderType type) {
  switch (type) {
    case VideoDecoderType::OPENCV:
#if defined(__BM168X__) || defined(__BM1684__) || defined(__BM1684X__)
      return std::make_shared<OpencvDecoder>();
#else
      LOGE("opencv decoder is not supported on this platform,%d",
           get_platform());
      return nullptr;
#endif
    case VideoDecoderType::IMAGE_FOLDER:
      return std::make_shared<ImageFolderDecoder>();
    default:
      return nullptr;
  }
}
