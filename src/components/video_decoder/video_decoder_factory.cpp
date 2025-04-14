#include "components/video_decoder/video_decoder_type.hpp"
#include "image_folder/image_folder_decoder.hpp"

#include "framework/utils/common_utils.hpp"
#include "framework/utils/tdl_log.hpp"
#if defined(__BM168X__) || defined(__BM1684__) || defined(__BM1684X__)
#include "opencv_decoder/opencv_decoder.hpp"
#elif defined(__CV180X__) || defined(__CV181X__) || defined(__CV184X__) || defined(__CV186X__)
#include "vi_decoder/vi_decoder.hpp"
#endif

std::shared_ptr<VideoDecoder> VideoDecoderFactory::createVideoDecoder(
    VideoDecoderType type) {
  switch (type) {
    case VideoDecoderType::OPENCV:
#if defined(__BM168X__) || defined(__BM1684__) || defined(__BM1684X__)
      return std::make_shared<OpencvDecoder>();
#else
      LOGE("opencv decoder is not supported on this platform,%d", get_platform());
      return nullptr;
#endif
    case VideoDecoderType::IMAGE_FOLDER:
      return std::make_shared<ImageFolderDecoder>();
    case VideoDecoderType::VI:
#if defined(__CV180X__) || defined(__CV181X__) || defined(__CV184X__) || defined(__CV186X__)
      return std::make_shared<ViDecoder>();
#else
      printf("vi decoder is not supported on this platform,%d", get_platform());
      return nullptr;
#endif
    default:
      return nullptr;
  }
}
