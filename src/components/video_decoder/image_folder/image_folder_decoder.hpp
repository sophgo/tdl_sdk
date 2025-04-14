#ifndef IMAGE_FOLDER_DECODER_HPP
#define IMAGE_FOLDER_DECODER_HPP

#include "components/video_decoder/video_decoder_type.hpp"

class ImageFolderDecoder : public VideoDecoder {
 public:
  ImageFolderDecoder();
  ~ImageFolderDecoder();

  int32_t init(const std::string &path,
               const std::map<std::string, int> &config = {}) override;
  int32_t read(std::shared_ptr<BaseImage> &image, int vi_chn = 0) override;

 private:
  uint32_t image_index_ = 0;
  std::vector<std::string> image_paths_;
  std::string default_image_ext_ = ".jpg";
  bool is_loop_ = false;
};

#endif
