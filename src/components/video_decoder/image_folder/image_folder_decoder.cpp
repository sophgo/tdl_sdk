#include "image_folder/image_folder_decoder.hpp"

#include <dirent.h>
#include <sys/stat.h>
#include <algorithm>
#include <iostream>
#include <string>

bool is_jpg(const std::string& filename, const std::string& img_ext) {
  auto pos = filename.rfind('.');
  if (pos == std::string::npos) return false;
  std::string ext = filename.substr(pos);
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  return ext == img_ext;
}

std::vector<std::string> scan_image_in_dir(const std::string& dir_path,
                                           const std::string& img_ext) {
  DIR* dir = opendir(dir_path.c_str());
  std::vector<std::string> image_paths;
  if (!dir) {
    perror("opendir");
    return image_paths;
  }
  int count = 0;
  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    // 跳过“.”、“..”
    if (entry->d_name[0] == '.') continue;

    std::string full = dir_path + "/" + entry->d_name;
    struct stat st;
    if (stat(full.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
      if (is_jpg(entry->d_name, img_ext)) {
        image_paths.push_back(full);
      }
    }
  }
  closedir(dir);
  return image_paths;
}

ImageFolderDecoder::ImageFolderDecoder() {
  type_ = VideoDecoderType::IMAGE_FOLDER;
}

ImageFolderDecoder::~ImageFolderDecoder() {
  image_paths_.clear();
  image_index_ = 0;
}

int32_t ImageFolderDecoder::init(const std::string& path,
                                 const std::map<std::string, int>& config) {
  path_ = path;
  image_paths_ = scan_image_in_dir(path, default_image_ext_);
  std::sort(image_paths_.begin(), image_paths_.end());
  image_index_ = 0;
  is_loop_ = config.find("is_loop") != config.end() ? (bool)config.at("is_loop")
                                                    : false;
  return 0;
}

int32_t ImageFolderDecoder::read(std::shared_ptr<BaseImage>& image,
                                 int vi_chn) {
  if (image_index_ >= image_paths_.size()) {
    if (is_loop_) {
      image_index_ = 0;
    } else {
      return -1;
    }
  }
  std::cout << "read image:" << image_paths_[image_index_] << std::endl;
  image = ImageFactory::readImage(image_paths_[image_index_]);
  image_index_++;
  frame_id_++;
  return 0;
}
