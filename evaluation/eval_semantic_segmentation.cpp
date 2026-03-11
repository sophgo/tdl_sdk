#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "tdl_model_factory.hpp"

std::string join_paths(const std::string &base, const std::string &relative) {
  if (base.empty()) return relative;
  if (relative.empty()) return base;

  if (!relative.empty() && relative[0] == '/') {
    return relative;
  }

  if (base.back() == '/') {
    return base + relative;
  } else {
    return base + "/" + relative;
  }
}

void save_mask_for_evaluation(std::shared_ptr<ModelSegmentationInfo> seg_meta,
                              const cv::Size &original_size,
                              const std::string &output_path) {
  uint32_t output_width = seg_meta->output_width;
  uint32_t output_height = seg_meta->output_height;

  // 创建模型输出尺寸的掩码
  cv::Mat mask(output_height, output_width, CV_8UC1, seg_meta->class_id);

  cv::Mat mask_original_size;
  cv::resize(mask, mask_original_size, original_size, 0, 0, cv::INTER_NEAREST);

  cv::imwrite(output_path, mask_original_size);
}

bool file_exists(const std::string &path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}

bool is_directory(const std::string &path) {
  struct stat buffer;
  if (stat(path.c_str(), &buffer) != 0) {
    return false;
  }
  return S_ISDIR(buffer.st_mode);
}

bool create_directory(const std::string &path) {
  std::string temp_path = path;
  size_t pos = 0;

  do {
    pos = temp_path.find_first_of('/', pos + 1);
    std::string sub_dir = temp_path.substr(0, pos);

    if (!file_exists(sub_dir) && mkdir(sub_dir.c_str(), 0777) != 0) {
      return false;
    }
  } while (pos != std::string::npos);

  return true;
}

std::string get_file_extension(const std::string &filename) {
  size_t dot_pos = filename.find_last_of('.');
  if (dot_pos == std::string::npos) {
    return "";
  }
  std::string ext = filename.substr(dot_pos);
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  return ext;
}

std::string get_filename_without_ext(const std::string &filepath) {
  size_t slash_pos = filepath.find_last_of('/');
  std::string filename = (slash_pos == std::string::npos)
                             ? filepath
                             : filepath.substr(slash_pos + 1);

  size_t dot_pos = filename.find_last_of('.');
  if (dot_pos == std::string::npos) {
    return filename;
  }
  return filename.substr(0, dot_pos);
}

std::vector<std::string> get_image_files_from_txt(
    const std::string &txt_path, const std::string &image_root = "") {
  std::vector<std::string> image_files;

  if (!file_exists(txt_path) || is_directory(txt_path)) {
    std::cerr << "Input txt file does not exist or is a directory: " << txt_path
              << std::endl;
    return image_files;
  }

  std::ifstream file(txt_path);
  if (!file.is_open()) {
    std::cerr << "Failed to open txt file: " << txt_path << std::endl;
    return image_files;
  }

  std::string line;
  while (std::getline(file, line)) {
    line.erase(line.find_last_not_of(" \t\n\r") + 1);
    line.erase(0, line.find_first_not_of(" \t\n\r"));

    if (line.empty()) {
      continue;
    }

    // 拼接完整路径
    std::string full_path =
        image_root.empty() ? line : join_paths(image_root, line);

    if (file_exists(full_path) && !is_directory(full_path)) {
      image_files.push_back(full_path);
    } else {
      std::cerr << "Invalid or non-existent file path: " << full_path
                << " (from: " << line << ")" << std::endl;
    }
  }

  file.close();
  return image_files;
}

cv::Size get_image_original_size(const std::string &image_path) {
  cv::Mat img = cv::imread(image_path);
  if (img.empty()) {
    return cv::Size(0, 0);
  }
  return img.size();
}

int main(int argc, char **argv) {
  if (argc != 6) {
    printf(
        "Usage: %s <model_id_name> <model_dir> <image_root> <txt_path> "
        "<output_dir> \n",
        argv[0]);
    printf(
        "       Processes images listed in txt_path (with image_root for "
        "relative paths) and saves results to output_dir\n");
    return -1;
  }
  std::string model_id_name = argv[1];
  std::string model_dir = argv[2];
  std::string txt_path = argv[3];
  std::string image_root = argv[4];
  std::string output_dir = argv[5];

  if (!file_exists(output_dir)) {
    create_directory(output_dir);
  }

  TDLModelFactory &model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();
  model_factory.setModelDir(model_dir);
  std::shared_ptr<BaseModel> model_seg = model_factory.getModel(model_id_name);
  if (!model_seg) {
    printf("Failed to create model_seg\n");
    return -1;
  }

  std::vector<std::string> image_files =
      get_image_files_from_txt(txt_path, image_root);
  if (image_files.empty()) {
    std::cerr << "No valid image files found in txt file: " << txt_path
              << std::endl;
    return -1;
  }

  std::cout << "Found " << image_files.size()
            << " valid image files to process..." << std::endl;

  for (size_t i = 0; i < image_files.size(); ++i) {
    const std::string &image_path = image_files[i];
    std::string filename = get_filename_without_ext(image_path);

    std::cout << "Processing [" << (i + 1) << "/" << image_files.size()
              << "]: " << image_path << std::endl;

    cv::Size original_size = get_image_original_size(image_path);
    if (original_size.width == 0 || original_size.height == 0) {
      std::cerr << "Failed to get original image size: " << image_path
                << std::endl;
      continue;
    }
    std::cout << "  - Original image size: " << original_size.width << "x"
              << original_size.height << std::endl;

    std::shared_ptr<BaseImage> image = ImageFactory::readImage(image_path);
    if (!image) {
      std::cerr << "Failed to read image: " << image_path << std::endl;
      continue;
    }

    std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
    std::vector<std::shared_ptr<BaseImage>> input_images = {image};

    model_seg->inference(input_images, out_datas);
    for (size_t j = 0; j < out_datas.size(); ++j) {
      std::shared_ptr<ModelSegmentationInfo> seg_meta =
          std::static_pointer_cast<ModelSegmentationInfo>(out_datas[j]);

      std::cout << "  - Model output size: " << seg_meta->output_width << "x"
                << seg_meta->output_height << std::endl;
      std::string output_mask_path = output_dir + "/" + filename + "_mask.png";
      save_mask_for_evaluation(seg_meta, original_size, output_mask_path);
      std::cout << "  - Saved evaluation mask (original size): "
                << output_mask_path << std::endl;
    }
  }

  std::cout << "Batch processing completed successfully!" << std::endl;
  return 0;
}