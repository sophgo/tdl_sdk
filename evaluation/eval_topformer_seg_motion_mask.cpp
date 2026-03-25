#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "tdl_model_factory.hpp"

struct ImageItem {
  std::string full_path;
  std::string relative_path;
};

static std::string join_paths(const std::string &base,
                              const std::string &relative) {
  if (base.empty()) return relative;
  if (relative.empty()) return base;
  if (relative[0] == '/') return relative;
  if (base.back() == '/') return base + relative;
  return base + "/" + relative;
}

static bool file_exists(const std::string &path) {
  struct stat buffer;
  return stat(path.c_str(), &buffer) == 0;
}

static bool is_directory(const std::string &path) {
  struct stat buffer;
  if (stat(path.c_str(), &buffer) != 0) return false;
  return S_ISDIR(buffer.st_mode);
}

static bool create_directory(const std::string &path) {
  if (path.empty()) return false;
  if (file_exists(path)) return is_directory(path);

  std::string temp_path = path;
  size_t pos = 0;
  do {
    pos = temp_path.find_first_of('/', pos + 1);
    std::string sub_dir = temp_path.substr(0, pos);
    if (sub_dir.empty()) continue;
    if (!file_exists(sub_dir) && mkdir(sub_dir.c_str(), 0777) != 0) {
      return false;
    }
  } while (pos != std::string::npos);
  return true;
}

static std::string trim(const std::string &s) {
  size_t left = s.find_first_not_of(" \t\n\r");
  if (left == std::string::npos) return "";
  size_t right = s.find_last_not_of(" \t\n\r");
  return s.substr(left, right - left + 1);
}

static std::string get_filename_without_ext(const std::string &filepath) {
  size_t slash_pos = filepath.find_last_of('/');
  std::string filename = (slash_pos == std::string::npos)
                             ? filepath
                             : filepath.substr(slash_pos + 1);
  size_t dot_pos = filename.find_last_of('.');
  if (dot_pos == std::string::npos) return filename;
  return filename.substr(0, dot_pos);
}

static std::string get_path_without_ext(const std::string &path) {
  size_t dot_pos = path.find_last_of('.');
  if (dot_pos == std::string::npos) return path;
  size_t slash_pos = path.find_last_of('/');
  if (slash_pos != std::string::npos && dot_pos < slash_pos) return path;
  return path.substr(0, dot_pos);
}

static std::string get_parent_dir(const std::string &path) {
  size_t slash_pos = path.find_last_of('/');
  if (slash_pos == std::string::npos) return "";
  return path.substr(0, slash_pos);
}

static std::string strip_root_prefix(const std::string &full_path,
                                     const std::string &root_path) {
  if (root_path.empty()) return full_path;
  std::string root = root_path;
  if (!root.empty() && root.back() == '/') {
    root.pop_back();
  }
  if (full_path == root) return "";
  if (full_path.size() > root.size() &&
      full_path.compare(0, root.size(), root) == 0 &&
      full_path[root.size()] == '/') {
    return full_path.substr(root.size() + 1);
  }
  return full_path;
}

static std::vector<ImageItem> get_image_items_from_txt(
    const std::string &txt_path, const std::string &image_root) {
  std::vector<ImageItem> image_items;
  if (!file_exists(txt_path) || is_directory(txt_path)) {
    std::cerr << "Invalid txt path: " << txt_path << std::endl;
    return image_items;
  }

  std::ifstream file(txt_path);
  if (!file.is_open()) {
    std::cerr << "Failed to open txt file: " << txt_path << std::endl;
    return image_items;
  }

  std::string line;
  int line_num = 0;
  while (std::getline(file, line)) {
    line_num++;
    line = trim(line);
    if (line.empty() || line[0] == '#') continue;

    std::string full_path =
        (line[0] == '/') ? line : join_paths(image_root, line);
    std::string relative_path =
        (line[0] == '/') ? strip_root_prefix(line, image_root) : line;
    if (relative_path.empty()) {
      relative_path = get_filename_without_ext(full_path);
    }

    if (!file_exists(full_path) || is_directory(full_path)) {
      std::cerr << "Skip invalid image at " << txt_path << ":" << line_num
                << " -> " << full_path << std::endl;
      continue;
    }
    image_items.push_back({full_path, relative_path});
  }
  return image_items;
}

static bool save_motion_mask_png(
    const std::shared_ptr<ModelBoxSegmentationInfo> &box_meta,
    const std::string &output_path) {
  if (!box_meta) return false;
  if (box_meta->mask_width == 0 || box_meta->mask_height == 0) {
    // Cached warmup frame, no valid output yet.
    std::cout << "Invalid mask size for output: " << output_path << std::endl;
    return true;
  }
  const int out_w = static_cast<int>(box_meta->mask_width);
  const int out_h = static_cast<int>(box_meta->mask_height);
  cv::Mat mask_model(out_h, out_w, CV_8UC1, cv::Scalar(0));
  if (!box_meta->box_seg.empty() && box_meta->box_seg[0].mask != nullptr) {
    std::memcpy(mask_model.data, box_meta->box_seg[0].mask,
                static_cast<size_t>(out_w) * static_cast<size_t>(out_h));
  }

  std::string parent_dir = get_parent_dir(output_path);
  if (!parent_dir.empty() && !file_exists(parent_dir)) {
    if (!create_directory(parent_dir)) {
      std::cerr << "Failed to create output parent directory: " << parent_dir
                << std::endl;
      return false;
    }
  }

  if (!cv::imwrite(output_path, mask_model)) {
    std::cerr << "Failed to save png: " << output_path << std::endl;
    return false;
  }
  return true;
}

int main(int argc, char **argv) {
  if (argc < 6) {
    printf(
        "Usage: %s <model_id_name> <model_dir> <image_root> <output_dir> "
        "<txt_path1> [txt_path2 ...]\n",
        argv[0]);
    printf(
        "Example: %s TOPFORMER_SEG_MOTION ./models ./images ./output "
        "val_a.txt val_b.txt\n",
        argv[0]);
    return -1;
  }

  const std::string model_id_name = argv[1];
  const std::string model_dir = argv[2];
  const std::string image_root = argv[3];
  const std::string output_dir = argv[4];

  if (!file_exists(output_dir)) {
    if (!create_directory(output_dir)) {
      std::cerr << "Failed to create output directory: " << output_dir
                << std::endl;
      return -1;
    }
  }

  std::vector<std::string> txt_paths;
  for (int i = 5; i < argc; ++i) {
    txt_paths.push_back(argv[i]);
  }

  TDLModelFactory &model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();
  model_factory.setModelDir(model_dir);

  int total_images = 0;
  int total_success = 0;
  for (size_t dataset_idx = 0; dataset_idx < txt_paths.size(); ++dataset_idx) {
    const std::string &txt_path = txt_paths[dataset_idx];
    std::vector<ImageItem> image_items =
        get_image_items_from_txt(txt_path, image_root);
    if (image_items.empty()) {
      std::cerr << "[Dataset " << (dataset_idx + 1)
                << "] no valid images: " << txt_path << std::endl;
      continue;
    }

    std::cout << "\n[Dataset " << (dataset_idx + 1) << "/" << txt_paths.size()
              << "] start: " << txt_path << ", images: " << image_items.size()
              << std::endl;

    // Recreate model for each dataset to clear internal cached frames.
    std::shared_ptr<BaseModel> model_seg =
        model_factory.getModel(model_id_name);
    if (!model_seg) {
      std::cerr << "Failed to create model: " << model_id_name << std::endl;
      return -1;
    }

    const std::map<std::string, float> infer_params = {{"with_mask", 1.0f}};

    for (size_t i = 0; i < image_items.size(); ++i) {
      const std::string &image_path = image_items[i].full_path;
      const std::string &relative_path = image_items[i].relative_path;
      total_images++;

      std::shared_ptr<BaseImage> image =
          ImageFactory::readImage(image_path, ImageFormat::GRAY);
      if (!image) {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        continue;
      }

      std::shared_ptr<ModelOutputInfo> out_data =
          std::make_shared<ModelBoxSegmentationInfo>();
      int32_t ret = model_seg->inference(image, out_data, infer_params);
      if (ret != 0 || out_data == nullptr ||
          out_data->getType() !=
              ModelOutputType::OBJECT_DETECTION_WITH_SEGMENTATION) {
        std::cerr << "Failed inference: " << image_path << std::endl;
        continue;
      }

      std::shared_ptr<ModelBoxSegmentationInfo> box_meta =
          std::static_pointer_cast<ModelBoxSegmentationInfo>(out_data);
      std::string output_mask_path = join_paths(
          output_dir, get_path_without_ext(relative_path) + "_mask.png");
      if (save_motion_mask_png(box_meta, output_mask_path)) {
        total_success++;
      }

      if ((i + 1) % 100 == 0 || (i + 1) == image_items.size()) {
        std::cout << "  progress: " << (i + 1) << "/" << image_items.size()
                  << std::endl;
      }
    }

    // Explicitly release model so next dataset starts with fresh cache state.
    model_seg.reset();
    std::cout << "[Dataset " << (dataset_idx + 1) << "] done: " << txt_path
              << std::endl;
  }

  std::cout << "\nAll datasets done. Saved " << total_success << "/"
            << total_images << " mask png files to: " << output_dir
            << std::endl;
  return 0;
}
