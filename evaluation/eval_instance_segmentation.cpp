#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>  // 确保包含OpenCV头文件
#include <sstream>
#include <string>
#include <vector>
#include "tdl_model_factory.hpp"

// 使用新的轮廓坐标还原逻辑
void visualize_maskOutlinePoint(
    std::shared_ptr<ModelBoxSegmentationInfo> obj_meta, uint32_t image_height,
    uint32_t image_width) {
  int proto_h = obj_meta->mask_height;
  int proto_w = obj_meta->mask_width;

  for (uint32_t i = 0; i < obj_meta->box_seg.size(); i++) {
    cv::Mat src(proto_h, proto_w, CV_8UC1, obj_meta->box_seg[i].mask,
                proto_w * sizeof(uint8_t));

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    // search for contours
    cv::findContours(src, contours, hierarchy, cv::RETR_TREE,
                     cv::CHAIN_APPROX_SIMPLE);
    // find the longest contour
    int longest_index = -1;
    size_t max_length = 0;
    for (size_t j = 0; j < contours.size(); j++) {  // 修改变量名冲突
      if (contours[j].size() > max_length) {
        max_length = contours[j].size();
        longest_index = j;
      }
    }
    if (longest_index >= 0 && max_length >= 1) {
      float ratio_height = (proto_h / static_cast<float>(image_height));
      float ratio_width = (proto_w / static_cast<float>(image_width));
      int source_y_offset, source_x_offset;
      if (ratio_height > ratio_width) {
        source_x_offset = 0;
        source_y_offset = (proto_h - image_height * ratio_width) / 2;
      } else {
        source_x_offset = (proto_w - image_width * ratio_height) / 2;
        source_y_offset = 0;
      }
      int source_region_height = proto_h - 2 * source_y_offset;
      int source_region_width = proto_w - 2 * source_x_offset;
      // calculate scaling factor
      float height_scale = static_cast<float>(image_height) /
                           static_cast<float>(source_region_height);
      float width_scale = static_cast<float>(image_width) /
                          static_cast<float>(source_region_width);
      obj_meta->box_seg[i].mask_point_size = max_length;
      obj_meta->box_seg[i].mask_point =
          new float[2 * max_length];  // 移除多余的sizeof(float)
      size_t k = 0;                   // 修改变量名冲突
      for (const auto& point : contours[longest_index]) {
        obj_meta->box_seg[i].mask_point[2 * k] =
            (point.x - source_x_offset) * width_scale;
        obj_meta->box_seg[i].mask_point[2 * k + 1] =
            (point.y - source_y_offset) * height_scale;
        k++;
      }
    }
  }
}

// ✅ 关键修改：保存 class_id + confidence + 归一化轮廓点
void save_normalized_contours(
    std::shared_ptr<ModelBoxSegmentationInfo> obj_meta,
    const std::string& output_path) {
  std::ofstream outfile(output_path);
  if (!outfile.is_open()) {
    std::cerr << "Failed to open output file: " << output_path << std::endl;
    return;
  }

  outfile << std::fixed << std::setprecision(6);

  for (size_t j = 0; j < obj_meta->box_seg.size(); j++) {
    if (obj_meta->box_seg[j].mask_point_size >= 3 &&
        obj_meta->box_seg[j].mask_point != nullptr) {
      int class_id = static_cast<int>(obj_meta->box_seg[j].class_id);
      float confidence = obj_meta->box_seg[j].score;  // ✅ 加入置信度

      outfile << class_id << " " << confidence;

      for (uint32_t k = 0; k < obj_meta->box_seg[j].mask_point_size; k++) {
        // 归一化到 [0,1]
        float norm_x =
            obj_meta->box_seg[j].mask_point[2 * k] / obj_meta->image_width;
        float norm_y =
            obj_meta->box_seg[j].mask_point[2 * k + 1] / obj_meta->image_height;

        outfile << " " << norm_x << " " << norm_y;
      }
      outfile << "\n";
    }
  }
  outfile.close();

  // ✅ 释放动态分配的内存（防止内存泄漏）
  for (auto& seg : obj_meta->box_seg) {
    if (seg.mask_point) {
      delete[] seg.mask_point;
      seg.mask_point = nullptr;
      seg.mask_point_size = 0;
    }
  }
}

// 文件操作辅助函数（保留必要功能）
bool file_exists(const std::string& path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}

bool is_directory(const std::string& path) {
  struct stat buffer;
  if (stat(path.c_str(), &buffer) != 0) {
    return false;
  }
  return S_ISDIR(buffer.st_mode);
}

bool create_directory(const std::string& path) {
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

std::string get_file_extension(const std::string& filename) {
  size_t dot_pos = filename.find_last_of('.');
  if (dot_pos == std::string::npos) {
    return "";
  }
  std::string ext = filename.substr(dot_pos);
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  return ext;
}

std::string get_filename_without_ext(const std::string& filepath) {
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

std::string get_relative_path(const std::string& full_path,
                              const std::string& root_path) {
  if (full_path.find(root_path) == 0) {
    return full_path.substr(root_path.length());
  }
  return full_path;
}

// 从txt文件读取图片名称列表，结合image_root构建完整路径
std::vector<std::string> get_images_from_list(const std::string& image_root,
                                              const std::string& list_file) {
  std::vector<std::string> image_files;

  if (!file_exists(list_file) || is_directory(list_file)) {
    std::cerr << "List file does not exist or is a directory: " << list_file
              << std::endl;
    return image_files;
  }

  if (!file_exists(image_root) || !is_directory(image_root)) {
    std::cerr << "Image root directory does not exist or is not a directory: "
              << image_root << std::endl;
    return image_files;
  }

  std::ifstream infile(list_file);
  if (!infile.is_open()) {
    std::cerr << "Failed to open list file: " << list_file << std::endl;
    return image_files;
  }

  std::string line;
  int line_num = 0;

  while (std::getline(infile, line)) {
    line_num++;

    // 跳过空行和注释行
    if (line.empty() || line[0] == '#' || line[0] == '/') {
      continue;
    }

    // 去除首尾空格
    line.erase(line.find_last_not_of(" \t\n\r") + 1);
    line.erase(0, line.find_first_not_of(" \t\n\r"));

    if (line.empty()) {
      continue;
    }

    // 构建完整路径
    std::string full_path = image_root + "/" + line;

    // 检查文件是否存在
    if (file_exists(full_path) && !is_directory(full_path)) {
      image_files.push_back(full_path);
      std::cout << "Found image [" << line_num << "]: " << full_path
                << std::endl;
    } else {
      std::cerr << "Warning: Image not found at line " << line_num << ": "
                << full_path << std::endl;
    }
  }

  infile.close();
  return image_files;
}

int main(int argc, char** argv) {
  if (argc != 6) {
    printf(
        "Usage: %s <model_id_name> <model_dir> <image_list.txt> <image_root>"
        "<output_dir>\n",
        argv[0]);
    return -1;
  }

  std::string model_id_name = argv[1];
  std::string model_dir = argv[2];
  std::string image_list_file = argv[3];
  std::string image_root = argv[4];
  std::string output_dir = argv[5];

  if (!file_exists(output_dir)) {
    if (!create_directory(output_dir)) {
      std::cerr << "Failed to create output directory: " << output_dir
                << std::endl;
      return -1;
    }
  }

  // 初始化模型
  TDLModelFactory& model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();
  model_factory.setModelDir(model_dir);
  std::shared_ptr<BaseModel> model = model_factory.getModel(model_id_name);
  if (!model) {
    std::cerr << "Failed to create model" << std::endl;
    return -1;
  }
  model->setModelThreshold(0.001f);  // 保留低分检测，由评估脚本过滤

  std::vector<std::string> image_files =
      get_images_from_list(image_root, image_list_file);
  if (image_files.empty()) {
    std::cerr << "No valid images found." << std::endl;
    return -1;
  }

  std::cout << "Processing " << image_files.size() << " images...\n";

  for (size_t idx = 0; idx < image_files.size(); ++idx) {
    const std::string& img_path = image_files[idx];
    std::string base_name = get_filename_without_ext(img_path);
    std::cout << "[" << (idx + 1) << "/" << image_files.size() << "] "
              << base_name << std::endl;

    auto image = ImageFactory::readImage(img_path);
    if (!image) {
      std::cerr << "Skip: failed to read " << img_path << std::endl;
      continue;
    }

    std::vector<std::shared_ptr<ModelOutputInfo>> outputs;
    std::vector<std::shared_ptr<BaseImage>> inputs = {image};
    model->inference(inputs, outputs);

    if (outputs.empty()) continue;

    auto obj_meta =
        std::static_pointer_cast<ModelBoxSegmentationInfo>(outputs[0]);
    uint32_t img_w = image->getWidth();
    uint32_t img_h = image->getHeight();

    // 保存图像尺寸用于后续归一化
    obj_meta->image_width = img_w;
    obj_meta->image_height = img_h;

    // 使用新的轮廓提取函数
    visualize_maskOutlinePoint(obj_meta, img_h, img_w);

    std::string out_txt = output_dir + "/" + base_name + ".txt";
    save_normalized_contours(obj_meta, out_txt);  // ✅ 自动释放内存
  }

  std::cout << "\n✅ All done. Output saved to: " << output_dir << std::endl;
  return 0;
}