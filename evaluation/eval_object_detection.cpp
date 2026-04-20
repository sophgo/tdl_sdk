#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "tdl_model_factory.hpp"
std::string get_result_save_path(const std::string& img_path_or_name,
                                 const std::string& res_root) {
  size_t last_slash = img_path_or_name.find_last_of('/');
  std::string dir_part, filename, filename_no_ext;

  // 拆分文件名和目录（区分两种情况）
  if (last_slash != std::string::npos) {
    // 若包含相对路径
    dir_part = img_path_or_name.substr(0, last_slash);
    filename = img_path_or_name.substr(last_slash + 1);
  } else {
    // 纯文件名
    dir_part = "";
    filename = img_path_or_name;
  }

  // 提取无后缀的文件名（两种情况通用）
  size_t dot = filename.find_last_of('.');
  filename_no_ext =
      (dot == std::string::npos) ? filename : filename.substr(0, dot);

  // 拼接结果路径（根据是否有目录决定是否创建子文件夹）
  std::string res_dir, res_file;
  if (!dir_part.empty()) {
    // 有目录：在res_root下创建对应子文件夹
    res_dir = res_root + "/" + dir_part;
    res_file = res_dir + "/" + filename_no_ext + ".txt";

    // 递归创建目录（仅当有目录时才需要创建）
    std::string mkdir_cmd = "mkdir -p \"" + res_dir + "\"";
    int ret = system(mkdir_cmd.c_str());
    if (ret != 0) {
      std::cerr << "创建目录失败: " << res_dir << "\n";
      return "";
    }
  } else {
    // 无目录：直接在res_root下保存文件
    res_file = res_root + "/" + filename_no_ext + ".txt";
  }

  return res_file;
}

void save_detection_results(const std::string& save_path,
                            const std::shared_ptr<ModelBoxInfo>& obj_meta) {
  std::ofstream file(save_path);
  if (file.is_open()) {
    for (const auto& bbox : obj_meta->bboxes) {
      file << bbox.class_id << " " << bbox.x1 << " " << bbox.y1 << " "
           << bbox.x2 << " " << bbox.y2 << " " << bbox.score << "\n";
    }
  }
}

void bench_mark_all(const std::string& bench_path,
                    const std::string& image_root, const std::string& res_path,
                    std::shared_ptr<BaseModel> model /* 仍按值传入 */) {
  std::ifstream bench_fstream(bench_path);
  if (!bench_fstream.is_open()) {
    std::cerr << "打开 benchmark 文件失败: " << bench_path << "\n";
    return;
  }

  std::string img_rel_path;
  while (bench_fstream >> img_rel_path) {
    const std::string img_path = image_root + "/" + img_rel_path;
    std::cout << "Process: " << img_path << "\n"
              << "\n";
    std::shared_ptr<BaseImage> image = ImageFactory::readImage(img_path);
    if (!image) {
      std::cerr << "读取图像失败: " << img_path << "\n";
      continue;
    }
    std::vector<std::shared_ptr<BaseImage>> inputs{image};
    std::vector<std::shared_ptr<ModelOutputInfo>> outputs;
    model->inference(inputs, outputs);

    for (auto& out : outputs) {
      if (out->getType() != ModelOutputType::OBJECT_DETECTION) continue;
      std::shared_ptr<ModelBoxInfo> obj_meta =
          std::static_pointer_cast<ModelBoxInfo>(out);

      std::string save_path = get_result_save_path(img_rel_path, res_path);
      if (!save_path.empty()) {
        save_detection_results(save_path, obj_meta);
      }
    }
    inputs.clear();
    outputs.clear();
  }
  model.reset();
}

int main(int argc, char* argv[]) {
  if (argc != 6 && argc != 7) {
    printf(
        "\nUsage: %s MODEL_NAME MODEL_DIR BENCH_PATH IMAGE_ROOT RES_PATH "
        "[CONF_THRESHOLD] "
        "[NMS_THRESHOLD]\n\n"
        "\tMODEL_NAME, detection model name should be one of:\n"
        "\t  General models:\n"
        "\t    yolov6, yolov8, yolov10\n"
        "\t  COCO models:\n"
        "\t    yolov6-coco80, yolov8-coco80, yolov10-coco80\n"
        "\t  MobileDetV2 models:\n"
        "\t    mobiledetv2-pedestrian\n"
        "\t  Specialized YOLOv8 models:\n"
        "\t    yolov8n-hand, yolov8n-pet-person, yolov8n-person-vehicle\n"
        "\t    yolov8n-hand-face-person, yolov8n-head-person\n"
        "\t    yolov8n-head-hardhat, yolov8n-fire-smoke, yolov8n-fire\n"
        "\t    yolov8n-head-shoulder, yolov8n-license-plate\n"
        "\t    yolov8n-traffic-light, yolov8n-monitor-person\n"
        "\tMODEL_DIR, store cvimodel or bmodel path\n"
        "\tBENCH_PATH, txt for storing image names(relative path + file name)\n"
        "\tIMAGE_ROOT, store images path(root directory)\n"
        "\tRES_PATH, save result path(automatically create subfolders)\n"
        "\tCONF_THRESHOLD (optional), confidence threshold (default: 0.5)\n",
        argv[0]);
    return -1;
  }

  std::string model_name = argv[1];
  std::string model_dir = argv[2];
  std::string bench_path = argv[3];
  std::string image_root = argv[4];
  std::string res_path = argv[5];

  float conf_threshold = 0.01;
  if (argc > 6) {
    conf_threshold = std::stof(argv[6]);
  }

  TDLModelFactory& model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();

  std::shared_ptr<BaseModel> model;
  struct stat path_stat;
  if (stat(model_dir.c_str(), &path_stat) == 0) {
    if (S_ISDIR(path_stat.st_mode)) {  // model_dir是文件夹：原有getModel调用
      model_factory.setModelDir(model_dir);
      model = model_factory.getModel(model_name);
    } else if (S_ISREG(
                   path_stat.st_mode)) {  // model_dir是绝对路径：新getModel调用
      model = model_factory.getModel(model_name, model_dir);
    } else {
      printf("Error: MODEL_DIR is neither dir nor file\n");
      return -1;
    }
  } else {
    printf("Error: Cannot access MODEL_DIR: %s\n", model_dir.c_str());
    return -1;
  }

  if (!model) {
    printf("Failed to create model\n");
    return -1;
  }

  model->setModelThreshold(conf_threshold);

  std::cout << "model opened:" << model_name << std::endl;
  bench_mark_all(bench_path, image_root, res_path, model);

  return 0;
}