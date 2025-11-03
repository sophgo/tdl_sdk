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

void save_detection_results(
    const std::string& save_path,
    const std::shared_ptr<ModelBoxLandmarkInfo>& obj_meta) {
  std::ofstream file(save_path);
  if (file.is_open()) {
    for (const auto& bbox : obj_meta->box_landmarks) {
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

  std::string image_name;
  while (bench_fstream >> image_name) {
    const std::string img_path = image_root + image_name;
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
      std::shared_ptr<ModelBoxLandmarkInfo> face_meta =
          std::static_pointer_cast<ModelBoxLandmarkInfo>(out);

      size_t dot = image_name.find_last_of('.');
      std::string base =
          (dot == std::string::npos ? image_name : image_name.substr(0, dot));
      save_detection_results(res_path + base + ".txt", face_meta);
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
        "\t    scrfd-face\n"
        "\tMODEL_DIR, store cvimodel or bmodel path\n"
        "\tBENCH_PATH, txt for storing image names\n"
        "\tIMAGE_ROOT, store images path\n"
        "\tRES_PATH, save result path\n"
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

  bench_mark_all(bench_path, image_root, res_path, model);

  return 0;
}