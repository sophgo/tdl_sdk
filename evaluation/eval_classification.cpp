#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

void save_classification_results(
    std::ofstream& ofs,
    const std::vector<std::shared_ptr<ModelOutputInfo>>& out_datas,
    const std::string& image_name, int real_id) {
  std::stringstream res_ss;
  res_ss << image_name << ",";
  int predict_id = -1;
  for (const auto& out_data : out_datas) {
    if (out_data->getType() == ModelOutputType::CLASSIFICATION) {
      std::shared_ptr<ModelClassificationInfo> cls_meta =
          std::static_pointer_cast<ModelClassificationInfo>(out_data);
      predict_id = cls_meta->topk_class_ids[0];
      break;
    }
  }
  res_ss << predict_id << "," << real_id << "\n";
  ofs << res_ss.str();
}

void bench_mark_all(std::string bench_path, std::string image_root,
                    std::string res_path, std::shared_ptr<BaseModel> model) {
  std::fstream file(bench_path);
  if (!file.is_open()) {
    printf("can not open bench path %s\n", bench_path.c_str());
    return;
  }
  printf("open bench path %s success!\n", bench_path.c_str());
  std::string line;
  int cnt = 0;
  std::ofstream ofs(
      res_path, std::ios_base::app);  // 或者使用 std::ios_base::trunc 根据需求
  if (!ofs.is_open()) {
    std::cerr << "Failed to open result file: " << res_path << std::endl;
    return;
  }
  while (getline(file, line)) {
    if (!line.empty()) {
      std::stringstream ss(line);
      std::string image_name;
      while (ss >> image_name) {
        size_t first_slash = image_name.find('/');
        int real_id = -1;
        if (first_slash != std::string::npos) {
          std::string id_str = image_name.substr(0, first_slash);
          try {
            real_id = std::stoi(id_str);  // 字符串转整数ID
          } catch (...) {
            std::cerr << "Warning: 路径中real_ID无效 " << image_name
                      << std::endl;
          }
        } else {
          std::cerr << "Warning: 传入路径格式错误(无'/')" << image_name
                    << std::endl;
        }
        if (++cnt % 100 == 0) {
          std::cout << "processing idx: " << cnt << std::endl;
        }
        auto img_path = image_root + image_name;
        std::shared_ptr<BaseImage> image = ImageFactory::readImage(img_path);
        if (!image) {
          std::cerr << "Failed to read image: " << img_path << std::endl;
          continue;
        }
        std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
        std::vector<std::shared_ptr<BaseImage>> input_images = {image};
        model->inference(input_images, out_datas);
        save_classification_results(ofs, out_datas, image_name, real_id);
      }
    }
  }
  ofs.close();
  std::cout << "write done!" << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc != 6) {
    printf(
        "\nUsage: %s MODEL_NAME MODEL_DIR BENCH_PATH IMAGE_ROOT RES_PATH\n\n"
        "\tMODEL_DIR, store cvimodel or bmodel path\n"
        "\tMODEL_NAME, detection model name should be one of:\n"
        "\t    mask_cls, isp_scene_cls, rgbliveness_cls\n"
        "\tBENCH_PATH, txt for storing image names\n"
        "\tIMAGE_ROOT, store images path\n"
        "\tRES_TXT_PATH, save result txt path\n",
        argv[0]);
    return -1;
  }

  std::string model_name = argv[1];
  std::string model_dir = argv[2];
  std::string bench_path = argv[3];
  std::string image_root = argv[4];
  std::string res_path = argv[5];

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

  bench_mark_all(bench_path, image_root, res_path, model);

  return 0;
}