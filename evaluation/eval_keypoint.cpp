#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "tdl_model_factory.hpp"

std::string get_basename_noext(const std::string& path) {
  size_t slash = path.find_last_of("/\\");
  size_t start = (slash == std::string::npos) ? 0 : slash + 1;
  size_t dot = path.find_last_of('.');
  if (dot == std::string::npos || dot < start) dot = path.length();
  return path.substr(start, dot - start);
}

void dump_kp_to_dir_txt(
    const std::string& img_path,
    const std::vector<std::shared_ptr<ModelOutputInfo>>& outs,
    const std::string& result_dir, std::string& model_id_name,
    float image_width, float image_height, float conf_threshold) {
  std::string base_name = get_basename_noext(img_path);
  std::string txt_path = result_dir;

  if (!txt_path.empty() && txt_path.back() != '/' && txt_path.back() != '\\')
    txt_path += "/";
  txt_path += base_name + ".txt";

  std::ofstream ofs(txt_path.c_str(), std::ios::trunc);
  if (!ofs.is_open()) {
    std::cerr << "open write file fail: " << txt_path << '\n';
    return;
  }

  for (size_t h = 0; h < outs.size(); ++h) {
    std::shared_ptr<ModelLandmarksInfo> lm =
        std::static_pointer_cast<ModelLandmarksInfo>(outs[h]);

    for (size_t k = 0; k < lm->landmarks_x.size(); ++k) {
      float x = lm->landmarks_x[k];
      float y = lm->landmarks_y[k];

      if (model_id_name == "KEYPOINT_FACE_V2" ||
          model_id_name == "KEYPOINT_SIMCC_PERSON17" ||
          model_id_name == "KEYPOINT_LICENSE_PLATE") {
        x /= image_width;
        y /= image_height;
      }

      if (model_id_name == "KEYPOINT_SIMCC_PERSON17" &&
          lm->landmarks_score[k] < conf_threshold) {
        x = 0;
        y = 0;
      }

      ofs << ' ' << x << ' ' << y << "\n";
    }
  }
  ofs.close();
}

int main(int argc, char* argv[]) {
  if (argc != 6 && argc != 7) {
    printf(
        "Usage: %s <model_id_name> <model_dir or model_path> <image_dir> "
        "<image_list.txt> <result_dir> ( optional: <conf_threshold>)  \n",
        argv[0]);
    return -1;
  }

  std::string model_id_name = argv[1];
  std::string model_dir = argv[2];
  std::string image_dir = argv[3];
  std::string list_txt = argv[4];
  std::string result_dir = argv[5];

  float conf_threshold = 0.5f;
  if (argc == 7) {
    conf_threshold = atof(argv[6]);
  }

  TDLModelFactory& model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();

  std::shared_ptr<BaseModel> kp_model;
  struct stat path_stat;
  if (stat(model_dir.c_str(), &path_stat) == 0) {
    if (S_ISDIR(path_stat.st_mode)) {  // model_dir是文件夹：原有getModel调用
      model_factory.setModelDir(model_dir);
      kp_model = model_factory.getModel(model_id_name);
    } else if (S_ISREG(
                   path_stat.st_mode)) {  // model_dir是绝对路径：新getModel调用
      kp_model = model_factory.getModel(model_id_name, model_dir);
    } else {
      printf("Error: MODEL_DIR is neither dir nor file\n");
      return -1;
    }
  } else {
    printf("Error: Cannot access MODEL_DIR: %s\n", model_dir.c_str());
    return -1;
  }

  if (!kp_model) {
    printf("Failed to create model\n");
    return -1;
  }

  std::ifstream fin(list_txt);
  if (!fin.is_open()) {
    std::perror(("open " + list_txt).c_str());
    return -1;
  }

  std::string img_name;
  size_t img_idx = 0;
  while (std::getline(fin, img_name)) {
    if (img_name.empty()) continue;
    ++img_idx;
    if (img_idx % 20 == 0) std::cout << "processed " << img_idx << " images\n";

    const std::string img_path = image_dir + "/" + img_name;
    auto img = ImageFactory::readImage(img_path);
    if (!img) {
      std::cerr << "readImage fail: " << img_path << '\n';
      continue;
    }

    std::vector<std::shared_ptr<ModelOutputInfo>> kp_out;
    kp_model->inference({img}, kp_out);

    uint32_t image_width = img->getWidth();
    uint32_t image_height = img->getHeight();

    dump_kp_to_dir_txt(img_path, kp_out, result_dir, model_id_name, image_width,
                       image_height, conf_threshold);
  }

  std::cout << "All done. total = " << img_idx << '\n';
  return 0;
}
