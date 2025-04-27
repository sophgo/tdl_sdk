#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "tdl_model_factory.hpp"

void construct_model_id_mapping(std::map<std::string, ModelType>& model_id_mapping) {
  model_id_mapping["mask_cls"] = ModelType::CLS_MASK;
  model_id_mapping["isp_scene_cls"] = ModelType::CLS_ISP_SCENE;
  model_id_mapping["rgbliveness_cls"] = ModelType::CLS_RGBLIVENESS;
}

void save_classification_results(const std::string& save_path, 
                               const std::vector<std::shared_ptr<ModelOutputInfo>>& out_datas,
                               const std::string& image_name) {
  std::stringstream res_ss;
  res_ss << image_name << " ";
  
  for (const auto& out_data : out_datas) {
    if (out_data->getType() != ModelOutputType::CLASSIFICATION) {
      continue;
    }

    std::shared_ptr<ModelClassificationInfo> cls_meta =
        std::static_pointer_cast<ModelClassificationInfo>(out_data);
    
    for (size_t i = 0; i < cls_meta->topk_class_ids.size(); i++) {
      res_ss << " " << cls_meta->topk_class_ids[i]<< " " << cls_meta->topk_scores[i];
    }
  }
  res_ss << "\n";

  FILE* fp = fopen(save_path.c_str(), "a");
  if (fp) {
    fwrite(res_ss.str().c_str(), res_ss.str().size(), 1, fp);
    fclose(fp);
  }
}

void bench_mark_all(std::string bench_path, std::string image_root, std::string res_path,
                   int class_num, std::shared_ptr<BaseModel> model) {
  std::fstream file(bench_path);
  if (!file.is_open()) {
    printf("can not open bench path %s\n", bench_path.c_str());
    return;
  }
  printf("open bench path %s success!\n", bench_path.c_str());
  
  // Clear or create the result file
  FILE* fp = fopen(res_path.c_str(), "w");
  if (fp) fclose(fp);

  std::string line;
  int cnt = 0;

  while (getline(file, line)) {
    if (!line.empty()) {
      std::stringstream ss(line);
      std::string image_name;
      while (ss >> image_name) {
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

        save_classification_results(res_path, out_datas, image_name);
      }
    }
  }
  std::cout << "write done!" << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc != 7) {
    printf(
        "\nUsage: %s MODEL_PATH MODEL_NAME CLASS_NUM BENCH_PATH IMAGE_ROOT RES_PATH\n\n"
        "\tMODEL_PATH, cvimodel path\n"
        "\tMODEL_NAME, detection model name should be one of:\n"
        "\t    mask_cls, isp_scene_cls, rgbliveness_cls\n"
        "\tCLASS_NUM, number of classes\n"
        "\tBENCH_PATH, txt for storing image names\n"
        "\tIMAGE_ROOT, store images path\n"
        "\tRES_PATH, save result path\n",
        argv[0]);
    return -1;
  }

  std::map<std::string, ModelType> model_id_mapping;
  construct_model_id_mapping(model_id_mapping);

  std::string model_path = argv[1];
  std::string model_name = argv[2];
  int class_num = std::stoi(argv[3]);
  std::string bench_path = argv[4];
  std::string image_root = argv[5];
  std::string res_path = argv[6];

  if (model_id_mapping.find(model_name) == model_id_mapping.end()) {
    printf("unsupported model: %s\n", model_name.c_str());
    return -1;
  }

  TDLModelFactory model_factory;
  std::shared_ptr<BaseModel> model = model_factory.getModel(model_id_mapping[model_name], model_path);
  if (!model) {
    printf("Failed to create model\n");
    return -1;
  }

  std::cout << "model opened:" << model_path << std::endl;
  bench_mark_all(bench_path, image_root, res_path, class_num, model);

  return 0;
}