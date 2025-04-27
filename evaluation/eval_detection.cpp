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
  // MobileDetV2 models
  model_id_mapping["mobiledetv2-pedestrian"] = ModelType::MBV2_DET_PERSON;
  
  // YOLO general models
  model_id_mapping["yolov6"] = ModelType::YOLOV6;
  model_id_mapping["yolov6-coco80"] = ModelType::YOLOV6_DET_COCO80;
  model_id_mapping["yolov8"] = ModelType::YOLOV8;
  model_id_mapping["yolov8-coco80"] = ModelType::YOLOV8_DET_COCO80;
  model_id_mapping["yolov10"] = ModelType::YOLOV10;
  model_id_mapping["yolov10-coco80"] = ModelType::YOLOV10_DET_COCO80;

  // YOLOv8 specialized models
  model_id_mapping["yolov8n-hand"] = ModelType::YOLOV8N_DET_HAND;
  model_id_mapping["yolov8n-pet-person"] = ModelType::YOLOV8N_DET_PET_PERSON;
  model_id_mapping["yolov8n-person-vehicle"] = ModelType::YOLOV8N_DET_PERSON_VEHICLE;
  model_id_mapping["yolov8n-hand-face-person"] = ModelType::YOLOV8N_DET_HAND_FACE_PERSON;
  model_id_mapping["yolov8n-head-person"] = ModelType::YOLOV8N_DET_HEAD_PERSON;
  model_id_mapping["yolov8n-head-hardhat"] = ModelType::YOLOV8N_DET_HEAD_HARDHAT;
  model_id_mapping["yolov8n-fire-smoke"] = ModelType::YOLOV8N_DET_FIRE_SMOKE;
  model_id_mapping["yolov8n-fire"] = ModelType::YOLOV8N_DET_FIRE;
  model_id_mapping["yolov8n-head-shoulder"] = ModelType::YOLOV8N_DET_HEAD_SHOULDER;
  model_id_mapping["yolov8n-license-plate"] = ModelType::YOLOV8N_DET_LICENSE_PLATE;
  model_id_mapping["yolov8n-traffic-light"] = ModelType::YOLOV8N_DET_TRAFFIC_LIGHT;
  model_id_mapping["yolov8n-monitor-person"] = ModelType::YOLOV8N_DET_MONITOR_PERSON;
}

void save_detection_results(const std::string& save_path, 
                           const std::shared_ptr<ModelBoxInfo>& obj_meta) {
  std::stringstream res_ss;
  for (size_t i = 0; i < obj_meta->bboxes.size(); i++) {
    res_ss << obj_meta->bboxes[i].class_id << " "
           << obj_meta->bboxes[i].x1 << " "
           << obj_meta->bboxes[i].y1 << " "
           << obj_meta->bboxes[i].x2 << " "
           << obj_meta->bboxes[i].y2 << " "
           << obj_meta->bboxes[i].score << "\n";
  }

  FILE* fp = fopen(save_path.c_str(), "w");
  if (fp) {
    fwrite(res_ss.str().c_str(), res_ss.str().size(), 1, fp);
    fclose(fp);
  }
}

void bench_mark_all(std::string bench_path, std::string image_root, std::string res_path,
                    std::shared_ptr<BaseModel> model) {
  std::fstream file(bench_path);
  if (!file.is_open()) {
    return;
  }

  std::string line;
  while (getline(file, line)) {
    if (!line.empty()) {
      std::stringstream ss(line);
      std::string image_name;
      while (ss >> image_name) {
        auto img_path = image_root + image_name;
        std::shared_ptr<BaseImage> image = ImageFactory::readImage(img_path);
        if (!image) {
          std::cerr << "Failed to read image: " << img_path << std::endl;
          continue;
        }

        std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
        std::vector<std::shared_ptr<BaseImage>> input_images = {image};
        model->inference(input_images, out_datas);

        for (auto& out_data : out_datas) {
          if (out_data->getType() != ModelOutputType::OBJECT_DETECTION) {
            continue;
          }

          std::shared_ptr<ModelBoxInfo> obj_meta =
              std::static_pointer_cast<ModelBoxInfo>(out_data);

          size_t pos = image_name.find_last_of("/");
          size_t start_idx = pos == std::string::npos ? 0 : pos + 1;
          std::string save_path =
              res_path + image_name.substr(start_idx, image_name.length() - 4 - (pos + 1)) + ".txt";
          
          save_detection_results(save_path, obj_meta);
        }
        break;
      }
    }
  }
  std::cout << "write done!" << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc != 5 && argc != 6) {
      printf(
          "\nUsage: %s MODEL_NAME MODEL_PATH BENCH_PATH IMAGE_ROOT RES_PATH [CONF_THRESHOLD] "
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
          "\tMODEL_PATH, cvimodel path\n"
          "\tBENCH_PATH, txt for storing image names\n"
          "\tIMAGE_ROOT, store images path\n"
          "\tRES_PATH, save result path\n"
          "\tCONF_THRESHOLD (optional), confidence threshold (default: 0.5)\n"
          "\tNMS_THRESHOLD (optional), NMS threshold (default: 0.5)\n",
          argv[0]);
      return -1;
  }
  std::map<std::string, ModelType> model_id_mapping;
  construct_model_id_mapping(model_id_mapping);

  std::string model_name = argv[1];
  std::string model_path = argv[2];
  std::string bench_path = argv[3];
  std::string image_root = argv[4];
  std::string res_path = argv[5];

  float conf_threshold = 0.01;
  if (argc > 6) {
    conf_threshold = std::stof(argv[6]);
  }

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

  model->setModelThreshold(conf_threshold);

  std::cout << "model opened:" << model_path << std::endl;
  bench_mark_all(bench_path, image_root, res_path, model);

  return 0;
}