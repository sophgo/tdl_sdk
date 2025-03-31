#include <dirent.h>
#include <experimental/filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "tdl_model_factory.hpp"
namespace fs = std::experimental::filesystem;

void constructModelIdMapping(
    std::map<std::string, ModelType> &model_id_mapping) {
  model_id_mapping["MBV2_DET_PERSON"] = ModelType::MBV2_DET_PERSON;
  model_id_mapping["YOLOV8N_DET_HAND"] = ModelType::YOLOV8N_DET_HAND;
  model_id_mapping["YOLOV8N_DET_PET_PERSON"] =
      ModelType::YOLOV8N_DET_PET_PERSON;
  model_id_mapping["YOLOV8N_DET_PERSON_VEHICLE"] =
      ModelType::YOLOV8N_DET_PERSON_VEHICLE;
  model_id_mapping["YOLOV8N_DET_HAND_FACE_PERSON"] =
      ModelType::YOLOV8N_DET_HAND_FACE_PERSON;
  model_id_mapping["YOLOV8N_DET_HEAD_PERSON"] =
      ModelType::YOLOV8N_DET_HEAD_PERSON;
  model_id_mapping["YOLOV8N_DET_HEAD_HARDHAT"] =
      ModelType::YOLOV8N_DET_HEAD_HARDHAT;
  model_id_mapping["YOLOV8N_DET_FIRE_SMOKE"] =
      ModelType::YOLOV8N_DET_FIRE_SMOKE;
  model_id_mapping["YOLOV8N_DET_FIRE"] = ModelType::YOLOV8N_DET_FIRE;
  model_id_mapping["YOLOV8N_DET_HEAD_SHOULDER"] =
      ModelType::YOLOV8N_DET_HEAD_SHOULDER;
  model_id_mapping["YOLOV8N_DET_LICENSE_PLATE"] =
      ModelType::YOLOV8N_DET_LICENSE_PLATE;
  model_id_mapping["YOLOV8N_DET_TRAFFIC_LIGHT"] =
      ModelType::YOLOV8N_DET_TRAFFIC_LIGHT;
  model_id_mapping["YOLOV8N_DET_MONITOR_PERSON"] =
      ModelType::YOLOV8N_DET_MONITOR_PERSON;
  model_id_mapping["SCRFD_DET_FACE"] = ModelType::SCRFD_DET_FACE;
}

void saveDetectionResults(std::string &dst_root, std::string &img_name,
                          const std::shared_ptr<ModelOutputInfo> &out_data) {
  if (!fs::exists(dst_root)) {
    fs::create_directories(dst_root);
  }
  if (dst_root.back() != '/') {
    dst_root += '/';
  }
  std::string txt_name = dst_root + img_name + ".txt";
  std::ofstream outfile(txt_name);

  if (out_data->getType() == ModelOutputType::OBJECT_DETECTION) {
    std::shared_ptr<ModelBoxInfo> obj_meta =
        std::static_pointer_cast<ModelBoxInfo>(out_data);
    for (const auto &bbox : obj_meta->bboxes) {
      outfile << std::fixed << std::setprecision(2) << bbox.x1 << " " << bbox.y1
              << " " << bbox.x2 << " " << bbox.y2 << " ";
      outfile << bbox.class_id << " " << std::fixed << std::setprecision(2)
              << bbox.score << std::endl;
    }
  } else if (out_data->getType() ==
             ModelOutputType::OBJECT_DETECTION_WITH_LANDMARKS) {
    std::shared_ptr<ModelBoxLandmarkInfo> obj_meta =
        std::static_pointer_cast<ModelBoxLandmarkInfo>(out_data);
    for (const auto &box_landmark : obj_meta->box_landmarks) {
      outfile << std::fixed << std::setprecision(2) << box_landmark.x1 << " "
              << box_landmark.y1 << " " << box_landmark.x2 << " "
              << box_landmark.y2 << " " << box_landmark.score << std::endl;
    }
  } else {
    std::cout << "out_data->getType() is not supported ";
  }
  outfile.close();
  std::cout << "write file " << txt_name << " done" << std::endl;
}

int main(int argc, char **argv) {
  std::string model_id_name = argv[1];
  std::string model_path = argv[2];
  std::string image_dir = argv[3];
  std::string dst_root = argv[4];
  float model_threshold;
  if (argc == 6) {
    model_threshold = atof(argv[5]);
  } else {
    model_threshold = 0.5;
  }

  std::map<std::string, ModelType> model_id_mapping;
  constructModelIdMapping(model_id_mapping);

  if (model_id_mapping.find(model_id_name) == model_id_mapping.end()) {
    printf("model_id_name not found\n");
    return -1;
  }
  ModelType model_id = model_id_mapping[model_id_name];

  TDLModelFactory model_factory;
  std::shared_ptr<BaseModel> model =
      model_factory.getModel(model_id, model_path);
  if (!model) {
    printf("Failed to create model\n");
    return -1;
  }
  model->setModelThreshold(model_threshold);

  for (const auto &entry : fs::directory_iterator(image_dir)) {
    std::string image_path = entry.path().string();
    std::shared_ptr<BaseImage> image = ImageFactory::readImage(image_path);
    std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
    std::vector<std::shared_ptr<BaseImage>> input_images = {image};
    model->inference(input_images, out_datas);
    std::string img_name = fs::path(image_path).stem().string();
    saveDetectionResults(dst_root, img_name, out_datas[0]);
  }
  return 0;
}