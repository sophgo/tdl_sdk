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
  model_id_mapping["YOLOV8N_DET_HEAD_HARDHAT"] =
      ModelType::YOLOV8N_DET_HEAD_HARDHAT;
  model_id_mapping["YOLOV8N_DET_PERSON_VEHICLE"] =
      ModelType::YOLOV8N_DET_PERSON_VEHICLE;
}

void saveDetectionResults(std::string &dst_root, std::string &img_name,
                          const std::shared_ptr<ModelBoxInfo> &obj_meta) {
  if (!fs::exists(dst_root)) {
    fs::create_directories(dst_root);
  }
  if (dst_root.back() != '/') {
    dst_root += '/';
  }
  std::string txt_name = dst_root + img_name + ".txt";
  std::ofstream outfile(txt_name);
  for (const auto &bbox : obj_meta->bboxes) {
    outfile << std::fixed << std::setprecision(2) << bbox.x1 << " " << bbox.y1
            << " " << bbox.x2 << " " << bbox.y2 << " ";
    outfile << bbox.class_id << " " << std::fixed << std::setprecision(2)
            << bbox.score << std::endl;
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
    if (out_datas[0]->getType() != ModelOutputType::OBJECT_DETECTION) {
      printf("out_datas[%d] is not ModelOutputType::OBJECT_DETECTION\n", 0);
      continue;
    }
    std::shared_ptr<ModelBoxInfo> obj_meta =
        std::static_pointer_cast<ModelBoxInfo>(out_datas[0]);
    std::string img_name = fs::path(image_path).stem().string();
    saveDetectionResults(dst_root, img_name, obj_meta);
  }
  return 0;
}