#include <cstdlib>
#include <iostream>
#include "tdl_model_factory.hpp"

int main(int argc, char **argv) {
  if (argc != 5) {
    printf("Usage: %s <model_id_name> <model_dir> <image_path> <loop_num>\n",
           argv[0]);
    printf("model_id_name:\n");
    for (auto &item : kAllModelTypes) {
      printf("%s\n", modelTypeToString(item).c_str());
    }
    return -1;
  }

  std::string model_id_name = argv[1];
  std::string model_dir = argv[2];
  std::string image_path = argv[3];

  int loop_num = atoi(argv[4]);
  if (loop_num <= 100) {
    std::cerr << "Invalid loop_num: " << argv[4]
              << ". loop_num must be greater than 100." << std::endl;
    return -1;
  }

  std::shared_ptr<BaseImage> image = ImageFactory::readImage(image_path);
  if (!image) {
    printf("Failed to create image\n");
    return -1;
  }

  TDLModelFactory &model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();
  model_factory.setModelDir(model_dir);
  std::shared_ptr<BaseModel> model = model_factory.getModel(model_id_name);
  if (!model) {
    printf("Failed to create model\n");
    return -1;
  }

  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  std::vector<std::shared_ptr<BaseImage>> input_images = {image};

  for (int i = 0; i < loop_num; i++) {
    out_datas.clear();
    model->inference(input_images, out_datas);
  }

  return 0;
}
