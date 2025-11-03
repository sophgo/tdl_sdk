#include <sys/stat.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "tdl_model_factory.hpp"

std::vector<std::string> read_file_lines(const std::string& file_path) {
  std::vector<std::string> lines;
  std::ifstream infile(file_path);
  std::string line;
  while (std::getline(infile, line)) {
    if (!line.empty()) lines.push_back(line);
  }
  return lines;
}

std::string replace_file_ext(const std::string& filename,
                             const std::string& new_ext) {
  size_t lastdot = filename.find_last_of(".");
  if (lastdot == std::string::npos) return filename + "." + new_ext;
  return filename.substr(0, lastdot + 1) + new_ext;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    printf("Usage: %s <model_id_name> <model_dir> <image_list> <save_dir>\n",
           argv[0]);
    return -1;
  }
  std::string model_id_name = argv[1];
  std::string model_dir = argv[2];
  std::string image_list = argv[3];
  std::string save_dir = argv[4];

  if (save_dir.back() != '/') save_dir += "/";

  std::vector<std::string> file_list = read_file_lines(image_list);
  if (file_list.empty()) {
    std::cout << "file_list empty\n";
    return -1;
  }

  TDLModelFactory& model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();

  std::shared_ptr<BaseModel> model;
  struct stat path_stat;
  if (stat(model_dir.c_str(), &path_stat) == 0) {
    if (S_ISDIR(path_stat.st_mode)) {  // model_dir是文件夹：原有getModel调用
      model_factory.setModelDir(model_dir);
      model = model_factory.getModel(model_id_name);
    } else if (S_ISREG(
                   path_stat.st_mode)) {  // model_dir是绝对路径：新getModel调用
      model = model_factory.getModel(model_id_name, model_dir);
    } else {
      printf("Error: MODEL_DIR is neither dir nor file\n");
      return -1;
    }
  } else {
    printf("Error: Cannot access MODEL_DIR: %s\n", model_dir.c_str());
    return -1;
  }

  if (!model) {
    std::cout << "Failed to create model\n";
    return -1;
  }
  model->setExportFeature(1);

  for (size_t i = 0; i < file_list.size(); ++i) {
    std::string input_image_path = file_list[i];

    std::shared_ptr<BaseImage> image =
        ImageFactory::readImage(input_image_path);
    if (!image) {
      std::cout << "Failed to read image: " << input_image_path << std::endl;
      continue;
    }

    std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
    std::vector<std::shared_ptr<BaseImage>> input_images = {image};
    model->inference(input_images, out_datas);

    if (out_datas.empty()) {
      std::cout << "No output for image: " << input_image_path << std::endl;
      continue;
    }

    std::shared_ptr<ModelBoxLandmarkInfo> obj_meta =
        std::static_pointer_cast<ModelBoxLandmarkInfo>(out_datas[0]);

    std::string filename =
        input_image_path.substr(input_image_path.find_last_of("/\\") + 1);
    std::string save_path = save_dir + replace_file_ext(filename, "txt");
    std::ofstream outFile(save_path);
    if (!outFile) {
      std::cout << "Error: Could not open file " << save_path << " for writing."
                << std::endl;
      continue;
    }

    for (size_t j = 0; j < 56; ++j) {
      outFile << obj_meta->feature[j] << " ";
    }
    outFile << std::endl;
    for (size_t j = 0; j < 14; ++j) {
      outFile << obj_meta->feature[56 + j] << " ";
    }

    outFile.close();
    std::cout << "processed :" << i + 1 << "/" << file_list.size() << "\t"
              << input_image_path << std::endl;
  }

  return 0;
}
