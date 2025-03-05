#include <dirent.h>
#include <sys/stat.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include "core/cvi_tdl_types_mem.h"
#include "core/cvtdl_core_types.h"
#include "image/base_image.hpp"
#include "models/tdl_model_factory.hpp"
int main(int argc, char** argv) {
  if (argc != 4) {
    printf("Usage: %s <model_dir> <dataset_dir> <result_file>\n", argv[0]);
    // dataset_dir下有各类别文件夹，每个文件夹下存放各类图片
    return -1;
  }

  std::string model_dir = argv[1];
  std::string dataset_dir = argv[2];
  std::string result_file = argv[3];

  std::filesystem::path result_path(result_file);
  if (!result_path.parent_path().empty() &&
      !std::filesystem::exists(result_path.parent_path())) {
    std::filesystem::create_directories(result_path.parent_path());
  }

  std::ofstream out_file(result_file);
  if (!out_file.is_open()) {
    printf("Failed to open result file: %s\n", result_file.c_str());
    return -1;
  }

  TDLModelFactory model_factory(model_dir);

  std::shared_ptr<BaseModel> model_cls =
      model_factory.getModel(TDL_MODEL_TYPE_FACE_ANTI_SPOOF_CLASSIFICATION);

  if (!model_cls) {
    printf("Failed to load classification model\n");
    return -1;
  }

  std::vector<std::string> class_names;
  DIR* dir = opendir(dataset_dir.c_str());
  if (dir != nullptr) {
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
      if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
        continue;
      }
      std::string path = dataset_dir + "/" + entry->d_name;
      struct stat path_stat;
      stat(path.c_str(), &path_stat);
      if (S_ISDIR(path_stat.st_mode)) {
        class_names.push_back(entry->d_name);
      }
    }
    closedir(dir);
  } else {
    std::cerr << "Can't open dataset dir: " << dataset_dir << std::endl;
    return -1;
  }
  if (class_names.empty()) {
    std::cerr << "Warning: No class directory found in " << dataset_dir
              << std::endl;
    return -1;
  }

  size_t total_files = 0;
  for (const auto& class_name : class_names) {
    std::string class_path = dataset_dir + "/" + class_name;
    for (const auto& entry : std::filesystem::directory_iterator(class_path)) {
      total_files++;
    }
  }

  size_t processed_files = 0;
  printf("\nStarting classification...\n");
  for (const auto& class_name : class_names) {
    std::string class_path = dataset_dir + "/" + class_name;
    for (const auto& entry : std::filesystem::directory_iterator(class_path)) {
      std::string img_path = entry.path().string();
      auto image = ImageFactory::readImage(img_path);
      if (!image) {
        printf("Failed to load image: %s\n", img_path.c_str());
        continue;
      }

      std::vector<std::shared_ptr<BaseImage>> input_images = {image};
      std::vector<std::shared_ptr<ModelOutputInfo>> out_cls;
      model_cls->inference(input_images, out_cls);

      cvtdl_class_meta_t* cls_meta =
          static_cast<cvtdl_class_meta_t*>(out_cls[0]);
      int pred_label = cls_meta->cls[0];
      out_file << img_path << " " << pred_label << " " << class_name
               << std::endl;

      processed_files++;
      float progress = (float)processed_files / total_files * 100;
      printf("\rProgress: [");
      int pos = 50 * progress / 100;
      for (int i = 0; i < 50; i++) {
        if (i < pos)
          printf("=");
        else if (i == pos)
          printf(">");
        else
          printf(" ");
      }
      printf("] %.2f%% (%zu/%zu)", progress, processed_files, total_files);
      fflush(stdout);
    }
  }
  printf("\nClassification completed!\n");

  out_file.close();
  return 0;
}