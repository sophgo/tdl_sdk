#include <sys/stat.h>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "tdl_model_factory.hpp"
#include "utils/common_utils.hpp"

#define AUDIOFORMATSIZE 2

static void append_result(const std::string& res_path,
                          const std::string& bin_path,
                          const std::shared_ptr<ModelClassificationInfo>& cls) {
  std::ofstream ofs(res_path, std::ios::app);
  if (!ofs) {
    std::perror(("open " + res_path).c_str());
    return;
  }
  ofs << bin_path << ' ' << cls->topk_class_ids[0] << ' ' << cls->topk_scores[0]
      << '\n';
}

int main(int argc, char* argv[]) {
  if (argc != 7) {
    printf(
        "Usage: %s <model_id_name> <model_dir> "
        "<bin_list_path> <sample_rate> <seconds> <res_txt_path>\n",
        argv[0]);
    printf("model_id_name candidates:\n");
    return -1;
  }

  std::string model_id_name = argv[1];
  std::string model_dir = argv[2];
  std::string list_path = argv[3];
  int sample_rate = std::atoi(argv[4]);
  int seconds = std::atoi(argv[5]);
  std::string res_txt_path = argv[6];
  const int frame_size = sample_rate * AUDIOFORMATSIZE * seconds;

  TDLModelFactory& model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();

  std::shared_ptr<BaseModel> model_sound;
  struct stat path_stat;
  if (stat(model_dir.c_str(), &path_stat) == 0) {
    if (S_ISDIR(path_stat.st_mode)) {  // model_dir是文件夹：原有getModel调用
      model_factory.setModelDir(model_dir);
      model_sound = model_factory.getModel(model_id_name);
    } else if (S_ISREG(
                   path_stat.st_mode)) {  // model_dir是绝对路径：新getModel调用
      model_sound = model_factory.getModel(model_id_name, model_dir);
    } else {
      printf("Error: MODEL_DIR is neither dir nor file\n");
      return -1;
    }
  } else {
    printf("Error: Cannot access MODEL_DIR: %s\n", model_dir.c_str());
    return -1;
  }

  if (!model_sound) {
    printf("Failed to create model\n");
    return -1;
  }

  {
    std::ofstream ofs(res_txt_path, std::ios::trunc);
    if (!ofs) {
      std::perror(("create " + res_txt_path).c_str());
      return -1;
    }
  }

  std::ifstream list_file(list_path);
  if (!list_file.is_open()) {
    std::perror(("open " + list_path).c_str());
    return -1;
  }

  std::string bin_path;
  size_t line_idx = 0;
  std::vector<uint8_t> host_buf(frame_size);

  while (std::getline(list_file, bin_path)) {
    ++line_idx;
    if (bin_path.empty()) continue;

    std::vector<uint8_t> buffer;
    if (!CommonUtils::readBinaryFile(bin_path, buffer)) {
      printf("read file failed\n");
      return -1;
    }

    auto in_image = ImageFactory::createImage(frame_size, 1, ImageFormat::GRAY,
                                              TDLDataType::UINT8, true);
    std::memcpy(in_image->getVirtualAddress()[0], host_buf.data(), frame_size);

    TDLModelFactory& model_factory = TDLModelFactory::getInstance();
    model_factory.loadModelConfig();

    std::shared_ptr<BaseModel> model_sound;
    struct stat path_stat;
    if (stat(model_dir.c_str(), &path_stat) == 0) {
      if (S_ISDIR(path_stat.st_mode)) {  // model_dir是文件夹：原有getModel调用
        model_factory.setModelDir(model_dir);
        model_sound = model_factory.getModel(model_id_name);
      } else if (S_ISREG(
                     path_stat
                         .st_mode)) {  // model_dir是绝对路径：新getModel调用
        model_sound = model_factory.getModel(model_id_name, model_dir);
      } else {
        printf("Error: MODEL_DIR is neither dir nor file\n");
        return -1;
      }
    } else {
      printf("Error: Cannot access MODEL_DIR: %s\n", model_dir.c_str());
      return -1;
    }

    std::vector<std::shared_ptr<BaseImage>> input_datas = {in_image};
    std::vector<std::shared_ptr<ModelOutputInfo>> outputs;

    model_sound->inference(input_datas, outputs);
    if (outputs.empty()) {
      std::cerr << "[WARN] inference output empty: " << bin_path << '\n';
      continue;
    }

    auto cls = std::static_pointer_cast<ModelClassificationInfo>(outputs[0]);

    append_result(res_txt_path, bin_path, cls);

    if (line_idx % 50 == 0) std::cout << "processed " << line_idx << " files\n";
  }

  std::cout << "All done, result saved to " << res_txt_path << '\n';
  return 0;
}
