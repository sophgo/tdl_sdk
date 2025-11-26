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
                          const std::shared_ptr<ModelClassificationInfo>& cls,
                          int real_id) {
  std::ofstream ofs(res_path, std::ios::app);
  if (!ofs) {
    std::perror(("open " + res_path).c_str());
    return;
  }
  ofs << bin_path << ',' << cls->topk_class_ids[0] << ',' << real_id << '\n';
}

int main(int argc, char* argv[]) {
  if (argc != 6) {
    printf(
        "Usage: %s <model_id_name> <model_dir> "
        "<bin_list_path> <audio_root> <res_txt_path>\n",
        argv[0]);
    printf("model_id_name candidates:\n");
    return -1;
  }

  std::string model_id_name = argv[1];
  std::string model_dir = argv[2];
  std::string list_path = argv[3];
  std::string audio_root = argv[4];
  std::string res_txt_path = argv[5];

  struct stat root_stat;
  if (stat(audio_root.c_str(), &root_stat) != 0 ||
      !S_ISDIR(root_stat.st_mode)) {
    printf("Error: audio_root is not a valid directory: %s\n",
           audio_root.c_str());
    return -1;
  }
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

  std::string rel_bin_path;
  size_t line_idx = 0;

  while (std::getline(list_file, rel_bin_path)) {
    ++line_idx;
    if (rel_bin_path.empty()) continue;

    int real_id = -1;
    size_t first_slash = rel_bin_path.find('/');
    if (first_slash != std::string::npos) {
      std::string id_str = rel_bin_path.substr(0, first_slash);
      try {
        real_id = std::stoi(id_str);  // 转换为整数ID
      } catch (...) {
        std::cerr << "Warning: 路径中real_ID无效 " << rel_bin_path << std::endl;
      }
    } else {
      std::cerr << "Warning: 路径格式错误(无'/') " << rel_bin_path << std::endl;
    }

    std::string full_bin_path = audio_root + "/" + rel_bin_path;

    std::vector<uint8_t> buffer;
    if (!CommonUtils::readBinaryFile(full_bin_path, buffer)) {
      printf("read file failed\n", full_bin_path.c_str());
      continue;
    }

    int frame_size = buffer.size();
    auto in_image = ImageFactory::createImage(frame_size, 1, ImageFormat::GRAY,
                                              TDLDataType::UINT8, true);
    std::memcpy(in_image->getVirtualAddress()[0], buffer.data(), frame_size);
    std::vector<std::shared_ptr<BaseImage>> input_datas = {in_image};
    std::vector<std::shared_ptr<ModelOutputInfo>> outputs;

    model_sound->inference(input_datas, outputs);
    if (outputs.empty()) {
      std::cerr << "[WARN] inference output empty: " << rel_bin_path << '\n';
      continue;
    }

    auto cls = std::static_pointer_cast<ModelClassificationInfo>(outputs[0]);

    append_result(res_txt_path, rel_bin_path, cls, real_id);

    if (line_idx % 50 == 0) std::cout << "processed " << line_idx << " files\n";
  }

  std::cout << "All done, result saved to " << res_txt_path << '\n';
  return 0;
}
