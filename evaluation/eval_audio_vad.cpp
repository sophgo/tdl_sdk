#include <sys/stat.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include "audio_classification/fsmn_vad.hpp"
#include "tdl_model_factory.hpp"
#include "utils/common_utils.hpp"

int bench_mark_all(const std::string& image_root, const std::string& image_dir,
                   const std::string& res_path,
                   std::shared_ptr<BaseModel> model) {
  std::ifstream bench_fstream(image_root);
  if (!bench_fstream.is_open()) {
    std::cerr << "打开 benchmark 文件失败: " << image_root << "\n";
    return -1;
  }

  std::ofstream file(res_path);
  if (!file.is_open()) {
    std::cerr << "打开结果文件失败: " << res_path << "\n";
    return -1;
  }

  std::string bin_data_name;
  int count = 0;
  while (bench_fstream >> bin_data_name) {
    const std::string bin_data_path = image_dir + "/" + bin_data_name;
    if (++count % 100 == 0) {
      std::cout << "processing idx: " << count << std::endl;
    }
    std::vector<uint8_t> buffer;
    if (!CommonUtils::readBinaryFile(bin_data_path, buffer)) {
      printf("read file failed\n");
      return -1;
    }

    int frame_size = buffer.size();

    std::shared_ptr<BaseImage> bin_data = ImageFactory::createImage(
        frame_size, 1, ImageFormat::GRAY, TDLDataType::UINT8, true);
    uint8_t* data_buffer = bin_data->getVirtualAddress()[0];
    memcpy(data_buffer, buffer.data(), frame_size * sizeof(uint8_t));

    std::shared_ptr<ModelOutputInfo> output_info;
    model->inference(bin_data, output_info);

    std::shared_ptr<ModelVADInfo> vad_meta =
        std::static_pointer_cast<ModelVADInfo>(output_info);

    // 提取第一个/前的数，非零则为1
    int label = 0;
    size_t first_slash = bin_data_name.find('/');
    if (first_slash != std::string::npos) {
      std::string label_str = bin_data_name.substr(0, first_slash);
      int label_val = std::stoi(label_str);
      label = (label_val != 0) ? 1 : 0;
    }

    file << bin_data_name << "," << (vad_meta->has_segments ? "1" : "0") << ","
         << label << "\n";
  }

  file.close();
  std::cout << "write done!" << std::endl;
  return 0;
}

int main(int argc, char** argv) {
  if (argc != 6) {
    printf(
        "Usage: %s <model_name> <model_dir> <image_root> "
        "<image_dir> <txt_result_path>",
        argv[0]);  // bin data with 16000 sr
    return -1;
  }

  std::string model_name = argv[1];
  std::string model_dir = argv[2];
  std::string image_root = argv[3];
  std::string image_dir = argv[4];
  std::string txt_result_path = argv[5];

  TDLModelFactory& model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();
  std::shared_ptr<BaseModel> model;

  struct stat path_stat;
  if (stat(model_dir.c_str(), &path_stat) == 0) {
    if (S_ISDIR(path_stat.st_mode)) {  // model_dir是文件夹：原有getModel调用
      model_factory.setModelDir(model_dir);
      model = model_factory.getModel(model_name);
    } else if (S_ISREG(
                   path_stat.st_mode)) {  // model_dir是绝对路径：新getModel调用
      model = model_factory.getModel(model_name, model_dir);
    } else {
      printf("Error: MODEL_DIR is neither dir nor file\n");
      return -1;
    }
  } else {
    printf("Error: Cannot access MODEL_DIR: %s\n", model_dir.c_str());
    return -1;
  }

  if (!model) {
    printf("Failed to create model\n");
    return -1;
  }

  bench_mark_all(image_root, image_dir, txt_result_path, model);

  return 0;
}
