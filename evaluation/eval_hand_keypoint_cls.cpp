#include <sys/stat.h>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include "tdl_model_factory.hpp"

int main(int argc, char** argv) {
  if (argc != 6) {
    printf(
        "Usage: %s <model_name> <model_dir> <txt_root> <input_txt_list> "
        "<result_txt>\n",
        argv[0]);
    return -1;
  }
  std::string model_name = argv[1];
  std::string model_dir = argv[2];
  std::string txt_root = argv[3];
  std::string input_txt_list = argv[4];
  std::string result_txt = argv[5];

  struct stat root_stat;
  if (stat(txt_root.c_str(), &root_stat) != 0 || !S_ISDIR(root_stat.st_mode)) {
    printf("Error: txt_root is not a valid directory: %s\n", txt_root.c_str());
    return -1;
  }

  if (txt_root.back() != '/') {
    txt_root += "/";
  }
  std::vector<std::string> keypoints_files;
  std::vector<int> real_ids;
  std::ifstream infile(input_txt_list);
  std::string line;
  while (std::getline(infile, line)) {
    if (!line.empty()) {
      // 读取真实ID
      std::stringstream ss(line);
      std::string real_id_str;
      std::getline(ss, real_id_str, '/');
      int real_id = std::stoi(real_id_str);
      real_ids.push_back(real_id);

      std::string full_path = txt_root + line;
      keypoints_files.push_back(full_path);
    }
  }
  infile.close();

  TDLModelFactory& model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();

  std::shared_ptr<BaseModel> model_hc;
  struct stat path_stat;
  if (stat(model_dir.c_str(), &path_stat) == 0) {
    if (S_ISDIR(path_stat.st_mode)) {  // model_dir是文件夹：原有getModel调用
      model_factory.setModelDir(model_dir);
      model_hc = model_factory.getModel(model_name);
    } else if (S_ISREG(
                   path_stat.st_mode)) {  // model_dir是绝对路径：新getModel调用
      model_hc = model_factory.getModel(model_name, model_dir);
    } else {
      printf("Error: MODEL_DIR is neither dir nor file\n");
      return -1;
    }
  } else {
    printf("Error: Cannot access MODEL_DIR: %s\n", model_dir.c_str());
    return -1;
  }

  if (!model_hc) {
    printf("Failed to create model_hc\n");
    return -1;
  }

  std::ofstream outfile(result_txt);
  if (!outfile.is_open()) {
    printf("Failed to open %s\n", result_txt.c_str());
    return -1;
  }

  for (size_t idx = 0; idx < keypoints_files.size(); ++idx) {
    const auto& keypoints_txt = keypoints_files[idx];
    int real_id = real_ids[idx];

    std::vector<float> keypoints(42);
    FILE* fp = fopen(keypoints_txt.c_str(), "r");
    if (!fp) {
      printf("Failed to open %s\n", keypoints_txt.c_str());
      outfile << keypoints_txt << " Failed_to_open_file\n";
      continue;
    }
    bool read_ok = true;
    for (int i = 0; i < 21; ++i) {
      float x, y;
      if (fscanf(fp, "%f %f", &x, &y) != 2) {
        printf("Failed to read keypoint %d from %s\n", i,
               keypoints_txt.c_str());
        outfile << keypoints_txt << " Failed_to_read_keypoints\n";
        read_ok = false;
        break;
      }
      keypoints[2 * i] = x;
      keypoints[2 * i + 1] = y;
    }
    fclose(fp);
    if (!read_ok) continue;

    std::shared_ptr<BaseImage> bin_data = ImageFactory::createImage(
        42, 1, ImageFormat::GRAY, TDLDataType::FP32, true);

    float* data_buffer =
        reinterpret_cast<float*>(bin_data->getVirtualAddress()[0]);
    memcpy(data_buffer, &keypoints[0], 42 * sizeof(float));
    std::vector<std::shared_ptr<BaseImage>> input_datas = {bin_data};

    std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
    model_hc->inference(input_datas, out_datas);

    if (!out_datas.empty()) {
      std::shared_ptr<ModelClassificationInfo> cls_meta =
          std::static_pointer_cast<ModelClassificationInfo>(out_datas[0]);
      outfile << keypoints_txt << "," << cls_meta->topk_class_ids[0] << ","
              << real_id << "\n";
    } else {
      outfile << keypoints_txt << " No_output\n";
    }
  }

  outfile.close();
  return 0;
}
