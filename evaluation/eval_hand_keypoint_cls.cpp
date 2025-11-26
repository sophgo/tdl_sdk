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
        "Usage: %s <model_name> <model_dir> <input_txt_list> <txt_root> "
        "<result_txt>\n",
        argv[0]);
    return -1;
  }
  std::string model_name = argv[1];
  std::string model_dir = argv[2];
  std::string input_txt_list = argv[3];
  std::string txt_root = argv[4];
  std::string result_txt = argv[5];

  struct stat root_stat;
  if (stat(txt_root.c_str(), &root_stat) != 0 || !S_ISDIR(root_stat.st_mode)) {
    printf("Error: txt_root is not a valid directory: %s\n", txt_root.c_str());
    return -1;
  }

  if (txt_root.back() != '/') {
    txt_root += "/";
  }

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

  int cnt = 0;
  std::ifstream infile(input_txt_list);
  std::string line;
  while (std::getline(infile, line)) {
    if (!line.empty()) {
      // 读取真实ID
      size_t first_slash = line.find('/');
      int real_id = -1;
      if (first_slash != std::string::npos) {
        std::string id_str = line.substr(0, first_slash);
        try {
          real_id = std::stoi(id_str);  // 字符串转整数ID
        } catch (...) {
          std::cerr << "Warning: 路径中real_ID无效 " << line << std::endl;
          continue;
        }
      } else {
        std::cout << line << std::endl;
        std::cerr << "Warning: 传入路径格式错误(无'/')" << line << std::endl;
        continue;
      }

      if (++cnt % 100 == 0) {
        std::cout << "processing idx: " << cnt << std::endl;
      }

      std::string full_path = txt_root + line;

      std::vector<float> keypoints(42);
      FILE* fp = fopen(full_path.c_str(), "r");
      if (!fp) {
        printf("Failed to open %s\n", full_path.c_str());
        continue;
      }
      bool read_ok = true;
      for (int i = 0; i < 21; ++i) {
        float x, y;
        if (fscanf(fp, "%f %f", &x, &y) != 2) {
          printf("Failed to read keypoint %d from %s\n", i, full_path.c_str());
          read_ok = false;
          break;
        }
        keypoints[2 * i] = x;
        keypoints[2 * i + 1] = y;
      }
      fclose(fp);
      if (!read_ok) return -1;

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
        outfile << line << "," << cls_meta->topk_class_ids[0] << "," << real_id
                << "\n";
      } else {
        std::cerr << "Warning: No_output" << line << std::endl;
      }
    }
  }
  infile.close();
  outfile.close();
  return 0;
}
