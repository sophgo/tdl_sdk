#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "tdl_model_factory.hpp"

std::string get_filename(const std::string& path) {
  size_t pos = path.find_last_of("/\\");
  if (pos == std::string::npos) return path;
  return path.substr(pos + 1);
}

std::string remove_extension(const std::string& filename) {
  size_t dot = filename.find_last_of('.');
  return (dot == std::string::npos) ? filename : filename.substr(0, dot);
}

static std::string dump_all_poses_to_txt(
    const std::vector<ObjectBoxLandmarkInfo>& box_landmarks, float image_width,
    float image_height, const std::string& img_path,
    const std::string& out_dir = ".") {
  std::string filename = get_filename(img_path);
  std::string prefix = remove_extension(filename);

  std::ostringstream oss;
  oss << out_dir << "/" << prefix << ".txt";
  std::string txt_path = oss.str();

  std::ofstream fout(txt_path);
  if (!fout.is_open()) {
    std::cerr << "Failed to open file for writing: " << txt_path << std::endl;
    return "";
  }
  fout << std::fixed << std::setprecision(6);

  // x1 y1 x2 y2 ... x17 y17
  for (const auto& one : box_landmarks) {
    for (int k = 0; k < 17; ++k) {
      fout << one.landmarks_x[k] / image_width << " "
           << one.landmarks_y[k] / image_height << "\n";
    }
    fout << "\n";
  }
  return txt_path;
}

int make_dir(const std::string& path) {
  struct stat st;
  if (stat(path.c_str(), &st) == 0) {
    if (S_ISDIR(st.st_mode)) return 0;
    return -1;
  }
  size_t pos = path.find_last_of("/\\");
  if (pos != std::string::npos) {
    if (make_dir(path.substr(0, pos)) != 0) return -1;
  }
  return mkdir(path.c_str(), 0755);
}

int main(int argc, char* argv[]) {
  if (argc != 6) {
    printf(
        "Usage: %s <model_id_name> <model_dir> <image_dir> <image_list.txt> "
        "<result_dir>\n",
        argv[0]);
    return -1;
  }

  std::string model_id_name = argv[1];
  std::string model_dir = argv[2];
  std::string image_dir = argv[3];  // 新增：图像根目录
  std::string list_path = argv[4];  // 图像列表（可为相对路径）
  std::string res_dir = argv[5];

  // 创建结果目录
  if (make_dir(res_dir) != 0) {
    std::cerr << "Failed to create result directory: " << res_dir << std::endl;
    return -1;
  }

  // 加载模型配置
  TDLModelFactory& model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();

  // 加载模型
  std::shared_ptr<BaseModel> model;
  struct stat path_stat;
  if (stat(model_dir.c_str(), &path_stat) == 0) {
    if (S_ISDIR(path_stat.st_mode)) {
      model_factory.setModelDir(model_dir);
      model = model_factory.getModel(model_id_name);
    } else if (S_ISREG(path_stat.st_mode)) {
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
    std::cerr << "Load model failed\n";
    return -1;
  }

  // 打开图像列表文件（支持相对路径）
  std::ifstream fin(list_path);
  if (!fin.is_open()) {
    std::perror(("open " + list_path).c_str());
    return -1;
  }

  std::string line;
  size_t img_idx = 0;
  while (std::getline(fin, line)) {
    if (line.empty() || line[0] == '#') continue;  // 跳过空行和注释

    // 拼接完整图像路径：image_dir / line
    std::string img_path = image_dir;
    if (!img_path.empty() && img_path.back() != '/' && !line.empty() &&
        line[0] != '/') {
      img_path += "/";
    }
    img_path += line;

    ++img_idx;
    std::cout << "processing " << img_path << "\n";

    auto img = ImageFactory::readImage(img_path);
    if (!img) {
      std::cerr << "readImage fail: " << img_path << '\n';
      continue;
    }

    std::vector<std::shared_ptr<ModelOutputInfo>> outs;
    model->inference({img}, outs);
    if (outs.empty()) {
      std::cerr << "No output from model for: " << img_path << '\n';
      continue;
    }

    uint32_t image_width = img->getWidth();
    uint32_t image_height = img->getHeight();

    auto meta = std::static_pointer_cast<ModelBoxLandmarkInfo>(outs[0]);
    std::string txt_path = dump_all_poses_to_txt(
        meta->box_landmarks, image_width, image_height, img_path, res_dir);
    if (!txt_path.empty()) {
      std::cout << "  saved: " << txt_path << std::endl;
    }

    if (img_idx % 20 == 0) {
      std::cout << "processed " << img_idx << " images\n";
    }
  }

  std::cout << "All done. Results saved to " << res_dir << '\n';
  return 0;
}