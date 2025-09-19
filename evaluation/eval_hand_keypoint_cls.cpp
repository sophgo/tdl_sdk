#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "tdl_model_factory.hpp"

int main(int argc, char** argv) {
  if (argc != 4) {
    printf("Usage: %s <model_name> <model_dir> <input_txt_list> <result_txt>\n",
           argv[0]);
    return -1;
  }
  std::string model_name = argv[1];
  std::string model_dir = argv[2];
  std::string input_txt_list = argv[3];
  std::string result_txt = argv[4];

  std::vector<std::string> keypoints_files;
  std::ifstream infile(input_txt_list);
  std::string line;
  while (std::getline(infile, line)) {
    if (!line.empty()) {
      keypoints_files.push_back(line);
    }
  }
  infile.close();

  TDLModelFactory& model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();
  model_factory.setModelDir(model_dir);
  std::shared_ptr<BaseModel> model_hc = model_factory.getModel(model_name);

  if (!model_hc) {
    printf("Failed to create model_hc\n");
    return -1;
  }

  std::ofstream outfile(result_txt);
  if (!outfile.is_open()) {
    printf("Failed to open %s\n", result_txt.c_str());
    return -1;
  }

  for (const auto& keypoints_txt : keypoints_files) {
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
      outfile << keypoints_txt << " " << cls_meta->topk_class_ids[0] << " "
              << cls_meta->topk_scores[0] << "\n";
    } else {
      outfile << keypoints_txt << " No_output\n";
    }
  }

  outfile.close();
  return 0;
}
