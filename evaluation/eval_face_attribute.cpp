#include <fstream>
#include <iostream>
#include <string>
#include "tdl_model_factory.hpp"

int main(int argc, char **argv) {
  if (argc != 4) {
    printf(
        "Usage: %s <model_id_name> <model_dir> <input_txt_path> "
        "<output_txt_path>\n",
        argv[0]);
    return -1;
  }
  std::string model_id_name = argv[1];
  std::string model_dir = argv[2];
  std::string input_txt_path = argv[3];
  std::string output_txt_path = argv[4];

  std::ifstream input_file(input_txt_path);
  if (!input_file.is_open()) {
    printf("Failed to open input txt file: %s\n", input_txt_path.c_str());
    return -1;
  }

  std::ofstream output_file(output_txt_path);
  if (!output_file.is_open()) {
    printf("Failed to open output txt file: %s\n", output_txt_path.c_str());
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

  std::string line;
  while (std::getline(input_file, line)) {
    if (line.empty()) {
      output_file << "\n";
      continue;
    }

    // 找到第一个 '|' 分隔符
    size_t sep_pos = line.find('|');
    std::string image_path;
    std::string rest;
    if (sep_pos != std::string::npos) {
      image_path = line.substr(0, sep_pos);
      rest = line.substr(sep_pos);  // 包含 '|'
    } else {
      image_path = line;
      rest = "";
    }

    if (image_path.empty()) {
      output_file << line << " - Failed: empty image path\n";
      continue;
    }

    std::shared_ptr<BaseImage> image = ImageFactory::readImage(image_path);
    if (!image) {
      printf("Failed to create image: %s\n", image_path.c_str());
      output_file << line << " - Failed to load image\n";
      continue;
    }

    std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
    std::vector<std::shared_ptr<BaseImage>> input_images = {image};
    model->inference(input_images, out_datas);

    // 构造结果字符串
    std::string result_str;

    for (size_t i = 0; i < out_datas.size(); i++) {
      if (out_datas[i]->getType() != ModelOutputType::CLS_ATTRIBUTE) {
        continue;
      }
      std::shared_ptr<ModelAttributeInfo> face_meta =
          std::static_pointer_cast<ModelAttributeInfo>(out_datas[i]);

      // 生成结果字符串
      result_str =
          "|pred_gender: " +
          std::string(face_meta->attributes[TDLObjectAttributeType::
                                                OBJECT_ATTRIBUTE_HUMAN_GENDER] >
                              0.5
                          ? "Male"
                          : "Female") +
          "|pred_age: " +
          std::to_string(
              int(face_meta->attributes
                      [TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_AGE] *
                  100)) +
          "|pred_glass: " +
          (face_meta->attributes[TDLObjectAttributeType::
                                     OBJECT_ATTRIBUTE_HUMAN_GLASSES] > 0.5
               ? "Yes"
               : "No") +
          "|pred_mask: " +
          (face_meta->attributes
                       [TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_MASK] >
                   0.5
               ? "Yes"
               : "No");
      break;  // 只处理第一个符合的
    }

    // 写入：原行 + 结果
    output_file << line << result_str << "\n";
  }

  input_file.close();
  output_file.close();

  return 0;
}
