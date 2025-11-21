#include <sys/stat.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include "tdl_model_factory.hpp"
#include "utils/common_utils.hpp"
#include "utils/tokenizer_bpe.hpp"
// 读取图片路径列表
std::vector<std::string> read_image_list(const std::string& list_file,
                                         std::vector<int>& real_classes) {
  std::vector<std::string> image_paths;
  std::ifstream infile(list_file);
  if (!infile.is_open()) {
    printf("Failed to open image list file: %s\n", list_file.c_str());
    return image_paths;
  }
  std::string line;
  while (std::getline(infile, line)) {
    if (!line.empty()) {
      std::stringstream ss(line);
      std::string real_cls_str;
      std::getline(ss, real_cls_str, '/');
      int real_cls = std::stoi(real_cls_str);
      real_classes.push_back(real_cls);

      image_paths.push_back(line);
    }
  }
  infile.close();
  return image_paths;
}

// 检查模型ID是否在允许的列表中
int is_valid_model_id(const char* id, const char* const valid_ids[],
                      int count) {
  for (int i = 0; i < count; i++) {
    if (strcmp(id, valid_ids[i]) == 0) {
      return 1;  // 有效ID
    }
  }
  return 0;  // 无效ID
}

bool file_exists(const std::string& path) {
  struct stat buf;
  return (stat(path.c_str(), &buf) == 0 && S_ISREG(buf.st_mode));
}
// 打印可用的模型ID
void print_valid_ids(const char* title, const char* const valid_ids[],
                     int count) {
  printf("%s 可用选项: ", title);
  for (int i = 0; i < count; i++) {
    printf("%s", valid_ids[i]);
    if (i < count - 1) {
      printf(", ");
    }
  }
  printf("\n");
}
int main(int argc, char** argv) {
  // 定义支持的图像模型ID
  const char* valid_img_models[] = {"FEATURE_CLIP_IMG",
                                    "FEATURE_MOBILECLIP2_IMG"};
  const int img_model_count =
      sizeof(valid_img_models) / sizeof(valid_img_models[0]);

  // 定义支持的文本模型ID
  const char* valid_text_models[] = {
      "FEATURE_CLIP_TEXT",
      "FEATURE_MOBILECLIP2_TEXT",
  };
  const int text_model_count =
      sizeof(valid_text_models) / sizeof(valid_text_models[0]);

  bool is_abspath_mode = (argc == 9);
  if (!(argc == 8 || argc == 9)) {
    printf("Usage (8参数):\n");
    printf(
        "  %s <model_dir> <img_model_id> <text_model_id> "
        "<image_root> <image_list_txt> <text_dir> <output_result_txt>\n",
        argv[0]);
    printf("Usage (9参数):\n");
    printf(
        "  %s <img_model_id> <text_model_id> "
        "<img_model_path> <text_model_path> "
        "<image_root> <image_list_txt> <text_dir> <output_result_txt>\n",
        argv[0]);
    return -1;
  }

  std::string img_model_id_name, text_model_id_name;
  std::string model_dir, img_model_path, text_model_path;
  std::string image_root, image_list_file, txt_dir, output_result_file;

  if (argc == 8) {
    model_dir = argv[1];
    img_model_id_name = argv[2];
    text_model_id_name = argv[3];
    image_root = argv[4];
    image_list_file = argv[5];
    txt_dir = argv[6];
    output_result_file = argv[7];
  } else {
    img_model_id_name = argv[1];
    text_model_id_name = argv[2];
    img_model_path = argv[3];
    text_model_path = argv[4];
    image_root = argv[5];
    image_list_file = argv[6];
    txt_dir = argv[7];
    output_result_file = argv[8];
  }

  // 检查图像模型ID是否有效
  if (!is_valid_model_id(img_model_id_name.c_str(), valid_img_models,
                         img_model_count)) {
    printf("错误: 无效的图像模型ID: %s\n", img_model_id_name.c_str());
    print_valid_ids("可用的图像模型ID", valid_img_models, img_model_count);
    return -1;
  }

  // 检查文本模型ID是否有效
  if (!is_valid_model_id(text_model_id_name.c_str(), valid_text_models,
                         text_model_count)) {
    printf("错误: 无效的文本模型ID: %s\n", text_model_id_name.c_str());
    print_valid_ids("可用的文本模型ID", valid_text_models, text_model_count);
    return -1;
  }

  if (is_abspath_mode) {
    if (!file_exists(img_model_path)) {
      printf("错误: 图像模型文件不存在: %s\n", img_model_path.c_str());
      return -1;
    }
    if (!file_exists(text_model_path)) {
      printf("错误: 文本模型文件不存在: %s\n", text_model_path.c_str());
      return -1;
    }
  }

  if (!image_root.empty() && image_root.back() != '/') {
    image_root += "/";
  }
  std::string encoder_file = txt_dir + "/encoder.txt";
  std::string bpe_file = txt_dir + "/vocab.txt";
  std::string input_file = txt_dir + "/input.txt";

  // 1. 读取图片路径列表
  std::vector<int> real_classes;
  std::vector<std::string> image_rel_paths =
      read_image_list(image_list_file, real_classes);
  if (image_rel_paths.empty()) {
    printf("No image paths found in file\n");
    return -1;
  }
  if (image_rel_paths.size() != real_classes.size()) {
    printf("Error: real classes count not match image count\n");
    return -1;
  }
  // 2. 加载模型工厂和模型（只加载一次）
  TDLModelFactory& model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();

  std::shared_ptr<BaseModel> model_clip_image;
  std::shared_ptr<BaseModel> model_clip_text;
  if (argc == 8) {
    model_factory.setModelDir(model_dir);
    model_clip_image = model_factory.getModel(img_model_id_name);
    model_clip_text = model_factory.getModel(text_model_id_name);
  } else {
    model_clip_image =
        model_factory.getModel(img_model_id_name, img_model_path);
    model_clip_text =
        model_factory.getModel(text_model_id_name, text_model_path);
  }

  if (!model_clip_image) {
    printf("Failed to load clip image model\n");
    return -1;
  }

  if (!model_clip_text) {
    printf("Failed to load clip text model\n");
    return -1;
  }

  // 3. 文本特征处理（只做一次）
  std::vector<std::vector<int32_t>> tokens;
  BytePairEncoder bpe(encoder_file, bpe_file);
  int result = bpe.tokenizerBPE(input_file, tokens);
  if (result != 0) {
    printf("Failed to tokenize text file\n");
    return -1;
  }

  std::vector<std::shared_ptr<BaseImage>> input_texts;
  for (size_t i = 0; i < tokens.size(); ++i) {
    std::shared_ptr<BaseImage> text = ImageFactory::createImage(
        77, 1, ImageFormat::GRAY, TDLDataType::INT32, true);
    uint8_t* txt_buffer = text->getVirtualAddress()[0];
    memcpy(txt_buffer, tokens[i].data(), 77 * sizeof(int32_t));
    input_texts.push_back(text);
  }

  std::vector<std::shared_ptr<ModelOutputInfo>> out_txt;
  model_clip_text->inference(input_texts, out_txt);

  if (out_txt.empty()) {
    printf("No text features extracted\n");
    return -1;
  }

  std::vector<std::vector<float>> text_features;
  for (size_t i = 0; i < out_txt.size(); i++) {
    std::shared_ptr<ModelFeatureInfo> feature_meta =
        std::static_pointer_cast<ModelFeatureInfo>(out_txt[i]);
    std::vector<float> feature_vec(feature_meta->embedding_num);
    float* feature_ptr = reinterpret_cast<float*>(feature_meta->embedding);
    for (size_t j = 0; j < feature_meta->embedding_num; j++) {
      feature_vec[j] = feature_ptr[j];
    }
    CommonUtils::normalize(feature_vec);
    text_features.push_back(feature_vec);
  }

  // 打开输出文件
  std::ofstream outfile(output_result_file);
  if (!outfile.is_open()) {
    printf("Failed to open output file: %s\n", output_result_file.c_str());
    return -1;
  }

  // 4. 逐张图片读取、推理、计算相似度，写入结果
  for (size_t i = 0; i < image_rel_paths.size(); ++i) {
    std::string image_full_path = image_root + image_rel_paths[i];
    auto image = ImageFactory::readImage(image_full_path);
    if (!image) {
      printf("Failed to load image: %s\n", image_full_path.c_str());
      continue;  // 跳过这张图片
    }

    std::vector<std::shared_ptr<BaseImage>> input_image = {image};
    std::vector<std::shared_ptr<ModelOutputInfo>> out_img;
    model_clip_image->inference(input_image, out_img);

    if (out_img.empty()) {
      printf("No image features extracted for image: %s\n",
             image_full_path.c_str());
      continue;
    }

    std::shared_ptr<ModelFeatureInfo> feature_meta =
        std::static_pointer_cast<ModelFeatureInfo>(out_img[0]);
    std::vector<float> image_feature(feature_meta->embedding_num);
    float* feature_ptr = reinterpret_cast<float*>(feature_meta->embedding);
    for (size_t j = 0; j < feature_meta->embedding_num; j++) {
      image_feature[j] = feature_ptr[j];
    }
    CommonUtils::normalize(image_feature);

    // 计算相似度
    std::vector<float> logits;
    for (size_t j = 0; j < text_features.size(); ++j) {
      float sim = CommonUtils::dot_product(image_feature, text_features[j]);
      logits.push_back(sim * 100.0f);
    }
    std::vector<float> probs = CommonUtils::softmax(logits);

    // 找最大概率对应的文本索引
    size_t max_idx = 0;
    float max_prob = probs[0];
    for (size_t j = 1; j < probs.size(); ++j) {
      if (probs[j] > max_prob) {
        max_prob = probs[j];
        max_idx = j;
      }
    }

    // 输出到控制台
    printf("Image %zu (%s) similarity:\n", i, image_full_path.c_str());
    for (size_t j = 0; j < probs.size(); ++j) {
      printf("  Text %zu: %.6f\n", j, probs[j]);
    }
    printf("  --> Max probability text index: %zu, prob: %.6f\n\n", max_idx,
           max_prob);

    // 写入结果文件：格式：图片路径 TAB 最大文本索引 TAB 最大概率
    outfile << image_full_path << "," << max_idx << "," << real_classes[i]
            << "\n";
  }

  outfile.close();

  return 0;
}