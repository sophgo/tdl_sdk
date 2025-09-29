#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "tdl_model_factory.hpp"
#include "utils/tokenizer_bpe.hpp"

// 读取图片路径列表
std::vector<std::string> read_image_list(const std::string& list_file) {
  std::vector<std::string> image_paths;
  std::ifstream infile(list_file);
  if (!infile.is_open()) {
    printf("Failed to open image list file: %s\n", list_file.c_str());
    return image_paths;
  }
  std::string line;
  while (std::getline(infile, line)) {
    if (!line.empty()) {
      image_paths.push_back(line);
    }
  }
  return image_paths;
}

// 计算点积
float dot_product(const std::vector<float>& a, const std::vector<float>& b) {
  float sum = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
  return sum;
}

// L2归一化
void normalize(std::vector<float>& v) {
  float norm = 0.0f;
  for (float f : v) norm += f * f;
  norm = std::sqrt(norm);
  if (norm > 1e-6)
    for (float& f : v) f /= norm;
}

std::vector<float> softmax(const std::vector<float>& logits) {
  std::vector<float> result(logits.size());
  float max_logit = *std::max_element(logits.begin(), logits.end());
  float sum = 0.0f;
  for (size_t i = 0; i < logits.size(); ++i) {
    result[i] = std::exp(logits[i] - max_logit);  // for numerical stability
    sum += result[i];
  }
  for (float& v : result) v /= sum;
  return result;
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

  if (argc != 7) {
    printf(
        "Usage: %s <model_dir> <img_model_id_name> <text_model_id_name> "
        "<image_list_txt> "
        "<text_dir> <output_result_txt>\n",
        argv[0]);
    return -1;
  }
  // 检查图像模型ID是否有效
  if (!is_valid_model_id(argv[2], valid_img_models, img_model_count)) {
    printf("错误: 无效的图像模型ID: %s\n", argv[2]);
    print_valid_ids("可用的图像模型ID", valid_img_models, img_model_count);
    return -1;
  }

  // 检查文本模型ID是否有效
  if (!is_valid_model_id(argv[3], valid_text_models, text_model_count)) {
    printf("错误: 无效的文本模型ID: %s\n", argv[3]);
    print_valid_ids("可用的文本模型ID", valid_text_models, text_model_count);
    return -1;
  }
  std::string model_dir = argv[1];
  std::string img_model_id_name = argv[2];
  std::string text_model_id_name = argv[3];
  std::string image_list_file = argv[4];  // 图片路径列表txt文件
  std::string txt_dir = argv[5];
  std::string encoder_file = txt_dir + "/encoder.txt";
  std::string bpe_file = txt_dir + "/vocab.txt";
  std::string input_file = txt_dir + "/input.txt";
  std::string output_result_file = argv[6];  // 新增输出文件路径

  // 1. 读取图片路径列表
  std::vector<std::string> image_paths = read_image_list(image_list_file);
  if (image_paths.empty()) {
    printf("No image paths found in file\n");
    return -1;
  }

  // 2. 加载模型工厂和模型（只加载一次）
  TDLModelFactory& model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();
  model_factory.setModelDir(model_dir);

  std::shared_ptr<BaseModel> model_clip_image =
      model_factory.getModel(img_model_id_name);
  if (!model_clip_image) {
    printf("Failed to load clip image model\n");
    return -1;
  }

  std::shared_ptr<BaseModel> model_clip_text =
      model_factory.getModel(text_model_id_name);
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
  // model_clip_text->inference(input_texts, out_txt);
  for (int i = 0; i < 1001; i++) {
    out_txt.clear();
    model_clip_text->inference(input_texts, out_txt);
  }

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
    normalize(feature_vec);
    text_features.push_back(feature_vec);
  }

  // 打开输出文件
  std::ofstream outfile(output_result_file);
  if (!outfile.is_open()) {
    printf("Failed to open output file: %s\n", output_result_file.c_str());
    return -1;
  }

  // 4. 逐张图片读取、推理、计算相似度，写入结果
  for (size_t i = 0; i < image_paths.size(); ++i) {
    auto image = ImageFactory::readImage(image_paths[i]);
    if (!image) {
      printf("Failed to load image: %s\n", image_paths[i].c_str());
      continue;  // 跳过这张图片
    }

    std::vector<std::shared_ptr<BaseImage>> input_image = {image};
    std::vector<std::shared_ptr<ModelOutputInfo>> out_img;
    model_clip_image->inference(input_image, out_img);

    if (out_img.empty()) {
      printf("No image features extracted for image: %s\n",
             image_paths[i].c_str());
      continue;
    }

    std::shared_ptr<ModelFeatureInfo> feature_meta =
        std::static_pointer_cast<ModelFeatureInfo>(out_img[0]);
    std::vector<float> image_feature(feature_meta->embedding_num);
    float* feature_ptr = reinterpret_cast<float*>(feature_meta->embedding);
    for (size_t j = 0; j < feature_meta->embedding_num; j++) {
      image_feature[j] = feature_ptr[j];
    }
    normalize(image_feature);

    // 计算相似度
    std::vector<float> logits;
    for (size_t j = 0; j < text_features.size(); ++j) {
      float sim = dot_product(image_feature, text_features[j]);
      logits.push_back(sim * 100.0f);
    }
    std::vector<float> probs = softmax(logits);

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
    printf("Image %zu (%s) similarity:\n", i, image_paths[i].c_str());
    for (size_t j = 0; j < probs.size(); ++j) {
      printf("  Text %zu: %.6f\n", j, probs[j]);
    }
    printf("  --> Max probability text index: %zu, prob: %.6f\n\n", max_idx,
           max_prob);

    // 写入结果文件：格式：图片路径 TAB 最大文本索引 TAB 最大概率
    outfile << image_paths[i] << "\t" << max_idx << "\t" << max_prob << "\n";
  }

  outfile.close();

  return 0;
}