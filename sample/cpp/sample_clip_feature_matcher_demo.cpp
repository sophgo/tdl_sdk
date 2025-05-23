#include <experimental/filesystem>
#include <iostream>
#include <string>
#include <vector>
#include "matcher/base_matcher.hpp"
#include "tdl_model_factory.hpp"
#include "utils/tokenizer_bpe.hpp"

namespace fs = std::experimental::filesystem;

std::vector<std::string> getImagesFromDir(const std::string& image_dir) {
  std::vector<std::string> image_paths;
  for (const auto& entry : fs::directory_iterator(image_dir)) {
    if (entry.path().extension() == ".jpg" ||
        entry.path().extension() == ".png" ||
        entry.path().extension() == ".jpeg") {
      image_paths.push_back(entry.path().string());
    }
  }
  return image_paths;
}

int main(int argc, char** argv) {
  if (argc != 6) {
    printf(
        "Usage: %s <model_dir> <image_dir> <encoder_file> <bpe_file> "
        "<text_file>\n",
        argv[0]);
    return -1;
  }

  // ------------------------------初始化参数------------------------------

  std::string model_dir = argv[1];
  std::string image_dir = argv[2];
  std::string encoder_file = argv[3];
  std::string bpe_file = argv[4];
  std::string text_file = argv[5];

  // 初始化模型工厂
  TDLModelFactory& model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();
  model_factory.setModelDir(model_dir);

  // 加载CLIP图像模型
  std::shared_ptr<BaseModel> model_clip_image =
      model_factory.getModel(ModelType::CLIP_FEATURE_IMG);
  if (!model_clip_image) {
    printf("Failed to load clip image model\n");
    return -1;
  }

  // 加载CLIP文本模型
  std::shared_ptr<BaseModel> model_clip_text =
      model_factory.getModel(ModelType::CLIP_FEATURE_TEXT);
  if (!model_clip_text) {
    printf("Failed to load clip text model\n");
    return -1;
  }

  // 获取图像路径列表
  std::vector<std::string> image_paths = getImagesFromDir(image_dir);
  if (image_paths.empty()) {
    printf("No images found in directory: %s\n", image_dir.c_str());
    return -1;
  }

  // ------------------------------提取底库图像特征------------------------------

  // 提取图像特征
  std::vector<std::shared_ptr<ModelFeatureInfo>> gallery_features;
  std::vector<std::string> image_labels;
  std::shared_ptr<BaseImage> image;
  std::vector<std::shared_ptr<BaseImage>> input_images;
  std::shared_ptr<ModelFeatureInfo> feature_info;
  image_labels.reserve(image_paths.size());
  input_images.reserve(image_paths.size());
  for (const std::string& image_path : image_paths) {
    image = ImageFactory::readImage(image_path);
    if (!image) {
      printf("Failed to load image: %s\n", image_path.c_str());
      continue;
    }
    input_images.push_back(image);
  }
  std::vector<std::shared_ptr<ModelOutputInfo>> out_features;
  model_clip_image->inference(input_images, out_features);
  for (size_t i = 0; i < out_features.size(); i++) {
    const std::shared_ptr<ModelOutputInfo>& out_feature = out_features[i];
    feature_info = std::static_pointer_cast<ModelFeatureInfo>(out_feature);
    gallery_features.push_back(feature_info);
    image_labels.push_back(fs::path(image_paths[i]).filename().string());
  }

  printf("Extracted features from %zu images\n", gallery_features.size());

  // ------------------------------提取文本特征------------------------------

  // 处理文本并提取特征
  std::vector<std::vector<int32_t>> tokens;
  BytePairEncoder bpe(encoder_file, bpe_file);
  int result = bpe.tokenizerBPE(text_file, tokens);
  if (result != 0) {
    printf("Failed to tokenize text\n");
    return -1;
  }

  std::vector<std::shared_ptr<BaseImage>> input_texts;
  input_texts.reserve(tokens.size());
  std::shared_ptr<BaseImage> text;
  for (const std::vector<int32_t>& token_vec : tokens) {
    text = ImageFactory::createImage(77, 1, ImageFormat::GRAY,
                                     TDLDataType::INT32, true);
    uint8_t* txt_buffer = text->getVirtualAddress()[0];
    memcpy(txt_buffer, token_vec.data(), 77 * sizeof(int32_t));
    input_texts.push_back(text);
  }

  std::vector<std::shared_ptr<ModelOutputInfo>> text_output_features;
  model_clip_text->inference(input_texts, text_output_features);

  std::vector<std::shared_ptr<ModelFeatureInfo>> query_features;
  query_features.reserve(text_output_features.size());
  for (const std::shared_ptr<ModelOutputInfo>& out_info :
       text_output_features) {
    feature_info = std::static_pointer_cast<ModelFeatureInfo>(out_info);
    query_features.push_back(feature_info);
  }

  // ------------------------------特征匹配------------------------------

  // 创建特征匹配器
  std::shared_ptr<BaseMatcher> matcher = BaseMatcher::getMatcher("bm");
  if (!matcher) {
    printf("Failed to create matcher\n");
    return -1;
  }

  // 加载特征库
  matcher->loadGallery(gallery_features);

  // 执行特征匹配
  MatchResult match_results;
  const int32_t topk = 5;  // 返回前5个最相似的结果
  matcher->queryWithTopK(query_features, topk, match_results);

  // 输出匹配结果
  printf("\nMatching Results:\n");
  for (size_t i = 0; i < match_results.indices.size(); i++) {
    printf("\nQuery %zu results:\n", i);
    for (size_t j = 0; j < match_results.indices[i].size(); j++) {
      int idx = match_results.indices[i][j];
      float score = match_results.scores[i][j];
      printf("Rank %zu: %s (Score: %.4f)\n", j + 1, image_labels[idx].c_str(),
             score);
    }
  }

  return 0;
}
