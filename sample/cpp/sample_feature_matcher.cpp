#include <cstring>
#include <experimental/filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
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

void checkModelIdName(char* argv[]) {
  // 图像模型
  std::string img_model_id_name = argv[1];
  std::string text_model_id_name = argv[2];
  if (img_model_id_name != "FEATURE_CLIP_IMG") {
    std::cerr << "model_id_name: " << img_model_id_name << " not supported"
              << std::endl;
    exit(1);
  }
  // 文本模型
  if (text_model_id_name != "FEATURE_CLIP_TEXT") {
    std::cerr << "model_id_name: " << text_model_id_name << " not supported"
              << std::endl;
    exit(1);
  }
}

// 从文本文件加载特征数据
std::vector<std::vector<float>> loadFeaturesFromFile(
    const std::string& filepath) {
  std::vector<std::vector<float>> features;
  std::ifstream file(filepath);
  std::string line;

  if (!file.is_open()) {
    throw std::runtime_error("无法打开文件: " + filepath);
  }

  while (std::getline(file, line)) {
    std::vector<float> feature;
    std::istringstream iss(line);
    float value;

    while (iss >> value) {
      feature.push_back(value);
    }

    if (!feature.empty()) {
      features.push_back(feature);
    }
  }

  file.close();
  return features;
}

// 创建特征信息对象
std::vector<std::shared_ptr<ModelFeatureInfo>> createModelFeatureInfos(
    const std::vector<std::vector<float>>& features) {
  std::vector<std::shared_ptr<ModelFeatureInfo>> feature_infos;

  for (const auto& feature : features) {
    auto feature_info = std::make_shared<ModelFeatureInfo>();
    int feature_dim = feature.size();

    // 分配内存并转换为UINT8
    feature_info->embedding = new uint8_t[feature_dim * sizeof(uint8_t)];
    uint8_t* dest = reinterpret_cast<uint8_t*>(feature_info->embedding);

    for (int j = 0; j < feature_dim; j++) {
      dest[j] = static_cast<uint8_t>(feature[j]);
    }

    feature_info->embedding_num = feature_dim;
    feature_info->embedding_type = TDLDataType::UINT8;
    feature_infos.push_back(feature_info);
  }

  return feature_infos;
}

void mul100_and_colSoftmax(std::vector<std::vector<float>>& scores) {
  if (scores.empty()) return;

  const std::size_t rows = scores.size();
  const std::size_t cols = scores[0].size();
  /* 1. 先全部乘 100 */
  for (auto& row : scores)
    for (auto& v : row) v *= 100.f;
  /* 2. 逐列做 soft-max */
  std::vector<float> colSum(cols, 0.f);
  // 2-a) 先做 exp，同时累计每列的和
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t c = 0; c < cols; ++c) {
      scores[r][c] = std::exp(scores[r][c]);
      colSum[c] += scores[r][c];
    }
  // 2-b) 再把每个元素除以对应列的和
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t c = 0; c < cols; ++c)
      scores[r][c] /= (colSum[c] + 1e-8f);  // 加个 eps 防止除 0
}
// 打印匹配结果
void printMatchResults(int argc, char* argv[], const int32_t topk,
                       MatchResult& results) {
  std::string img_dir;
  std::vector<std::string> image_paths;
  std::vector<std::string> image_labels;
  if (argc == 6) {
    img_dir = argv[4];
    image_paths = getImagesFromDir(img_dir);
    image_labels.reserve(image_paths.size());
    for (const std::string& image_path : image_paths) {
      image_labels.push_back(fs::path(image_path).filename().string());
    }

    mul100_and_colSoftmax(results.scores);
  }
  std::cout << "Top " << topk << "匹配结果:" << std::endl;
  for (size_t i = 0; i < results.indices.size(); ++i) {
    std::cout << "  查询特征 " << i << " 的匹配结果:" << std::endl;
    const auto& indices = results.indices[i];
    const auto& scores = results.scores[i];
    for (size_t j = 0; j < indices.size() && j < scores.size(); ++j) {
      if (argc == 6) {
        std::cout << " 特征库图像: " << std::setw(2) << image_labels[indices[j]]
                  << ", 相似度分数: " << std::fixed << std::setprecision(6)
                  << scores[j] << std::endl;
      } else {
        std::cout << " 特征库索引: " << std::setw(2) << indices[j]
                  << ", 相似度分数: " << std::fixed << std::setprecision(6)
                  << scores[j] << std::endl;
      }
    }
    std::cout << std::endl;
  }
}

void prepareFeatureFromImage(
    const std::shared_ptr<BaseModel>& model, const std::string& image_dir,
    std::vector<std::shared_ptr<ModelFeatureInfo>>& gallery_features) {
  // 获取图像路径列表
  std::vector<std::string> image_paths = getImagesFromDir(image_dir);
  if (image_paths.empty()) {
    printf("No images found in directory: %s\n", image_dir.c_str());
    exit(1);
  }

  std::vector<std::shared_ptr<BaseImage>> input_images;
  input_images.reserve(image_paths.size());
  for (const std::string& image_path : image_paths) {
    std::shared_ptr<BaseImage> image = ImageFactory::readImage(image_path);
    if (!image) {
      printf("Failed to load image: %s\n", image_path.c_str());
      continue;
    }
    input_images.push_back(image);
  }
  std::vector<std::shared_ptr<ModelOutputInfo>> out_features;
  model->inference(input_images, out_features);
  for (int i = 0; i < out_features.size(); i++) {
    const std::shared_ptr<ModelOutputInfo>& out_feature = out_features[i];
    std::shared_ptr<ModelFeatureInfo> feature_info =
        std::static_pointer_cast<ModelFeatureInfo>(out_feature);
    gallery_features.push_back(feature_info);
  }

  printf("Extracted features from %zu images\n", gallery_features.size());
}

void prepareFeatureFromText(
    const std::shared_ptr<BaseModel>& model, const std::string& txt_dir,
    std::vector<std::shared_ptr<ModelFeatureInfo>>& query_features) {
  std::string encoder_file = txt_dir + "/encoder.txt";
  std::string bpe_file = txt_dir + "/vocab.txt";
  std::string input_file = txt_dir + "/input.txt";
  // 处理文本并提取特征
  std::vector<std::vector<int32_t>> tokens;
  BytePairEncoder bpe(encoder_file, bpe_file);
  bpe.tokenizerBPE(input_file, tokens);

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
  model->inference(input_texts, text_output_features);

  query_features.reserve(text_output_features.size());
  for (const std::shared_ptr<ModelOutputInfo>& out_info :
       text_output_features) {
    std::shared_ptr<ModelFeatureInfo> feature_info =
        std::static_pointer_cast<ModelFeatureInfo>(out_info);
    query_features.push_back(feature_info);
  }
}

void prepareFeature(
    int argc, char* argv[],
    std::vector<std::shared_ptr<ModelFeatureInfo>>& gallery_features,
    std::vector<std::shared_ptr<ModelFeatureInfo>>& query_features) {
  if (argc == 4) {
    std::string matcher_type = argv[1];
    std::string gallery_txt_file = argv[2];
    std::string query_txt_file = argv[3];

    auto gallery_features_tmp = loadFeaturesFromFile(gallery_txt_file);
    auto query_features_tmp = loadFeaturesFromFile(query_txt_file);
    gallery_features = createModelFeatureInfos(gallery_features_tmp);
    query_features = createModelFeatureInfos(query_features_tmp);

  } else if (argc == 6) {
    checkModelIdName(argv);

    std::string img_model_id_name = argv[1];
    std::string text_model_id_name = argv[2];
    std::string model_dir = argv[3];
    std::string image_dir = argv[4];
    std::string txt_dir = argv[5];
    if (!txt_dir.empty() && txt_dir.back() == '/') {
      txt_dir.pop_back();
    }
    // 初始化模型工厂
    TDLModelFactory& model_factory = TDLModelFactory::getInstance();
    model_factory.loadModelConfig();
    model_factory.setModelDir(model_dir);

    // 加载CLIP图像模型
    std::shared_ptr<BaseModel> img_model =
        model_factory.getModel(img_model_id_name);

    // 加载CLIP文本模型
    std::shared_ptr<BaseModel> text_model =
        model_factory.getModel(text_model_id_name);

    prepareFeatureFromImage(img_model, image_dir, gallery_features);
    prepareFeatureFromText(text_model, txt_dir, query_features);
  }
}

void prepareMatcher(int argc, char* argv[],
                    std::shared_ptr<BaseMatcher>& matcher) {
  if (argc == 4) {
    std::string matcher_type = argv[1];
    matcher = BaseMatcher::getMatcher(matcher_type);
  } else if (argc == 6) {
    matcher = BaseMatcher::getMatcher("bm");
  }
}

int main(int argc, char* argv[]) {
  if (argc != 4 && argc != 6) {
    printf("Usage: %s <matcher_type> <gallery_txt_file> <query_txt_file>\n",
           argv[0]);
    printf("matcher_type: bm, cvi, 或 cpu\n");
    printf(
        "Usage: %s <img_model_id_name> <text_model_id_name> <model_dir> "
        "<image_dir> <txt_dir>\n",
        argv[0]);
    return -1;
  }

  // 初始化变量
  std::vector<std::shared_ptr<ModelFeatureInfo>> gallery_features;
  std::vector<std::shared_ptr<ModelFeatureInfo>> query_features;
  std::shared_ptr<BaseMatcher> matcher;
  MatchResult results;
  const int32_t topk = 5;

  // 初始化匹配器
  prepareMatcher(argc, argv, matcher);
  // 准备特征
  prepareFeature(argc, argv, gallery_features, query_features);
  // 加载特征库
  matcher->loadGallery(gallery_features);
  // 执行查询
  matcher->queryWithTopK(query_features, topk, results);

  // 输出结果
  printMatchResults(argc, argv, topk, results);

  return 0;
}