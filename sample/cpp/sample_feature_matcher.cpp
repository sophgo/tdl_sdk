#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "matcher/base_matcher.hpp"

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

// 打印匹配结果
void printMatchResults(const MatchResult& results) {
  std::cout << "TopK匹配结果:" << std::endl;
  for (size_t i = 0; i < results.indices.size(); ++i) {
    std::cout << "  查询特征 " << i << " 的匹配结果:" << std::endl;
    const auto& indices = results.indices[i];
    const auto& scores = results.scores[i];

    for (size_t j = 0; j < indices.size() && j < scores.size(); ++j) {
      std::cout << "    特征库索引: " << std::setw(2) << indices[j]
                << ", 相似度分数: " << std::fixed << std::setprecision(6)
                << scores[j] << std::endl;
    }
  }
  std::cout << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " <matcher_type> <gallery_path> <query_path>" << std::endl;
    std::cerr << "matcher_type: bm, cvi, 或 cpu" << std::endl;
    return 1;
  }

  std::string matcher_type = argv[1];
  std::string gallery_path = argv[2];
  std::string query_path = argv[3];

  // 初始化匹配器
  std::cout << "初始化" << matcher_type << "Matcher..." << std::endl;
  std::shared_ptr<BaseMatcher> matcher = BaseMatcher::getMatcher(matcher_type);

  // 加载底库特征
  std::cout << "加载底库特征..." << std::endl;
  auto gallery_features = loadFeaturesFromFile(gallery_path);
  if (gallery_features.empty()) {
    throw std::runtime_error("底库特征为空");
  }

  // 加载查询特征
  std::cout << "加载查询特征..." << std::endl;
  auto query_features = loadFeaturesFromFile(query_path);
  if (query_features.empty()) {
    throw std::runtime_error("查询特征为空");
  }

  // 创建特征信息对象并加载到匹配器
  std::cout << "创建特征库信息对象..." << std::endl;
  auto gallery_infos = createModelFeatureInfos(gallery_features);

  std::cout << "加载特征库..." << std::endl;
  int32_t result = matcher->loadGallery(gallery_infos);
  if (result != 0) {
    throw std::runtime_error("特征库加载失败，返回码: " +
                             std::to_string(result));
  }
  std::cout << "特征库加载成功！" << std::endl;

  // 创建查询特征信息对象
  std::cout << "创建查询特征信息对象..." << std::endl;
  auto query_infos = createModelFeatureInfos(query_features);

  // 执行查询
  std::cout << "执行特征匹配查询..." << std::endl;
  MatchResult results;
  int topk = 5;  // 设置返回前5个最相似的结果
  int32_t query_result = matcher->queryWithTopK(query_infos, topk, results);

  if (query_result != 0) {
    throw std::runtime_error("查询失败，返回码: " +
                             std::to_string(query_result));
  }

  // 输出结果
  printMatchResults(results);

  std::cout << "程序执行完毕" << std::endl;
  return 0;
}