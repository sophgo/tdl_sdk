#include <gtest.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>

#include "matcher/base_matcher.hpp"

namespace cvitdl {
namespace unitest {

// 生成随机特征数据的辅助函数
void generateRandomFeatures(float* features,
                            int num_features,
                            int feature_dim,
                            int seed = 42,
                            TDLDataType type = TDLDataType::FP32) {
  std::mt19937 gen(seed);

  // 根据不同数据类型设置不同的分布范围
  if (type == TDLDataType::UINT8) {
    // UINT8范围: 0-255
    std::uniform_int_distribution<int> dist(0, 255);
    for (int i = 0; i < num_features * feature_dim; ++i) {
      features[i] = dist(gen);
    }
  } else if (type == TDLDataType::INT8) {
    // INT8范围: -128 到 127
    std::uniform_int_distribution<int> dist(-128, 127);
    for (int i = 0; i < num_features * feature_dim; ++i) {
      features[i] = dist(gen);
    }
  } else if (type == TDLDataType::FP32) {
    // FP32范围: 使用正常分布生成在[-1, 1]之间的浮点数
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < num_features * feature_dim; ++i) {
      features[i] = dist(gen);
    }
  } else {
    // 默认情况使用0-127的整数
    std::uniform_int_distribution<int> dist(0, 127);
    for (int i = 0; i < num_features * feature_dim; ++i) {
      features[i] = dist(gen);
    }
  }
}

// 创建特征信息对象
std::vector<std::shared_ptr<ModelFeatureInfo>> createModelFeatureInfos(
    float* features, int num_features, int feature_dim, TDLDataType type) {
  std::vector<std::shared_ptr<ModelFeatureInfo>> feature_infos;

  for (int i = 0; i < num_features; ++i) {
    auto feature_info = std::make_shared<ModelFeatureInfo>();

    // 根据类型分配正确大小的内存
    if (type == TDLDataType::UINT8) {
      feature_info->embedding = new uint8_t[feature_dim * sizeof(uint8_t)];
      // 转换float到uint8_t
      uint8_t* dest = reinterpret_cast<uint8_t*>(feature_info->embedding);
      for (int j = 0; j < feature_dim; j++) {
        dest[j] = static_cast<uint8_t>(features[i * feature_dim + j]);
      }
    } else if (type == TDLDataType::INT8) {
      feature_info->embedding = new uint8_t[feature_dim * sizeof(int8_t)];
      // 转换float到int8_t
      int8_t* dest = reinterpret_cast<int8_t*>(feature_info->embedding);
      for (int j = 0; j < feature_dim; j++) {
        dest[j] = static_cast<int8_t>(features[i * feature_dim + j]);
      }
    } else if (type == TDLDataType::FP32) {
      feature_info->embedding = new uint8_t[feature_dim * sizeof(float)];
      // 直接复制float数据
      memcpy(feature_info->embedding, &features[i * feature_dim],
             feature_dim * sizeof(float));
    } else {
      throw std::invalid_argument("不支持的特征类型");
    }

    feature_info->embedding_num = feature_dim;
    feature_info->embedding_type = type;

    feature_infos.push_back(feature_info);
  }

  return feature_infos;
}

// 对比两个匹配结果是否一致
bool compareMatchResults(const MatchResult& result1,
                         const MatchResult& result2,
                         float tolerance = 1e-5) {
  if (result1.indices.size() != result2.indices.size() ||
      result1.scores.size() != result2.scores.size()) {
    std::cout << "错误：结果数组大小不匹配";
    return false;
  }

  for (size_t i = 0; i < result1.indices.size(); ++i) {
    if (result1.indices[i].size() != result2.indices[i].size()) {
      std::cout << "错误：查询 " << i << " 索引数组大小不匹配";
      return false;
    }

    if (result1.scores[i].size() != result2.scores[i].size()) {
      std::cout << "错误：查询 " << i << " 分数数组大小不匹配";
      return false;
    }

    for (size_t j = 0; j < result1.indices[i].size(); ++j) {
      if (result1.indices[i][j] != result2.indices[i][j]) {
        std::cout << "错误：查询 " << i << " 的第 " << j
                  << " 个匹配索引不一致：" << result1.indices[i][j] << " vs "
                  << result2.indices[i][j];
        return false;
      }

      float score_diff = std::abs(result1.scores[i][j] - result2.scores[i][j]);
      if (score_diff > tolerance) {
        std::cout << "错误：查询 " << i << " 的第 " << j
                  << " 个匹配分数不一致：" << result1.scores[i][j] << " vs "
                  << result2.scores[i][j] << "，差值: " << score_diff;
        return false;
      }
    }
  }

  return true;
}

class MatcherTestSuite : public ::testing::Test {
 public:
  MatcherTestSuite() = default;
  virtual ~MatcherTestSuite() = default;

 protected:
  // 测试参数
  int gallery_size = 1000;  // 特征库大小
  int feature_dim = 512;    // 特征维度
  int query_size = 2;       // 查询特征数量
  int topk = 5;             // 查询TopK结果

  std::shared_ptr<BaseMatcher> test_matcher_;
  std::shared_ptr<BaseMatcher> cpu_matcher_;

  // 特征数据
  float* gallery_features;
  float* query_features;
  float* new_feature;

  std::string matcher_type_;

  virtual void SetUp() override {
    // 获取环境变量或参数来设置匹配器类型
    char* type_env = getenv("MATCHER_TYPE");
    matcher_type_ = (type_env != nullptr) ? type_env : "cvi";

    test_matcher_ = BaseMatcher::getMatcher(matcher_type_);
    cpu_matcher_ = BaseMatcher::getMatcher("cpu");

    ASSERT_NE(test_matcher_, nullptr)
        << "无法创建" << matcher_type_ << "匹配器";
    ASSERT_NE(cpu_matcher_, nullptr) << "无法创建CPU匹配器";

    // 为特征库和查询特征分配内存
    gallery_features = new float[gallery_size * feature_dim];
    query_features = new float[query_size * feature_dim];
    new_feature = new float[feature_dim];
  }

  virtual void TearDown() override {
    // 释放资源
    delete[] gallery_features;
    delete[] query_features;
    delete[] new_feature;
  }
};

TEST_F(MatcherTestSuite, UINT8FeatureMatching) {
  std::cout << "开始测试UINT8类型特征匹配" << std::endl;

  // 根据数据类型生成对应范围的随机特征数据
  generateRandomFeatures(gallery_features, gallery_size, feature_dim, 42,
                         TDLDataType::UINT8);
  generateRandomFeatures(query_features, query_size, feature_dim, 43,
                         TDLDataType::UINT8);

  // 创建特征库信息对象
  std::vector<std::shared_ptr<ModelFeatureInfo>> gallery_infos =
      createModelFeatureInfos(gallery_features, gallery_size, feature_dim,
                              TDLDataType::UINT8);

  // 加载特征库到两个Matcher
  EXPECT_EQ(cpu_matcher_->loadGallery(gallery_infos), 0);
  EXPECT_EQ(test_matcher_->loadGallery(gallery_infos), 0);

  // 创建查询特征
  std::vector<std::shared_ptr<ModelFeatureInfo>> query_infos =
      createModelFeatureInfos(query_features, query_size, feature_dim,
                              TDLDataType::UINT8);

  // 执行查询
  MatchResult results_cpu, results_test;
  EXPECT_EQ(cpu_matcher_->queryWithTopK(query_infos, topk, results_cpu), 0);
  EXPECT_EQ(test_matcher_->queryWithTopK(query_infos, topk, results_test), 0);

  // 比较结果
  EXPECT_TRUE(compareMatchResults(results_cpu, results_test));
}

TEST_F(MatcherTestSuite, INT8FeatureMatching) {
  std::cout << "开始测试INT8类型特征匹配" << std::endl;

  // 根据数据类型生成对应范围的随机特征数据
  generateRandomFeatures(gallery_features, gallery_size, feature_dim, 42,
                         TDLDataType::INT8);
  generateRandomFeatures(query_features, query_size, feature_dim, 43,
                         TDLDataType::INT8);

  // 创建特征库信息对象
  std::vector<std::shared_ptr<ModelFeatureInfo>> gallery_infos =
      createModelFeatureInfos(gallery_features, gallery_size, feature_dim,
                              TDLDataType::INT8);

  // 加载特征库到两个Matcher
  EXPECT_EQ(cpu_matcher_->loadGallery(gallery_infos), 0);
  EXPECT_EQ(test_matcher_->loadGallery(gallery_infos), 0);

  // 创建查询特征
  std::vector<std::shared_ptr<ModelFeatureInfo>> query_infos =
      createModelFeatureInfos(query_features, query_size, feature_dim,
                              TDLDataType::INT8);

  // 执行查询
  MatchResult results_cpu, results_test;
  EXPECT_EQ(cpu_matcher_->queryWithTopK(query_infos, topk, results_cpu), 0);
  EXPECT_EQ(test_matcher_->queryWithTopK(query_infos, topk, results_test), 0);

  // 比较结果
  EXPECT_TRUE(compareMatchResults(results_cpu, results_test));
}

TEST_F(MatcherTestSuite, FP32FeatureMatching) {
  std::cout << "开始测试FP32类型特征匹配" << std::endl;

  // 根据数据类型生成对应范围的随机特征数据
  generateRandomFeatures(gallery_features, gallery_size, feature_dim, 42,
                         TDLDataType::FP32);
  generateRandomFeatures(query_features, query_size, feature_dim, 43,
                         TDLDataType::FP32);

  // 创建特征库信息对象
  std::vector<std::shared_ptr<ModelFeatureInfo>> gallery_infos =
      createModelFeatureInfos(gallery_features, gallery_size, feature_dim,
                              TDLDataType::FP32);

  // 加载特征库到两个Matcher
  EXPECT_EQ(cpu_matcher_->loadGallery(gallery_infos), 0);
  EXPECT_EQ(test_matcher_->loadGallery(gallery_infos), 0);

  // 创建查询特征
  std::vector<std::shared_ptr<ModelFeatureInfo>> query_infos =
      createModelFeatureInfos(query_features, query_size, feature_dim,
                              TDLDataType::FP32);

  // 执行查询
  MatchResult results_cpu, results_test;
  EXPECT_EQ(cpu_matcher_->queryWithTopK(query_infos, topk, results_cpu), 0);
  EXPECT_EQ(test_matcher_->queryWithTopK(query_infos, topk, results_test), 0);

  // 比较结果
  EXPECT_TRUE(compareMatchResults(results_cpu, results_test));
}

TEST_F(MatcherTestSuite, FeatureUpdate) {
  std::cout << "开始测试特征更新功能" << std::endl;

  // 先执行FP32特征加载
  generateRandomFeatures(gallery_features, gallery_size, feature_dim, 42,
                         TDLDataType::FP32);
  generateRandomFeatures(query_features, query_size, feature_dim, 43,
                         TDLDataType::FP32);

  std::vector<std::shared_ptr<ModelFeatureInfo>> gallery_infos =
      createModelFeatureInfos(gallery_features, gallery_size, feature_dim,
                              TDLDataType::FP32);

  EXPECT_EQ(cpu_matcher_->loadGallery(gallery_infos), 0);
  EXPECT_EQ(test_matcher_->loadGallery(gallery_infos), 0);

  // 生成新特征并更新
  generateRandomFeatures(new_feature, 1, feature_dim, 44, TDLDataType::FP32);

  EXPECT_EQ(cpu_matcher_->updateGalleryCol(new_feature, 0), 0);
  EXPECT_EQ(test_matcher_->updateGalleryCol(new_feature, 0), 0);

  // 执行查询，比较更新后的结果
  std::vector<std::shared_ptr<ModelFeatureInfo>> query_infos =
      createModelFeatureInfos(query_features, query_size, feature_dim,
                              TDLDataType::FP32);

  MatchResult results_cpu, results_test;
  EXPECT_EQ(cpu_matcher_->queryWithTopK(query_infos, topk, results_cpu), 0);
  EXPECT_EQ(test_matcher_->queryWithTopK(query_infos, topk, results_test), 0);

  EXPECT_TRUE(compareMatchResults(results_cpu, results_test));
}

}  // namespace unitest
}  // namespace cvitdl
