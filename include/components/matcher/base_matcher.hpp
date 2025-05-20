#ifndef BASE_MATCHER_HPP
#define BASE_MATCHER_HPP

#include <cstdint>
#include <memory>
#include <vector>
#include "common/model_output_types.hpp"

struct MatchResult {
  // 匹配索引，indices[i][j]表示第i个查询第j个最佳匹配的索引
  std::vector<std::vector<int>> indices;
  // 匹配分数，scores[i][j]表示第i个查询第j个最佳匹配的分数
  std::vector<std::vector<float>> scores;
};

class BaseMatcher {
 public:
  BaseMatcher();
  virtual ~BaseMatcher();
  // 加载特征库
  virtual int32_t loadGallery(
      const std::vector<std::shared_ptr<ModelFeatureInfo>>& gallery_features);
  // 查询特征
  virtual int32_t queryWithTopK(
      const std::vector<std::shared_ptr<ModelFeatureInfo>>& query_features,
      int32_t topk,
      MatchResult& results);
  // 更新特征库
  virtual int32_t updateGalleryCol(void* p_data, int col);

  virtual int32_t getGalleryFeatureNum() const;
  virtual int32_t getQueryFeatureNum() const;
  virtual int32_t getFeatureDim() const;

  // 创建匹配器实例
  static std::shared_ptr<BaseMatcher> getMatcher(std::string matcher_type);

 protected:
  virtual void init(int device_id) = 0;
  const std::vector<std::shared_ptr<ModelFeatureInfo>>* gallery_features_;
  const std::vector<std::shared_ptr<ModelFeatureInfo>>* query_features_;
  int32_t gallery_features_num_ = 0;
  int32_t query_features_num_ = 0;
  int32_t feature_dim_ = 0;
  bool is_loaded_ = false;
};

#endif  // BASE_MATCHER_HPP
