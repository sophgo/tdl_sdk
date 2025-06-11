#include "matcher/base_matcher.hpp"
#include <iostream>
#include "cpu_matcher/cpu_matcher.hpp"
#if defined(__BM168X__) || defined(__CV184X__)
#include "bm_matcher/bm_matcher.hpp"
#endif
#if defined(__CV181X__)
#include "cvi_matcher/cvi_matcher.hpp"
#endif

BaseMatcher::BaseMatcher() {}

BaseMatcher::~BaseMatcher() {}

void BaseMatcher::init(int device_id) {}

int32_t BaseMatcher::loadGallery(
    const std::vector<std::shared_ptr<ModelFeatureInfo>>& gallery_features) {
  // 加载特征库
  return 0;
}

int32_t BaseMatcher::queryWithTopK(
    const std::vector<std::shared_ptr<ModelFeatureInfo>>& query_features,
    int32_t topk, MatchResult& results) {
  // 查询特征
  return 0;
}

int32_t BaseMatcher::updateGalleryCol(void* p_data, int col) {
  // 更新特征库
  return 0;
}

int32_t BaseMatcher::getGalleryFeatureNum() const {
  return gallery_features_num_;
}

int32_t BaseMatcher::getQueryFeatureNum() const { return query_features_num_; }

int32_t BaseMatcher::getFeatureDim() const { return feature_dim_; }

std::shared_ptr<BaseMatcher> BaseMatcher::getMatcher(std::string matcher_type) {
#if defined(__BM168X__) || defined(__CV184X__)
  if (matcher_type == "bm") {
    return std::make_shared<BmMatcher>();
  }
#endif
#if defined(__CV181X__)
  if (matcher_type == "cvi") {
    return std::make_shared<CviMatcher>();
  }
#endif
  if (matcher_type == "cpu") {
    return std::make_shared<CpuMatcher>();
  } else {
    throw std::invalid_argument("Only support cpu, bm, cvi matcher");
  }
}
