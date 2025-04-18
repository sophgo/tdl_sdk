#include "matcher/base_matcher.hpp"
#include <iostream>
#ifdef USE_BM_MATCHER
#include "bm_matcher/bm_matcher.hpp"
#endif
#ifdef USE_CVI_MATCHER
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

std::shared_ptr<BaseMatcher> BaseMatcher::getMatcher() {
#ifdef USE_BM_MATCHER
  return std::make_shared<BmMatcher>();
#endif
#ifdef USE_CVI_MATCHER
  return std::make_shared<CviMatcher>();
#endif
  std::cout << "USE_NO_MATCHER" << std::endl;
  return nullptr;
}
