#include "matcher/base_matcher.hpp"

#include "bm_matcher/bm_matcher.hpp"

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
    int32_t topk, std::vector<MatchResult>& results) {
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

std::shared_ptr<BaseMatcher> BaseMatcher::getMatcher(
    const std::string& matcher_type) {
  if (matcher_type == "bm") {
    return std::make_shared<BmMatcher>();
  }
  return nullptr;
}
