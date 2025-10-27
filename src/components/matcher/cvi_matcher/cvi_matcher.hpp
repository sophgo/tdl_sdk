#pragma once
#include "matcher/base_matcher.hpp"

#include "common/common_types.hpp"
#include "utils/cvikernel.h"

class CviMatcher : public BaseMatcher {
 public:
  CviMatcher(int device_id = 0);
  ~CviMatcher() override;

  int32_t loadGallery(const std::vector<std::shared_ptr<ModelFeatureInfo>>
                          &gallery_features) override;
  int32_t queryWithTopK(
      const std::vector<std::shared_ptr<ModelFeatureInfo>> &query_features,
      int32_t topk, MatchResult &results) override;
  int32_t updateGalleryCol(void *p_data, int col) override;

 private:
  // 初始化
  void init(int device_id) override;
  // CviMatcher特有的方法
  int createHandle(CviRtHandle *rt_handle, KernelContext **cvk_ctx);
  int destroyHandle(CviRtHandle rt_handle, KernelContext *cvk_ctx);
  int cosSimilarityRegister(const FeatureArray &feature_array);
  int cosSimilarityRun(const void *feature, const uint32_t k, uint32_t *index,
                       float *scores, uint32_t *size);

  // 基类变量
  const std::vector<std::shared_ptr<ModelFeatureInfo>> *gallery_features_;
  const std::vector<std::shared_ptr<ModelFeatureInfo>> *query_features_;
  uint32_t gallery_features_num_ = 0;
  uint32_t query_features_num_ = 0;
  uint32_t feature_dim_ = 0;
  bool is_loaded_ = false;

  // 独有变量
  CviRtHandle rt_handle_;
  KernelContext *kernel_context_ = NULL;
  TPUFeatureArrayInfo tpu_feature_info_;
  CPUFeatureArrayInfo cpu_feature_info_;
  bool use_cpu_ = true;

  // 特征数据类型
  TDLDataType feature_data_type_;
};