#ifndef CPU_MATCHER_HPP
#define CPU_MATCHER_HPP

#include <Eigen/Dense>
#include <memory>
#include <vector>
#include "common/common_types.hpp"
#include "matcher/base_matcher.hpp"

class CpuMatcher : public BaseMatcher {
 public:
  CpuMatcher();
  ~CpuMatcher() override;

  // 加载特征库
  int32_t loadGallery(const std::vector<std::shared_ptr<ModelFeatureInfo>>&
                          gallery_features) override;

  // 查询特征
  int32_t queryWithTopK(
      const std::vector<std::shared_ptr<ModelFeatureInfo>>& query_features,
      int32_t topk, MatchResult& results) override;

  // 更新特征库
  int32_t updateGalleryCol(void* p_data, int col) override;

 protected:
  void init(int device_id = 0) override;

 private:
  // 计算两个特征向量之间的余弦相似度（所有数据类型都转换为float计算）
  float cosineSimilarity(const float* feat1, const float* feat2, int dim);

  // 将不同类型的数据转换为float类型
  void convertToFloat(const void* src, float* dst, int dim,
                      TDLDataType data_type);

  // 将特征数据转换为float并归一化，然后复制到Eigen矩阵的指定行
  void normalizeAndCopyToEigen(
      const void* src, TDLDataType src_type,
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
          dst_matrix,
      int row_idx);

  TDLDataType feature_data_type_;

  // 使用Eigen矩阵存储特征库数据
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      gallery_features_eigen_;
};

#endif  // CPU_MATCHER_HPP
