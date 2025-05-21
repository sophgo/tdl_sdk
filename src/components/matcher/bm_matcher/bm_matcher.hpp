#ifndef BMM_MATCHER_HPP
#define BMM_MATCHER_HPP

#include <pthread.h>
#include <Eigen/Eigen>
#include <map>
#include <memory>
#include <vector>

#include "common/model_output_types.hpp"
#include "matcher/base_matcher.hpp"

#define MAX_QUERY_FEATURES_NUM 50

class BmMatcher : public BaseMatcher {
 public:
  BmMatcher(int device_id = 0);
  ~BmMatcher();

  int32_t loadGallery(const std::vector<std::shared_ptr<ModelFeatureInfo>>
                          &gallery_features) override;
  int32_t queryWithTopK(
      const std::vector<std::shared_ptr<ModelFeatureInfo>> &query_features,
      int32_t topk, MatchResult &results) override;
  int32_t updateGalleryCol(void *p_data, int col) override;

 private:
  // 初始化
  void init(int device_id) override;
  // BmMatcher特有的方法
  void requestBuffer();
  void loadBuffer(float *p_gallery);
  void queryFeatureWithTopk(float *p_features, int query_features_num, int topk,
                            std::vector<std::vector<int>> &indices,
                            std::vector<std::vector<float>> &scores);
  void dot(float *p_features, int query_features_num, float *p_result);
  void dotImpl(float *p_features, int query_features_num);
  void freeDevMem();
  // 特征归一化函数
  void normalizeFeature(float *feature, int feature_dim);

  // 将不同类型的数据转换为float类型
  void convertToFloat(const void *src, float *dst, int dim,
                      TDLDataType data_type);

  // 基类变量
  const std::vector<std::shared_ptr<ModelFeatureInfo>> *gallery_features_;
  const std::vector<std::shared_ptr<ModelFeatureInfo>> *query_features_;
  uint32_t gallery_features_num_ = 0;
  uint32_t query_features_num_ = 0;
  uint32_t feature_dim_ = 0;
  bool is_loaded_ = false;
  // 特征数据类型
  TDLDataType feature_data_type_ = TDLDataType::FP32;
  // 独有变量
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      gallery_features_eigen_;

  float *p_result_buffer_;
  pthread_mutex_t lock_ = PTHREAD_MUTEX_INITIALIZER;

#ifdef USE_BM1684
  void *handle_inst_;
  void *devmem_a_, *devmem_b_, *devmem_r_;
  void *tmp_line_dev_mem_ = nullptr;
  void *devmem_b_img_;
  void *tmp_line_img_ = nullptr;
#endif
};

#endif  // BMM_MATCHER_HPP
