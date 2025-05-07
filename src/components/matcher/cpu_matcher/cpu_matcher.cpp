#include "cpu_matcher.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

CpuMatcher::CpuMatcher() { init(0); }

CpuMatcher::~CpuMatcher() {}

void CpuMatcher::init(int device_id) {}

// 将不同类型的数据转换为float类型
void CpuMatcher::convertToFloat(const void* src, float* dst, int dim,
                                TDLDataType data_type) {
  switch (data_type) {
    case TDLDataType::FP32: {
      const float* float_data = static_cast<const float*>(src);
      std::copy(float_data, float_data + dim, dst);
      break;
    }
    case TDLDataType::INT8: {
      const int8_t* int8_data = static_cast<const int8_t*>(src);
      for (int i = 0; i < dim; ++i) {
        dst[i] = static_cast<float>(int8_data[i]);
      }
      break;
    }
    case TDLDataType::UINT8: {
      const uint8_t* uint8_data = static_cast<const uint8_t*>(src);
      for (int i = 0; i < dim; ++i) {
        dst[i] = static_cast<float>(uint8_data[i]);
      }
      break;
    }
    default:
      std::cout << "不支持的特征数据类型!" << std::endl;
      break;
  }
}

void CpuMatcher::normalizeAndCopyToEigen(
    const void* src, TDLDataType src_type,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        dst_matrix,
    int row_idx) {
  std::vector<float> temp_buffer(feature_dim_);
  convertToFloat(src, temp_buffer.data(), feature_dim_, src_type);

  // 对特征进行归一化
  float norm = 0.0f;
  for (int j = 0; j < feature_dim_; j++) {
    norm += temp_buffer[j] * temp_buffer[j];
  }
  norm = std::sqrt(norm);
  if (norm > 1e-10) {
    for (int j = 0; j < feature_dim_; j++) {
      temp_buffer[j] /= norm;
    }
  }

  // 将归一化后的特征复制到Eigen矩阵
  for (int j = 0; j < feature_dim_; j++) {
    dst_matrix(row_idx, j) = temp_buffer[j];
  }
}

int32_t CpuMatcher::loadGallery(
    const std::vector<std::shared_ptr<ModelFeatureInfo>>& gallery_features) {
  // 保存原始特征对象引用
  gallery_features_ = &gallery_features;
  gallery_features_num_ = gallery_features.size();

  if (gallery_features_num_ == 0) {
    std::cout << "特征库为空!" << std::endl;
    is_loaded_ = false;
    return -1;
  }

  // 获取特征数据类型
  feature_data_type_ = (*gallery_features_)[0]->embedding_type;

  // 提取第一个特征的维度
  if (feature_data_type_ == TDLDataType::FP32 ||
      feature_data_type_ == TDLDataType::INT8 ||
      feature_data_type_ == TDLDataType::UINT8) {
    feature_dim_ = (*gallery_features_)[0]->embedding_num;
  } else {
    std::cout << "不支持的特征数据类型: "
              << static_cast<int>(feature_data_type_) << std::endl;
    return -1;
  }

  // 使用Eigen矩阵存储特征数据
  gallery_features_eigen_ =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(
          gallery_features_num_, feature_dim_);

  // 将特征数据复制到Eigen矩阵并转换为float
  for (int i = 0; i < gallery_features_num_; ++i) {
    auto& feature = (*gallery_features_)[i];
    if (feature->embedding_num != feature_dim_) {
      std::cout << "特征维度不一致!" << std::endl;
      return -1;
    }

    if (feature->embedding_type != feature_data_type_) {
      std::cout << "特征数据类型不一致!" << std::endl;
      return -1;
    }

    normalizeAndCopyToEigen(feature->embedding, feature_data_type_,
                            gallery_features_eigen_, i);
  }

  is_loaded_ = true;
  return 0;
}

int32_t CpuMatcher::queryWithTopK(
    const std::vector<std::shared_ptr<ModelFeatureInfo>>& query_features,
    int32_t topk, MatchResult& results) {
  if (!is_loaded_) {
    std::cout << "特征库尚未加载!" << std::endl;
    return -1;
  }

  query_features_ = &query_features;
  query_features_num_ = query_features.size();

  if (query_features_num_ == 0) {
    std::cout << "查询特征为空!" << std::endl;
    return -1;
  }

  // 使用Eigen矩阵存储查询特征
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      query_features_eigen(query_features_num_, feature_dim_);

  // 转换查询特征数据为float并归一化
  for (int i = 0; i < query_features_num_; ++i) {
    auto& feature = (*query_features_)[i];
    if (feature->embedding_num != feature_dim_) {
      std::cout << "查询特征维度与库不一致!" << std::endl;
      return -1;
    }

    if (feature->embedding_type != feature_data_type_) {
      std::cout << "查询特征数据类型与库不一致!" << std::endl;
      return -1;
    }

    normalizeAndCopyToEigen(feature->embedding, feature_data_type_,
                            query_features_eigen, i);
  }

  // 确保topk不超过特征库大小
  topk = std::min(topk, gallery_features_num_);

  // 使用矩阵乘法计算相似度
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      similarities = query_features_eigen * gallery_features_eigen_.transpose();

  // 初始化结果结构
  results.indices.resize(query_features_num_);
  results.scores.resize(query_features_num_);

  // 对每个查询特征找出topk结果
  for (int i = 0; i < query_features_num_; ++i) {
    std::vector<std::pair<float, int>> sim_pairs;
    sim_pairs.reserve(gallery_features_num_);

    for (int j = 0; j < gallery_features_num_; ++j) {
      sim_pairs.push_back(std::make_pair(similarities(i, j), j));
    }

    // 按相似度降序排序
    std::sort(sim_pairs.begin(), sim_pairs.end(),
              [](const std::pair<float, int>& a,
                 const std::pair<float, int>& b) { return a.first > b.first; });

    // 保存topk结果
    results.indices[i].resize(topk);
    results.scores[i].resize(topk);

    for (int k = 0; k < topk; ++k) {
      results.indices[i][k] = sim_pairs[k].second;
      results.scores[i][k] = sim_pairs[k].first;
    }
  }

  return 0;
}

int32_t CpuMatcher::updateGalleryCol(void* p_data, int col) {
  if (!is_loaded_) {
    std::cout << "特征库尚未加载!" << std::endl;
    return -1;
  }

  if (col < 0 || col >= gallery_features_num_) {
    std::cout << "更新列索引超出范围!" << std::endl;
    return -1;
  }

  normalizeAndCopyToEigen(p_data, feature_data_type_, gallery_features_eigen_,
                          col);

  return 0;
}

// 余弦相似度计算（所有数据类型都已转换为float）
float CpuMatcher::cosineSimilarity(const float* feat1, const float* feat2,
                                   int dim) {
  float dot = 0.0f;
  float norm1 = 0.0f;
  float norm2 = 0.0f;

  for (int i = 0; i < dim; ++i) {
    dot += feat1[i] * feat2[i];
    norm1 += feat1[i] * feat1[i];
    norm2 += feat2[i] * feat2[i];
  }

  // 避免除零错误
  if (norm1 < 1e-6 || norm2 < 1e-6) {
    return 0.0f;
  }

  return dot / (std::sqrt(norm1) * std::sqrt(norm2));
}
