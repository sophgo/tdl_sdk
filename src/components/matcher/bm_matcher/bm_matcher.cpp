#include "bm_matcher/bm_matcher.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <queue>
#include "bm_matcher/common.hpp"
#include "bmcv_api.h"

// 将不同类型的数据转换为float类型
void BmMatcher::convertToFloat(const void* src, float* dst, int dim,
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

void BmMatcher::normalizeFeature(float* feature, int feature_dim) {
  float norm = 0.0f;
  // 计算L2范数
  for (int i = 0; i < feature_dim; i++) {
    norm += feature[i] * feature[i];
  }
  norm = std::sqrt(norm);

  // 防止除零错误
  if (norm > 1e-10) {
    for (int i = 0; i < feature_dim; i++) {
      feature[i] /= norm;
    }
  }
}

void BmMatcher::init(int device_id) {
#ifdef USE_BM1684
  handle_inst_ = BMContext::cnn_bm168x_handle(device_id);
  devmem_a_ = new bm_device_mem_t();
  devmem_b_ = new bm_device_mem_t();
  devmem_r_ = new bm_device_mem_t();
  devmem_b_img_ = new bm_image();
  tmp_line_img_ = new bm_image();
  tmp_line_dev_mem_ = new bm_device_mem_t();
#endif
}

BmMatcher::BmMatcher(int device_id) : is_loaded_(false) {
  init(device_id);
  p_result_buffer_ = 0;
}

BmMatcher::~BmMatcher() {
  freeDevMem();
#ifdef USE_BM1684
  delete (bm_device_mem_t*)devmem_a_;
  delete (bm_device_mem_t*)devmem_b_;
  delete (bm_device_mem_t*)devmem_r_;
  delete (bm_device_mem_t*)tmp_line_dev_mem_;
  delete (bm_image*)devmem_b_img_;
  delete (bm_image*)tmp_line_img_;
#endif
}

void BmMatcher::freeDevMem() {
  if (is_loaded_) {
#ifdef USE_BM1684
    bm_free_device((bm_handle_t)handle_inst_, *(bm_device_mem_t*)devmem_a_);
    bm_free_device((bm_handle_t)handle_inst_, *(bm_device_mem_t*)devmem_b_);
    bm_free_device((bm_handle_t)handle_inst_, *(bm_device_mem_t*)devmem_r_);
    bm_free_device((bm_handle_t)handle_inst_,
                   *(bm_device_mem_t*)tmp_line_dev_mem_);
#endif
    // bm_image_destroy(*(bm_image*)devmem_b_img_);
    // bm_image_destroy(*(bm_image*)tmp_line_img_);
  }

  if (p_result_buffer_ != 0) delete[] p_result_buffer_;
  p_result_buffer_ = 0;
  is_loaded_ = false;
}

int32_t BmMatcher::loadGallery(
    const std::vector<std::shared_ptr<ModelFeatureInfo>>& gallery_features) {
  if (gallery_features.empty()) {
    std::cout << "特征库为空!" << std::endl;
    is_loaded_ = false;
    return -1;
  }

  // 保存原始特征对象引用
  gallery_features_ = &gallery_features;
  gallery_features_num_ = gallery_features.size();

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

  float* gallery_data = new float[gallery_features_num_ * feature_dim_];

  // 将特征数据复制到内部存储并转换为float
  for (uint32_t i = 0; i < gallery_features_num_; ++i) {
    auto& feature = (*gallery_features_)[i];
    if (feature->embedding_num != feature_dim_) {
      std::cout << "特征维度不一致!" << std::endl;
      delete[] gallery_data;
      return -1;
    }

    if (feature->embedding_type != feature_data_type_) {
      std::cout << "特征数据类型不一致!" << std::endl;
      delete[] gallery_data;
      return -1;
    }

    // 将数据转换为float
    convertToFloat(feature->embedding, gallery_data + i * feature_dim_,
                   feature_dim_, feature_data_type_);
  }

  loadBuffer(gallery_data);
  delete[] gallery_data;

  return 0;
}

int32_t BmMatcher::queryWithTopK(
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

  // 检查查询特征维度和数据类型是否与特征库一致
  for (uint32_t i = 0; i < query_features_num_; ++i) {
    auto& feature = (*query_features_)[i];
    if (feature->embedding_num != feature_dim_) {
      std::cout << "查询特征维度与库不一致!" << std::endl;
      return -1;
    }

    if (feature->embedding_type != feature_data_type_) {
      std::cout << "查询特征数据类型与库不一致!" << std::endl;
      return -1;
    }
  }

  // 将查询特征转换为float*格式
  float* p_query_feature = new float[query_features_num_ * feature_dim_];

  // 转换查询特征数据为float并归一化
  for (uint32_t i = 0; i < query_features_num_; ++i) {
    // 将数据转换为float
    convertToFloat((*query_features_)[i]->embedding,
                   p_query_feature + i * feature_dim_, feature_dim_,
                   feature_data_type_);
  }

  // 确保topk不超过特征库大小
  topk = std::min(topk, (int32_t)gallery_features_num_);

  std::vector<std::vector<int>> indices;
  std::vector<std::vector<float>> scores;

  // 使用优化的矩阵乘法计算相似度
  queryFeatureWithTopk(p_query_feature, query_features_num_, topk, indices,
                       scores);

  results.indices = indices;
  results.scores = scores;

  delete[] p_query_feature;
  return 0;
}

int32_t BmMatcher::updateGalleryCol(void* p_data, int col) {
  if (!is_loaded_) {
    std::cout << "特征库尚未加载!" << std::endl;
    return -1;
  }

  if (col < 0 || col >= (int)gallery_features_num_) {
    std::cout << "更新列索引超出范围!" << std::endl;
    return -1;
  }

  // 将传入的特征数据转换为float
  float* feature_data = new float[feature_dim_];
  convertToFloat(p_data, feature_data, feature_dim_, feature_data_type_);

  pthread_mutex_lock(&lock_);

#ifdef USE_BM1684
  bm_handle_t handle = (bm_handle_t)handle_inst_;
  bm_memcpy_s2d_partial(handle, *(bm_device_mem_t*)tmp_line_dev_mem_,
                        feature_data, feature_dim_ * sizeof(float));
  // 拷贝到设备内存的对应位置
  size_t offset = col * feature_dim_ * sizeof(float);
  bm_memcpy_d2d_byte(handle, *(bm_device_mem_t*)devmem_b_, offset,
                     *(bm_device_mem_t*)tmp_line_dev_mem_, 0,
                     feature_dim_ * sizeof(float));
#else
  for (uint32_t j = 0; j < feature_dim_; ++j) {
    gallery_features_eigen_(col, j) = feature_data[j];
  }
#endif

  delete[] feature_data;
  pthread_mutex_unlock(&lock_);
  return 0;
}

void BmMatcher::queryFeatureWithTopk(float* p_features, int query_features_num,
                                     int topk,
                                     std::vector<std::vector<int>>& indices,
                                     std::vector<std::vector<float>>& scores) {
  pthread_mutex_lock(&lock_);

  // 计算相似度
  dotImpl(p_features, query_features_num);

  // 初始化结果
  indices.resize(query_features_num);
  scores.resize(query_features_num);

  // 提取每个查询的TopK结果
  for (int i = 0; i < query_features_num; i++) {
    indices[i].resize(topk);
    scores[i].resize(topk);

    // 当前查询的相似度结果
    float* curr_result = p_result_buffer_ + i * gallery_features_num_;

    // 使用优先队列找出TopK
    std::priority_queue<std::pair<float, int>,
                        std::vector<std::pair<float, int>>,
                        std::greater<std::pair<float, int>>>
        q;

    for (uint32_t j = 0; j < gallery_features_num_; j++) {
      if (q.size() < (size_t)topk) {
        q.push(std::make_pair(curr_result[j], j));
      } else if (q.top().first < curr_result[j]) {
        q.pop();
        q.push(std::make_pair(curr_result[j], j));
      }
    }

    // 从队列中提取结果(从大到小排序)
    for (int k = topk - 1; k >= 0; k--) {
      auto top = q.top();
      indices[i][k] = top.second;
      scores[i][k] = top.first;
      q.pop();
    }
  }

  pthread_mutex_unlock(&lock_);
}

void BmMatcher::dot(float* p_features, int query_features_num,
                    float* p_result) {
  if (!is_loaded_) return;
  pthread_mutex_lock(&lock_);
  dotImpl(p_features, query_features_num);
  memcpy(p_result, p_result_buffer_,
         query_features_num * gallery_features_num_ * sizeof(float));
  pthread_mutex_unlock(&lock_);
}

void BmMatcher::requestBuffer() {
  pthread_mutex_lock(&lock_);

  p_result_buffer_ = new float[MAX_QUERY_FEATURES_NUM * gallery_features_num_];

#ifdef USE_BM1684
  bm_handle_t handle = (bm_handle_t)handle_inst_;
  bm_malloc_device_byte(handle, (bm_device_mem_t*)devmem_a_,
                        MAX_QUERY_FEATURES_NUM * feature_dim_ * sizeof(float));
  bm_malloc_device_byte(handle, (bm_device_mem_t*)devmem_b_,
                        gallery_features_num_ * feature_dim_ * sizeof(float));
  bm_malloc_device_byte(
      handle, (bm_device_mem_t*)devmem_r_,
      MAX_QUERY_FEATURES_NUM * gallery_features_num_ * sizeof(float));
  bm_malloc_device_byte(handle, (bm_device_mem_t*)tmp_line_dev_mem_,
                        feature_dim_ * sizeof(float));

  bm_image_create(handle, feature_dim_, gallery_features_num_, FORMAT_GRAY,
                  DATA_TYPE_EXT_FLOAT32, (bm_image*)devmem_b_img_);
  bm_image_attach(*(bm_image*)devmem_b_img_, (bm_device_mem_t*)devmem_b_);

  bm_image_create(handle, feature_dim_, 1, FORMAT_GRAY, DATA_TYPE_EXT_FLOAT32,
                  (bm_image*)tmp_line_img_);
  bm_image_attach(*(bm_image*)tmp_line_img_,
                  (bm_device_mem_t*)tmp_line_dev_mem_);
#else
  gallery_features_eigen_ =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(
          gallery_features_num_, feature_dim_);
#endif

  pthread_mutex_unlock(&lock_);
}

void BmMatcher::loadBuffer(float* p_gallery) {
  requestBuffer();
  pthread_mutex_lock(&lock_);

  float* ptr_b = p_gallery;

#ifdef USE_BM1684
  bm_handle_t handle = (bm_handle_t)handle_inst_;

  // 直接将特征数据拷贝到设备内存
  bm_memcpy_s2d_partial(handle, *(bm_device_mem_t*)devmem_b_, ptr_b,
                        gallery_features_num_ * feature_dim_ * sizeof(float));

#else
  for (uint32_t i = 0; i < gallery_features_num_; ++i) {
    for (uint32_t j = 0; j < feature_dim_; ++j) {
      gallery_features_eigen_(i, j) = p_gallery[i * feature_dim_ + j];
    }
  }
#endif

  pthread_mutex_unlock(&lock_);
  is_loaded_ = true;
}

void BmMatcher::dotImpl(float* p_features, int query_features_num) {
  assert(query_features_num <= MAX_QUERY_FEATURES_NUM);

  // 先对查询特征进行归一化
  for (int i = 0; i < query_features_num; ++i) {
    normalizeFeature(p_features + i * feature_dim_, feature_dim_);
  }

#ifdef USE_BM1684
  bm_handle_t handle = (bm_handle_t)handle_inst_;

  // BM1684设备上的查询特征归一化和上传
  bm_memcpy_s2d_partial(handle, *(bm_device_mem_t*)devmem_a_, p_features,
                        query_features_num * feature_dim_ * sizeof(float));

  // BM1684上对库特征进行归一化（需要先下载到主机内存）
  float* gallery_data = new float[gallery_features_num_ * feature_dim_];
  bm_memcpy_d2s_partial(handle, gallery_data, *(bm_device_mem_t*)devmem_b_,
                        gallery_features_num_ * feature_dim_ * sizeof(float));

  // 对库特征进行归一化 - 每个特征向量连续存储
  for (uint32_t i = 0; i < gallery_features_num_; ++i) {
    normalizeFeature(gallery_data + i * feature_dim_, feature_dim_);
  }

  // 将归一化后的库特征上传回设备内存
  bm_memcpy_s2d_partial(handle, *(bm_device_mem_t*)devmem_b_, gallery_data,
                        gallery_features_num_ * feature_dim_ * sizeof(float));

  delete[] gallery_data;

  // 使用GEMM计算相似度矩阵（归一化后的点积就是余弦相似度）
  bm_status_t ret = bmcv_gemm_ext(
      handle, query_features_num, gallery_features_num_, feature_dim_, 1.0f,
      (bm_device_mem_t*)devmem_a_, 0, 0, (bm_device_mem_t*)devmem_b_, 0, 0,
      0.0f, (bm_device_mem_t*)devmem_r_, 0, 0);
  if (ret != BM_SUCCESS) {
    std::cout << "bmcv_gemm_ext failed" << std::endl;
  }

  bm_memcpy_d2s(handle, (void*)p_result_buffer_, *(bm_device_mem_t*)devmem_r_);
#else

  Eigen::Map<
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      query_map(p_features, query_features_num, feature_dim_);

  // 对库特征进行归一化
  for (uint32_t i = 0; i < gallery_features_num_; ++i) {
    Eigen::RowVectorXf row = gallery_features_eigen_.row(i);
    float norm = row.norm();
    if (norm > 1e-10) {
      gallery_features_eigen_.row(i) = row / norm;
    }
  }

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result =
      query_map * gallery_features_eigen_.transpose();

  memcpy(p_result_buffer_, result.data(),
         query_features_num * gallery_features_num_ * sizeof(float));
#endif
}
