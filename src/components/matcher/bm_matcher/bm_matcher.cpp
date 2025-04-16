#include "bm_matcher/bm_matcher.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <queue>
#include "bm_matcher/common.hpp"
#include "bmcv_api.h"

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
    return -1;
  }

  gallery_features_ = gallery_features;
  gallery_features_num_ = gallery_features.size();
  feature_dim_ = gallery_features[0]->embedding_num;

  float* gallery_data = new float[gallery_features_num_ * feature_dim_];

  for (uint32_t i = 0; i < gallery_features_num_; ++i) {
    float* feature_data =
        reinterpret_cast<float*>(gallery_features_[i]->embedding);
    memcpy(gallery_data + i * feature_dim_, feature_data,
           feature_dim_ * sizeof(float));
  }

  loadBuffer(gallery_data);
  delete[] gallery_data;
  gallery_features_.clear();
  return 0;
}

int32_t BmMatcher::queryWithTopK(
    const std::vector<std::shared_ptr<ModelFeatureInfo>>& query_features,
    int32_t topk, std::vector<MatchResult>& results) {
  if (!is_loaded_) {
    return -1;
  }

  query_features_ = query_features;
  query_features_num_ = query_features.size();
  if (query_features[0]->embedding_num != feature_dim_) {
    return -1;
  }

  // 将查询特征转换为float*格式
  float* p_query_feature = new float[query_features_num_ * feature_dim_];
  for (uint32_t i = 0; i < query_features_num_; ++i) {
    float* feature_data =
        reinterpret_cast<float*>(query_features_[i]->embedding);
    memcpy(p_query_feature + i * feature_dim_, feature_data,
           feature_dim_ * sizeof(float));
  }

  std::vector<std::vector<int>> indices;
  std::vector<std::vector<float>> scores;
  queryFeatureWithTopk(p_query_feature, query_features_num_, topk, indices,
                       scores);

  results.resize(query_features_num_);
  for (uint32_t i = 0; i < query_features_num_; ++i) {
    results[i].indices = indices[i];
    results[i].scores = scores[i];
  }

  delete[] p_query_feature;
  return 0;
}

int32_t BmMatcher::updateGalleryCol(void* p_data, int col_idx) {
  pthread_mutex_lock(&lock_);

  int start_idx = col_idx;
  if (start_idx >= gallery_features_num_)
    std::cout << "gallery number:" << gallery_features_num_
              << ",update index:" << col_idx << std::endl;

  float* p_sub_b = (float*)p_data;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> vec(
      feature_dim_, 1);
  memcpy(vec.data(), p_sub_b, feature_dim_ * 1 * sizeof(float));
  gallery_features_eigen_.col(start_idx) = vec;

  pthread_mutex_unlock(&lock_);
  return 0;
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
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> gallery(
      feature_dim_, gallery_features_num_);
  gallery_features_eigen_ = gallery;
#endif

  pthread_mutex_unlock(&lock_);
}

void BmMatcher::loadBuffer(float* p_gallery) {
  requestBuffer();
  pthread_mutex_lock(&lock_);

  float* ptr_b = p_gallery;

#ifdef USE_BM1684
  bm_handle_t handle = (bm_handle_t)handle_inst_;

  bm_memcpy_s2d_partial(handle, *(bm_device_mem_t*)devmem_b_, ptr_b,
                        gallery_features_num_ * feature_dim_ * sizeof(float));

#else
  float* pdata = gallery_features_eigen_.data();
  memcpy(pdata, p_gallery,
         gallery_features_num_ * feature_dim_ * sizeof(float));
#endif

  pthread_mutex_unlock(&lock_);
  is_loaded_ = true;
}

void BmMatcher::queryFeatureWithTopk(float* p_query_feature,
                                     int query_features_num, int topk,
                                     std::vector<std::vector<int>>& indices,
                                     std::vector<std::vector<float>>& scores) {
  if (!is_loaded_) return;

  pthread_mutex_lock(&lock_);

  dotImpl(p_query_feature, query_features_num);

  indices.clear();
  indices.resize(query_features_num);
  scores.clear();
  scores.resize(query_features_num);

  for (int i = 0; i < query_features_num; i++) {
    std::priority_queue<std::pair<float, int>> top_k;
    for (int j = 0; j < gallery_features_num_; j++) {
      float score = p_result_buffer_[i * gallery_features_num_ + j];
      top_k.push(std::make_pair(score, j));
    }

    // 填充topk结果
    std::vector<int> ind(topk);
    std::vector<float> score(topk);
    int idx = topk - 1;
    while (!top_k.empty() && idx >= 0) {
      auto& pr = top_k.top();
      ind[idx] = pr.second;
      score[idx] = pr.first;
      top_k.pop();
      idx--;
    }

    indices[i] = ind;
    scores[i] = score;
  }
  pthread_mutex_unlock(&lock_);
}

void BmMatcher::dot(float* p_features, int query_features_num,
                    float* p_result) {
  pthread_mutex_lock(&lock_);
  dotImpl(p_features, query_features_num);
  memcpy(p_result, p_result_buffer_,
         query_features_num * gallery_features_num_ * sizeof(float));
  pthread_mutex_unlock(&lock_);
}

void BmMatcher::dotImpl(float* p_features, int query_features_num) {
  assert(query_features_num <= MAX_QUERY_FEATURES_NUM);
#ifdef USE_BM1684
  bm_handle_t handle = (bm_handle_t)handle_inst_;
  bm_memcpy_s2d_partial(handle, *(bm_device_mem_t*)devmem_a_, p_features,
                        query_features_num * feature_dim_ * sizeof(float));

  bm_status_t ret = bmcv_gemm_ext(
      handle, query_features_num, gallery_features_num_, feature_dim_, 1.0f,
      (bm_device_mem_t*)devmem_a_, 0, 0, (bm_device_mem_t*)devmem_b_, 0, 0,
      0.0f, (bm_device_mem_t*)devmem_r_, 0, 0);
  if (ret != BM_SUCCESS) {
    std::cout << "bmcv_gemm_ext failed" << std::endl;
  }

  bm_memcpy_d2s(handle, (void*)p_result_buffer_, *(bm_device_mem_t*)devmem_r_);
#else
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> feature(
      feature_dim_, query_features_num);
  float* pdata = feature.data();
  memcpy(pdata, p_features, query_features_num * feature_dim_ * sizeof(float));
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ret =
      feature.transpose() * gallery_features_eigen_;
  memcpy(p_result_buffer_, ret.data(),
         query_features_num * gallery_features_num_ * sizeof(float));
#endif
}
