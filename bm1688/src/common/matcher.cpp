#include "common/matcher.hpp"

#include <bmcv_api.h>
#include <bmcv_api_ext.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <queue>

#include "common/common.hpp"

void bmcv_util_copy_to(bm_handle_t handle, bm_image *p_src_img, int offsetx,
                       int offsety, bm_image *p_dst_img,
                       bool pad_zero /*=false)*/) {
  assert(offsetx >= 0 && offsety >= 0);
  assert(p_src_img->width + offsetx <= p_dst_img->width &&
         p_src_img->height + offsety <= p_dst_img->height);

  bmcv_copy_to_atrr_t copy_to_attr;
  copy_to_attr.start_x = offsetx;
  copy_to_attr.start_y = offsety;
  copy_to_attr.padding_r = 0;
  copy_to_attr.padding_g = 0;
  copy_to_attr.padding_b = 0;
  copy_to_attr.if_padding = pad_zero;
  if (BM_SUCCESS !=
      bmcv_image_copy_to(handle, copy_to_attr, *p_src_img, *p_dst_img)) {
    LOG(FATAL) << "bmcv copy to failed";
  }
}

void Matcher ::init(int device_id) {
  handle_inst_ = BMContext::cnn_bm168x_handle(device_id);
  devmem_a_ = new bm_device_mem_t();
  devmem_b_ = new bm_device_mem_t();
  devmem_r_ = new bm_device_mem_t();
  devmem_b_img_ = new bm_image();
  tmp_line_img_ = new bm_image();
  tmp_line_dev_mem_ = new bm_device_mem_t();
}

Matcher::Matcher(int device_id /*= 0*/) : is_loaded(false), M(0), K(0), N(1) {
  init(device_id);
  m_pResultBuffer = 0;
}

Matcher::~Matcher() {
  free_dev_mem();
  delete (bm_device_mem_t *)devmem_a_;
  delete (bm_device_mem_t *)devmem_b_;
  delete (bm_device_mem_t *)devmem_r_;
  delete (bm_device_mem_t *)tmp_line_dev_mem_;
  delete (bm_image *)devmem_b_img_;
  delete (bm_image *)tmp_line_img_;
}

void Matcher::free_dev_mem() {
  if (is_loaded) {
    bm_free_device((bm_handle_t)handle_inst_, *(bm_device_mem_t *)devmem_a_);
    bm_free_device((bm_handle_t)handle_inst_, *(bm_device_mem_t *)devmem_b_);
    bm_free_device((bm_handle_t)handle_inst_, *(bm_device_mem_t *)devmem_r_);
    bm_free_device((bm_handle_t)handle_inst_,
                   *(bm_device_mem_t *)tmp_line_dev_mem_);

    bm_image_destroy(*(bm_image *)devmem_b_img_);
    bm_image_destroy(*(bm_image *)tmp_line_img_);
  }

  if (m_pResultBuffer != 0) delete[] m_pResultBuffer;
  m_pResultBuffer = 0;
  is_loaded = false;
}

void Matcher::update_gallery_col(void *p_data, int col_idx) {
  pthread_mutex_lock(&lock_);

  int start_idx = col_idx;
  if (start_idx >= N)
    LOG(FATAL) << "gallery number:" << N << ",update index:" << col_idx;

  bm_handle_t handle = (bm_handle_t)handle_inst_;
  bm_memcpy_s2d_partial(handle, *(bm_device_mem_t *)tmp_line_dev_mem_, p_data,
                        K * sizeof(float));
  bmcv_util_copy_to(handle, (bm_image *)tmp_line_img_, col_idx, 0,
                    (bm_image *)devmem_b_img_, false);

  pthread_mutex_unlock(&lock_);
}
void Matcher::request_buffer(int num_gallery, int feat_dim) {
  if (num_gallery == N && feat_dim == K) return;

  pthread_mutex_lock(&lock_);
  K = feat_dim;
  N = num_gallery;
  M = 1;

  m_pResultBuffer = new float[max_M * N];

  printf("allocate buffer,(M, K, N) = (%d, %d, %d)\n", M, K, N);

  bm_handle_t handle = (bm_handle_t)handle_inst_;
  bm_malloc_device_byte(handle, (bm_device_mem_t *)devmem_a_,
                        max_M * K * sizeof(float));
  bm_malloc_device_byte(handle, (bm_device_mem_t *)devmem_b_,
                        N * K * sizeof(float));
  bm_malloc_device_byte(handle, (bm_device_mem_t *)devmem_r_,
                        max_M * N * sizeof(float));
  bm_malloc_device_byte(handle, (bm_device_mem_t *)tmp_line_dev_mem_,
                        K * sizeof(float));

  bm_image_create(handle, K, N, FORMAT_GRAY, DATA_TYPE_EXT_FLOAT32,
                  (bm_image *)devmem_b_img_);
  bm_image_attach(*(bm_image *)devmem_b_img_, (bm_device_mem_t *)devmem_b_);

  bm_image_create(handle, K, 1, FORMAT_GRAY, DATA_TYPE_EXT_FLOAT32,
                  (bm_image *)tmp_line_img_);
  bm_image_attach(*(bm_image *)tmp_line_img_,
                  (bm_device_mem_t *)tmp_line_dev_mem_);

  pthread_mutex_unlock(&lock_);
}

void Matcher::load_buffer(float *p_gallery, int num_gallery, int feat_dim,
                          int num_allocated /*= 0*/) {
  if (num_allocated != 0) {
    if (num_allocated < num_gallery) {
      std::cout << "allocated size:" << num_allocated
                << " should be larger than realsize:" << num_gallery
                << std::endl;
      assert(0);
    }
    request_buffer(num_allocated, feat_dim);
  } else if (!is_loaded) {
    request_buffer(num_gallery, feat_dim);
  }
  if (num_gallery == 0) return;
  pthread_mutex_lock(&lock_);

  float *ptr_b = p_gallery;
  if (num_gallery > N) {
    std::cout << "feature number larger than allocated buffer size,allocated:"
              << N << ",got:" << num_gallery << std::endl;
    assert(0);
  }
  bm_handle_t handle = (bm_handle_t)handle_inst_;

  bm_memcpy_s2d_partial(handle, *(bm_device_mem_t *)devmem_b_, ptr_b,
                        num_gallery * K * sizeof(float));

  pthread_mutex_unlock(&lock_);
  is_loaded = true;
}

void Matcher::load_raw(float *p_gallery, int num_gallery, int feature_size) {
  pthread_mutex_lock(&lock_);
  free_dev_mem();

  K = feature_size;
  N = num_gallery;
  M = 1;

  m_pResultBuffer = new float[max_M * N];

  printf("(M, K, N) = (%d, %d, %d)\n", M, K, N);

  float *ptr_b = p_gallery;

  bm_handle_t handle = (bm_handle_t)handle_inst_;
  bm_malloc_device_byte(handle, (bm_device_mem_t *)devmem_a_,
                        max_M * K * sizeof(float));
  bm_malloc_device_byte(handle, (bm_device_mem_t *)devmem_b_,
                        N * K * sizeof(float));
  bm_malloc_device_byte(handle, (bm_device_mem_t *)devmem_r_,
                        max_M * N * sizeof(float));
  bm_malloc_device_byte(handle, (bm_device_mem_t *)tmp_line_dev_mem_,
                        K * sizeof(float));

  bm_memcpy_s2d_partial(handle, *(bm_device_mem_t *)devmem_b_, ptr_b,
                        N * K * sizeof(float));

  bm_image_create(handle, K, N, FORMAT_GRAY, DATA_TYPE_EXT_FLOAT32,
                  (bm_image *)devmem_b_img_);
  bm_image_attach(*(bm_image *)devmem_b_img_, (bm_device_mem_t *)devmem_b_);

  bm_image_create(handle, K, 1, FORMAT_GRAY, DATA_TYPE_EXT_FLOAT32,
                  (bm_image *)tmp_line_img_);
  bm_image_attach(*(bm_image *)tmp_line_img_,
                  (bm_device_mem_t *)tmp_line_dev_mem_);

  is_loaded = true;
  pthread_mutex_unlock(&lock_);
}

std::vector<std::pair<int, float>> Matcher::query_raw(float *p_features,
                                                      int num_features) {
  std::vector<std::pair<int, float>> ret;
  if (!is_loaded) return ret;
  pthread_mutex_lock(&lock_);
  dot_impl(p_features, num_features);
  float *ptr_res_start = m_pResultBuffer;

  for (int i = 0; i < num_features; i++) {
    // std::vector<float> res;
    int max_idx = std::distance(
        ptr_res_start, std::max_element(ptr_res_start, ptr_res_start + N));
    float score = ptr_res_start[max_idx];
    ret.push_back(std::make_pair(max_idx, score));
    ptr_res_start = ptr_res_start + N;
  }
  pthread_mutex_unlock(&lock_);
  return ret;
}
void Matcher::query_raw_with_topk(float *p_features, int num_features, int topk,
                                  std::vector<std::vector<int>> &indices,
                                  std::vector<std::vector<float>> &scores) {
  if (!is_loaded) {
    LOG(WARNING) << "matcher not loaded";
    return;
  }
  if (topk == 1) {
    std::vector<std::pair<int, float>> res =
        query_raw(p_features, num_features);
    for (auto kv : res) {
      std::vector<int> ind = {kv.first};
      std::vector<float> score = {kv.second};
      indices.push_back(ind);
      scores.push_back(score);
      LOG(INFO) << "ind:" << kv.first << ",socre:" << kv.second;
    }
    LOG(INFO) << "indsize:" << indices.size() << ",scoresize:" << scores.size();
    return;
  }
  pthread_mutex_lock(&lock_);
  LOG(INFO) << "dot query with top:" << topk << ",numfeats:" << num_features
            << ",gallerynum:" << N;
  dot_impl(p_features, num_features);
  float *ptr_res_start = m_pResultBuffer;

  for (int i = 0; i < num_features; i++) {
    std::stringstream ss;
    ss << "query " << i;
    std::priority_queue<std::pair<float, int>,
                        std::vector<std::pair<float, int>>,
                        std::greater<std::pair<float, int>>>
        q;
    for (int j = 0; j < N; j++) {
      if (q.size() < topk) {
        q.push(std::pair<float, int>(ptr_res_start[j], j));
      } else if (q.top().first < ptr_res_start[j]) {
        q.pop();
        q.push(std::pair<float, int>(ptr_res_start[j], j));
      }
    }
    std::vector<int> ind(topk);
    std::vector<float> score(topk);
    for (int k = 0; k < topk; k++) {
      auto qt = q.top();
      int id = qt.second;
      float s = qt.first;
      q.pop();
      ss << ",top" << (topk - k - 1) << ",ind:" << id << ",score:" << s;
      ind[topk - k - 1] = id;
      score[topk - k - 1] = s;
    }
    LOG(INFO) << "ind size:" << ind.size() << ss.str();
    indices.push_back(ind);
    scores.push_back(score);
    ptr_res_start = ptr_res_start + N;
  }
  pthread_mutex_unlock(&lock_);
}
void Matcher::dot(float *p_features, int num_features, float *p_result) {
  if (!is_loaded) return;
  pthread_mutex_lock(&lock_);
  dot_impl(p_features, num_features);
  memcpy(p_result, m_pResultBuffer, num_features * N * sizeof(float));
  pthread_mutex_unlock(&lock_);
}

void Matcher::dot_impl(float *p_features, int num_features) {
  M = num_features;

  if (M > max_M) {
    LOG(FATAL) << "error input shape, max support:(" << max_M << "," << K
               << "),input:(" << M << "," << K << ")";
  }
  //  assert(M<=max_M);
  float *ptr_a = p_features;
  bm_handle_t handle = (bm_handle_t)handle_inst_;
  bm_device_mem_t *p_devmem_a = (bm_device_mem_t *)devmem_a_;
  bm_device_mem_t *p_devmem_b = (bm_device_mem_t *)devmem_b_;
  bm_device_mem_t *p_devmem_r = (bm_device_mem_t *)devmem_r_;

  p_devmem_a->size = num_features * K * sizeof(float);
  p_devmem_r->size = num_features * N * sizeof(float);
  LOG(INFO) << "start to copy A";
  bm_memcpy_s2d_partial(handle, *p_devmem_a, p_features, p_devmem_a->size);
  LOG(INFO) << "start to gemm";

  bmcv_gemm(handle, false, false, M, N, K, 1, *p_devmem_a, K, *p_devmem_b, N, 0,
            *p_devmem_r, N);
  LOG(INFO) << "start to copy output result";

  bm_memcpy_d2s(handle, m_pResultBuffer, *p_devmem_r);
  LOG(INFO) << "finish copy output result";
  p_devmem_a->size = max_M * K * sizeof(float);
  p_devmem_r->size = max_M * N * sizeof(float);
}

void Matcher::update_addition_result(float *p_features, int num_features) {
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> feature(
      num_features, K);
  memcpy(feature.data(), p_features, num_features * K * sizeof(float));

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result;
  for (auto &kv : updated_items_) {
    int idx = kv.first;
    result = feature * kv.second;
    float *p_q_res = result.data();
    for (int k = 0; k < num_features; k++) {
      float *p_res = m_pResultBuffer + k * N;
      p_res[idx] = p_q_res[k];
    }
  }
}
uint64_t Matcher::get_gallery_dev_addr() {
  uint64_t dev_addr = bm_mem_get_device_addr(*(bm_device_mem_t *)devmem_b_);
  return dev_addr;
}
