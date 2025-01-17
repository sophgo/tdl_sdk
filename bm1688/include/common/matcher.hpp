#ifndef __FILE_MATCHER_HPP__
#define __FILE_MATCHER_HPP__

#include <Eigen/Eigen>

class Matcher {
public:
  Matcher(int device_id = 0);
  ~Matcher();
  void request_buffer(int num_gallery, int feat_dim);
  void load_buffer(float *p_gallery, int num_gallery, int feat_dim,
                   int num_allocated = 0);
  void load_raw(float *p_gallery, int num_gallery, int feature_size);
  std::vector<std::pair<int, float>> query_raw(float *p_features,
                                               int num_features);
  void update_gallery_col(void *p_data, int col);
  void dot(float *p_features, int num_features, float *p_result);
  void update_addition_result(float *p_features, int num_features);
  void query_raw_with_topk(float *p_features, int num_features, int topk,
                           std::vector<std::vector<int>> &indices,
                           std::vector<std::vector<float>> &scores);

  uint64_t get_gallery_dev_addr();
  void lock() { pthread_mutex_lock(&lock_); }
  void unlock() { pthread_mutex_unlock(&lock_); }
  void set_loaded(bool is_load) { is_loaded = is_load; }

private:
  void init(int device_id);
  void dot_impl(float *p_features, int num_features);
  void free_dev_mem();

  bool is_loaded;
  int M, K, N;
  const int max_M = 50;
  float *m_pResultBuffer;
  int uid_;
  pthread_mutex_t lock_ = PTHREAD_MUTEX_INITIALIZER;
  void *handle_inst_;
  void *devmem_a_, *devmem_b_, *devmem_r_;
  void *tmp_line_dev_mem_ = nullptr;

  void *devmem_b_img_;
  void *tmp_line_img_ = nullptr;

  std::map<int, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                              Eigen::RowMajor>>
      updated_items_;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      gallery_features_;
};

#endif
