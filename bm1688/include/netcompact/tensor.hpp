#ifndef INCLUDE_NETCOMPACT_TENSOR_H_
#define INCLUDE_NETCOMPACT_TENSOR_H_

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>

#include <bmlib_runtime.h>



namespace nncompact {

//device memory could not be allocated manually here,just use memory from BMBlob
class Tensor {
 public:
  //mem_type:0 for host memory,1 for device memory
  Tensor(int data_size = 4, int mem_type = 0, void* p_context_handle = 0);

  ~Tensor() { release(); };

  void reshape(int n, int c, int h, int w);

  void resize(int size);

  void release();

  void alloc_mem(int size);

  std::vector<int> get_shape();

  int get_num_elems();

  int get_capacity();

  int get_size() { return num_elems_ * data_size_; }

  int get_data_size() { return data_size_; }

  int get_batch_size() { return shape_[1] * shape_[2] * shape_[3] * data_size_; }

  void* get_context_handle() { return context_handle_; }

  int get_mem_type() { return mem_type_; }

  int width();

  int height();

  int channels();

  int batch_num_elems();

  float* get_data(); //compatial for original interface

  void* get_device_data() { return device_memory_; }

  void share_data(void* p_data);

  std::vector<cv::Mat> construct_mat(int batch_idx);

  void flush();

  void set_zero(int size = 0);

  void sync_data(Tensor* p_other);

  void from_mat(const cv::Mat& img, int batch_idx = -1, int channel_idx = -1);

  void from_host_mem(const char* p_src_mem, int batch_idx = -1, int channel_idx = -1);

  void invalidate_device_mem();



  bm_device_mem_t get_device_memory() { return *(bm_device_mem_t*) device_memory_; };

  void from_dev_mem(const bm_device_mem_t* dev_mem, int batch_idx = -1, int channel_idx = -1);

  void from_dev_mem_with_size(const bm_device_mem_t* dev_mem, int size);

  void to_dev_mem(const bm_device_mem_t* dev_mem);

  void to_host_mem(char* p_dst_mem);



  void dump_to_file(const std::string& strfile);

  void load_from_file(const std::string& strfile);

 private:
  char* host_memory_;
  void* device_memory_;
  void* context_handle_;
  void* virtual_addr_ = 0;//only for soc mode

  std::vector<int> shape_;
  int num_elems_;
  int capacity_;
  int data_size_; //sizeof data type,fp32 is 4,int8 is 1
  int mem_type_;//0 for host memory,1 for device memory
  bool own_data_;

  Tensor(const Tensor&) = delete;

  Tensor& operator=(const Tensor&) = delete;

};  // class Tensor
}

#endif  // INCLUDE_NETCOMPACT_TENSOR_H_
