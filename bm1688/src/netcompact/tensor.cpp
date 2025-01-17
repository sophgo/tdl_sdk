#include "netcompact/tensor.hpp"

#include <cassert>
#include <cstring>
#include <log/Logger.hpp>

#include "common/status.hpp"

namespace nncompact {

Tensor::Tensor(int data_size /*=4*/, int mem_type /*=0*/,
               void *p_context_handle /*=0*/) {
  shape_.resize(4, 0);
  num_elems_ = shape_[0] * shape_[1] * shape_[2] * shape_[3];
  data_size_ = data_size;
  capacity_ = num_elems_ * data_size_;

  mem_type_ = mem_type;
  device_memory_ = 0;
  own_data_ = true;

  host_memory_ = 0;
  device_memory_ = 0;
  context_handle_ = p_context_handle;
}

void Tensor::reshape(int n, int c, int h, int w) {
  shape_[0] = n;
  shape_[1] = c;
  shape_[2] = h;
  shape_[3] = w;
  resize(n * c * h * w);
}

void Tensor::resize(int num_elements) {
  int size = num_elements * data_size_;
  if (size > capacity_ && own_data_) {
    release();
    alloc_mem(size);
  }
  num_elems_ = num_elements;
}

void Tensor::alloc_mem(int size) {
  if (mem_type_ == 0) {
    host_memory_ = new char[size];
  } else {
    assert(context_handle_ != nullptr);
    auto p_dev = new bm_device_mem_t();
    bm_malloc_device_byte(bm_handle_t(context_handle_), p_dev, size);
    device_memory_ = p_dev;
#if defined(USE_ARM)
    unsigned long long addr;
    bm_mem_mmap_device_mem((bm_handle_t)context_handle_,
                           (bm_device_mem_t *)device_memory_, &addr);
    virtual_addr_ = (void *)addr;
#endif
  }
  capacity_ = size;
  own_data_ = true;
}

void Tensor::release() {
  if (own_data_) {
    if (mem_type_ == 0 && host_memory_) {
      delete[] host_memory_;
      host_memory_ = 0;
    } else if (device_memory_) {
#ifdef USE_ARM
      if (virtual_addr_) {
        bm_mem_unmap_device_mem((bm_handle_t)context_handle_, virtual_addr_,
                                capacity_);
        virtual_addr_ = 0;
      }
#endif
      bm_device_mem_t *p_dev = (bm_device_mem_t *)device_memory_;
      bm_free_device(bm_handle_t(context_handle_), *p_dev);
      delete p_dev;
      device_memory_ = nullptr;
    }
  }
  capacity_ = 0;
}

void Tensor::sync_data(Tensor *p_other) {
  std::vector<int> shape = p_other->get_shape();

  for (int i = 0; i < shape.size(); i++) {
    // LOG(INFO)<<"shape "<<i<<",cur:"<<shape_[i]<<",other:"<<shape[i];
    assert(shape[i] == shape_[i]);
  }

  if (mem_type_ == 0) {
    if (p_other->mem_type_ == 0) {
      memcpy(host_memory_, p_other->get_data(), get_size());
    } else {
      bm_memcpy_d2s_partial(bm_handle_t(context_handle_), host_memory_,
                            p_other->get_device_memory(), get_size());
    }
  } else {
    bm_device_mem_t *dev_mem = (bm_device_mem_t *)device_memory_;
    if (p_other->mem_type_ == 0) {
      bm_memcpy_s2d_partial(bm_handle_t(context_handle_), *dev_mem,
                            p_other->get_data(), get_size());
    } else {
      bm_device_mem_t dev_mem_other = p_other->get_device_memory();
      bm_memcpy_d2d_byte(bm_handle_t(context_handle_), *dev_mem, 0,
                         dev_mem_other, 0, get_size());
    }
  }
}

void Tensor::from_dev_mem(const bm_device_mem_t *dev_mem,
                          int batch_idx /*= -1*/, int channel_idx /*=-1*/) {
  int copy_size = get_size();
  int offset_pos = 0;
  if (batch_idx != -1) {
    int batch_size = copy_size / shape_[0];
    assert(batch_idx >= 0 && batch_idx < shape_[0]);
    offset_pos = batch_size * batch_idx;
    copy_size = batch_size;
    if (channel_idx != -1) {
      copy_size = shape_[2] * shape_[3] * data_size_;
      assert(channel_idx >= 0 && channel_idx < shape_[1]);
      offset_pos += channel_idx * copy_size;
    }
  }

  // TODO(fuquan.ke): if input tensor type is not device and use bmcv
  // this interface would not work properly
  if (mem_type_ == 0) {
    // copy data from device memory to host
    bm_memcpy_d2s_partial(bm_handle_t(context_handle_), host_memory_, *dev_mem,
                          get_size());
  } else {
    bm_memcpy_d2d_byte(bm_handle_t(context_handle_),
                       *(bm_device_mem_t *)device_memory_, offset_pos, *dev_mem,
                       0, copy_size);
  }
}

void Tensor::from_dev_mem_with_size(const bm_device_mem_t *dev_mem, int size) {
  int copy_size = get_size();
  if (size > copy_size) {
    LOG(FATAL) << "copy size overflow,maxsize:" << copy_size << ",want "
               << size;
  }
  if (mem_type_ == 0) {
    // copy data from device memory to host
    bm_memcpy_d2s_partial(bm_handle_t(context_handle_), host_memory_, *dev_mem,
                          size);
  } else {
    bm_memcpy_d2d_byte(bm_handle_t(context_handle_),
                       *(bm_device_mem_t *)device_memory_, 0, *dev_mem, 0,
                       size);
  }
}

void Tensor::to_dev_mem(const bm_device_mem_t *dev_mem) {
  if (mem_type_ == 0) {
    bm_memcpy_s2d_partial(bm_handle_t(context_handle_), *dev_mem, host_memory_,
                          get_size());
  } else {
    bm_memcpy_d2d_byte(bm_handle_t(context_handle_), *dev_mem, 0,
                       *(bm_device_mem_t *)device_memory_, 0, get_size());
  }
}

void Tensor::to_host_mem(char *p_dst_mem) {
  if (mem_type_ == 0) {
    memcpy(p_dst_mem, host_memory_, get_size());
  } else {
    bm_memcpy_d2s_partial(bm_handle_t(context_handle_), p_dst_mem,
                          *(bm_device_mem_t *)device_memory_, get_size());
  }
}

void Tensor::invalidate_device_mem() {
  if (mem_type_ == 1) {
#ifdef USE_ARM
    bm_mem_invalidate_device_mem(bm_handle_t(context_handle_),
                                 (bm_device_mem_t *)device_memory_);
#endif
  }
}

void Tensor::from_host_mem(const char *p_src_mem, int batch_idx /*= -1*/,
                           int channel_idx /*=-1*/) {
  int copy_size = get_size();
  int offset_pos = 0;
  if (batch_idx != -1) {
    int batch_size = copy_size / shape_[0];
    assert(batch_idx >= 0 && batch_idx < shape_[0]);
    offset_pos = batch_size * batch_idx;
    copy_size = batch_size;
    if (channel_idx != -1) {
      copy_size = shape_[2] * shape_[3] * data_size_;
      assert(channel_idx >= 0 && channel_idx < shape_[1]);
      offset_pos += channel_idx * copy_size;
    }
  }
  if (mem_type_ == 0) {
    memcpy(host_memory_ + offset_pos, p_src_mem, copy_size);
  } else {
    uint64_t image_bgr_devaddr =
        bm_mem_get_device_addr(*(bm_device_mem_t *)device_memory_);
    bm_device_mem_t dst_addr =
        bm_mem_from_device(image_bgr_devaddr + offset_pos, copy_size);
    bm_memcpy_s2d_partial(bm_handle_t(context_handle_), dst_addr,
                          const_cast<char *>(p_src_mem), copy_size);
  }
}

// TODO: differ in bmcv
void Tensor::from_mat(const cv::Mat &img, int batch_idx /*= -1*/,
                      int channel_idx /*=-1*/) {
  int sz = CV_ELEM_SIZE(img.type());
  int line_w = img.cols * sz;
  if (img.step[0] == line_w)
    return from_host_mem((const char *)img.data, batch_idx, channel_idx);

  int copy_size = get_size();
  LOG(INFO) << "imgsize:" << img.size() << ",step:" << img.step[0]
            << ",linew:" << line_w;
  int offset_pos = 0;
  if (batch_idx != -1) {
    int batch_size = copy_size / shape_[0];
    assert(batch_idx >= 0 && batch_idx < shape_[0]);
    offset_pos = batch_size * batch_idx;
    copy_size = batch_size;
    if (channel_idx != -1) {
      copy_size = shape_[2] * shape_[3] * data_size_;
      assert(channel_idx >= 0 && batch_idx < shape_[1]);
      offset_pos += channel_idx * copy_size;
      assert(line_w * img.rows == copy_size);
    }
  }
  for (int r = 0; r < img.rows; r++) {
    const uchar *ptr_row = img.ptr<uchar>(r);
    if (mem_type_ == 0) {
      memcpy(host_memory_ + offset_pos + line_w * r, ptr_row, line_w);
    } else {
      uint64_t image_bgr_devaddr =
          bm_mem_get_device_addr(*(bm_device_mem_t *)device_memory_);
      bm_device_mem_t dst_addr = bm_mem_from_device(
          image_bgr_devaddr + offset_pos + r * line_w, line_w);
      bm_memcpy_s2d_partial(bm_handle_t(context_handle_), dst_addr,
                            const_cast<char *>((const char *)ptr_row), line_w);
    }
  }
}

std::vector<cv::Mat> Tensor::construct_mat(int batch_idx) {
  int data_size = get_data_size();
  int batch_size = get_size() / shape_[0];
  int type = CV_32FC1;
  if (data_size == 1) type = CV_8SC1;
  uchar *p_dst_data = 0;
#ifdef USE_ARM
  if (mem_type_ == 0)
    p_dst_data = (uchar *)get_data();
  else {
    p_dst_data = (uchar *)virtual_addr_;
    LOG(INFO) << "got virtual address from device memory";
  }
#else
  if (mem_type_ == 1)
    LOG(FATAL) << "pcie mode do not support construct mat from device memory";
  p_dst_data = (uchar *)get_data();
#endif

  uchar *p_img_data = p_dst_data + batch_idx * batch_size;
  // int channel_size = shape_[3] * shape_[2];
  int channel_size = shape_[3] * shape_[2] * data_size;
  std::vector<cv::Mat> channels;
  for (int i = 0; i < shape_[1]; i++) {
    cv::Mat channel(cv::Size(shape_[3], shape_[2]), type,
                    p_img_data + i * channel_size);
    channels.push_back(channel);
  }
  return channels;
}

void Tensor::flush() {
#if defined(USE_ARM)
  if (mem_type_ == 1) {
    bm_mem_flush_partial_device_mem((bm_handle_t)context_handle_,
                                    (bm_device_mem_t *)device_memory_, 0,
                                    get_size());
  }

#endif
}

void Tensor::set_zero(int size /*=0*/) {
  int max_size = get_size();
  if (size == 0)
    size = max_size;
  else if (size > max_size) {
    size = max_size;
  }

  if (mem_type_ == 0) {
    memset(host_memory_, 0, size);
  }
}

void Tensor::share_data(void *p_data) {
  release();
  if (mem_type_ == 0) {
    host_memory_ = (char *)p_data;
  } else {
    device_memory_ = p_data;
  }
  own_data_ = false;
}

int Tensor::width() { return shape_[3]; }

int Tensor::height() { return shape_[2]; }

int Tensor::channels() { return shape_[1]; }

int Tensor::batch_num_elems() { return shape_[1] * shape_[2] * shape_[3]; }

int Tensor::get_num_elems() {
  return shape_[0] * shape_[1] * shape_[2] * shape_[3];
}

int Tensor::get_capacity() { return capacity_; }

float *Tensor::get_data() {
  if (mem_type_ == 0) {
    return (float *)host_memory_;
  } else {
#ifdef USE_ARM
    return (float *)
        virtual_addr_;  // make sure the device memory has been invalidated
#else
    LOG(FATAL) << "not supported for pcie mode";
#endif
  }
}

std::vector<int> Tensor::get_shape() { return shape_; }

void Tensor::dump_to_file(const std::string &strfile) {
  int data_size = get_size();
  void *p_data = get_data();
  if (p_data) {
    FILE *fp = fopen(strfile.c_str(), "wb");
    if (fp == nullptr) {
      LOG(WARNING) << "error,open file :" << strfile << " failed";
    } else {
      fwrite(p_data, data_size, 1, fp);
      LOG(INFO) << "write size:" << data_size;
      fclose(fp);
    }
  } else {
    LOG(WARNING) << "has no host memory,can't dump";
  }
}

void Tensor::load_from_file(const std::string &strfile) {
  int data_size = get_size();
  void *p_data = get_data();
  FILE *fp = fopen(strfile.c_str(), "rb");
  if (fp == nullptr) {
    LOG(WARNING) << "error,open file :" << strfile << " failed";
  } else {
    fseek(fp, 0, SEEK_END);
    int len = ftell(fp);
    if (len != data_size) {
      LOG(FATAL) << "filesize not equal,got:" << len << ",expect:" << data_size;
    }
    fseek(fp, 0, SEEK_SET);
    fread(p_data, data_size, 1, fp);
  }
  fclose(fp);
}
}  // namespace nncompact
