#include "framework/base_model.hpp"

#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <log/Logger.hpp>

#include "netcompact/net_factory.hpp"

BaseModel::~BaseModel() {}
void BaseModel::add_avail_n(const std::vector<int> &avail_n) {
  avail_n_ = avail_n;
  std::sort(avail_n_.begin(), avail_n_.end(), std::greater<int>());
}

void BaseModel::set_input_n(const int n) {
  std::shared_ptr<nncompact::Tensor> input_tensor =
      net_->get_input_tensor(input_layer_);
  input_tensor->reshape(n, channel_, input_geometry_.height,
                        input_geometry_.width);
  input_n_ = n;
}

int BaseModel::get_fit_n(const int left_size) {
  int fit_n = 0;

  for (size_t i = 0; i < avail_n_.size(); ++i) {
    if (avail_n_[i] <= left_size) {
      fit_n = avail_n_[i];
      break;
    }
  }

  return fit_n;
}

int BaseModel::get_input_n() { return input_n_; }

int BaseModel::get_max_input_n() {
  return avail_n_.size() > 0 ? avail_n_[0] : 0;
}
int BaseModel::get_device_id() {
  if (net_) {
    return net_->get_device_id();
  } else {
    LOG(ERROR) << "net_ has not been setup";
    return 0;
  }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void BaseModel::wrap_input_layer(const int batch,
                                 std::vector<cv::Mat> &input_channels) {
  std::shared_ptr<nncompact::Tensor> input_tensor =
      net_->get_input_tensor(input_layer_);
  std::vector<int> input_shape = input_tensor->get_shape();
  int c = input_shape[1];
  int h = input_shape[2];
  int w = input_shape[3];

  char *input_data = (char *)input_tensor->get_data();
  int data_size = input_tensor->get_data_size();
  input_data += batch * w * h * c * data_size;
  for (int i = 0; i < c; ++i) {
    cv::Mat channel;
    if (data_size == 4) {
      channel = cv::Mat(h, w, CV_32FC1, input_data);
    } else {
      channel = cv::Mat(h, w, CV_8SC1, input_data);
    }

    input_channels.push_back(channel);
    input_data += w * h * data_size;
  }
}

void BaseModel::wrap_input_layer(const int batch,
                                 std::vector<cv::Mat> &input_channels,
                                 bool is_float) {
  std::shared_ptr<nncompact::Tensor> input_tensor =
      net_->get_input_tensor(input_layer_);
  std::vector<int> input_shape = input_tensor->get_shape();
  int c = input_shape[1];
  int h = input_shape[2];
  int w = input_shape[3];

  float *input_data = input_tensor->get_data();
  // int data_size = input_tensor->get_data_size();
  input_data += batch * w * h * c;
  for (int i = 0; i < c; ++i) {
    cv::Mat channel;
    channel = cv::Mat(h, w, CV_32FC1, input_data);
    input_channels.push_back(channel);
    input_data += w * h;
  }
}

void BaseModel::wrap_input_layer(const int batch,
                                 const std::vector<float> &arr) {
  std::shared_ptr<nncompact::Tensor> input_tensor =
      net_->get_input_tensor(input_layer_);
  std::vector<int> input_shape = input_tensor->get_shape();
  int c = input_shape[1];
  int h = input_shape[2];
  int w = input_shape[3];
  float *input_data = input_tensor->get_data();
  input_data += batch * w * h * c;
  int index = 0;
  for (int i = 0; i < c; ++i) {
    for (int j = 0; j < h; ++j) {
      for (int k = 0; k < w; ++k) {
        index = i * w * h + j * w + k;
        input_data[index] = arr[index];
      }
    }
  }
}

void BaseModel::setup_net(stNetParam &net_param) {
  nncompact::NetFactory net_factory;

  net_ = net_factory.create_net(nncompact::NetFactory::BM1684, net_param);
  net_->setup();
  for (auto &n : net_->input_names_) {
    net_->add_input(n);
  }
  for (auto &n : net_->output_names_) {
    net_->add_output(n);
  }
  std::vector<int> avail_n = net_->available_batches_[net_->input_names_[0]];

  add_avail_n(avail_n);
  means_ = net_param.mean;
  scales_ = net_param.scale;

  const bm_net_info_t *net_info = (const bm_net_info_t *)net_->get_net_info();
  if (net_info->input_dtypes[0] == BM_INT8) {
    is_int8_model_ = true;
  }
  for (int i = 0; i < net_info->output_num; i++) {
    output_scales_[net_info->output_names[i]] = net_info->output_scales[i];
  }

  check_mean_scale();

  std::vector<int> input0_shape = net_->input_shapes_[net_->input_names_[0]];
  input_geometry_ = cv::Size(input0_shape[3], input0_shape[2]);
  channel_ = input0_shape[1];
  input_layer_ = net_->input_names_[0];
  output_layer_ = net_->output_names_[0];
  use_rgb_ = net_param.use_rgb;
  temp_resized_ = cv::Mat::zeros(input_geometry_, CV_8UC3);

  std::stringstream ss;
  ss << "batches:";
  for (auto n : avail_n_) {
    ss << n << ",";
  }
  ss << "input_names:";
  for (auto &n : net_->input_names_) {
    ss << n << ",";
  }
  ss << "output_names:";
  for (auto &n : net_->output_names_) {
    ss << n << ",";
  }
  ss << "mean:";
  for (auto m : means_) {
    ss << m << ",";
  }
  ss << "scale:";
  for (auto s : scales_) {
    ss << s << ",";
  }
  ss << "input_shape:" << input_geometry_ << "use_rgb:" << use_rgb_;
  LOG(INFO) << ss.str();
}

void BaseModel::check_mean_scale() {
  int mean_size = means_.size();
  int scale_size = scales_.size();

  float m = 0;
  if (mean_size != 0) m = means_[0];

  float s = 1.0;
  if (scale_size != 0) s = scales_[0];
  // if is int8 model,overide it
  const bm_net_info_t *net_info = (const bm_net_info_t *)net_->get_net_info();
  if (net_info->input_dtypes[0] == BM_INT8) {
    float s1 = net_info->input_scales[0];
    m *= s1;
    s *= s1;
    LOG(INFO) << "configured means and scales would be cleared for net:"
              << net_info->name;
    means_.clear();
    scales_.clear();
  }

  if (mean_size != channel_) {
    for (int i = means_.size(); i < channel_; i++) {
      means_.push_back(m);
    }
  }

  if (scale_size != channel_) {
    for (int i = scales_.size(); i < channel_; i++) {
      scales_.push_back(s);
    }
  }

  for (int i = 0; i < channel_; i++) {
    cv::Mat tmp = cv::Mat::zeros(input_geometry_, CV_8U);
    temp_bgr_.push_back(tmp);
  }
}

void BaseModel::create_bgr_channels(std::vector<cv::Mat> &bgr) {
  int type = CV_32FC1;
  const bm_net_info_t *net_info = (const bm_net_info_t *)net_->get_net_info();
  if (net_info->input_dtypes[0] == BM_INT8) {
    type = CV_8SC1;
  } else if (net_info->input_dtypes[0] == BM_UINT8) {
    type = CV_8UC1;
  }

  LOG(INFO) << "channel size:" << bgr.size() << ",img_size:" << input_geometry_
            << ",type:" << type;
  if (bgr.size() == 3) {
    int is_same = 1;
    for (int i = 0; i < 3; i++) {
      if (bgr[i].type() != type || bgr[i].cols != input_geometry_.width ||
          bgr[i].rows != input_geometry_.height) {
        is_same = 0;
        std::cout << bgr[i].type() << "," << type << ",size:" << bgr[i].cols
                  << "," << input_geometry_.width << std::endl;
        break;
      }
    }
    if (!is_same) bgr.clear();
  } else {
    bgr.clear();
  }
  for (int i = bgr.size(); i < 3; i++) {
    bgr.push_back(cv::Mat::zeros(input_geometry_, type));
  }
}

bmStatus_t BaseModel::forward(bool syn) {
  net_->forward(syn);
  return BM_COMMON_SUCCESS;
}
// rescale_param:scalex,scaley,offsetx,offsety
// outx = (model_x*input_geometry.width+offsetx)*scalex
void BaseModel::preprocess_opencv_async(const cv::Mat &img,
                                        cv::Mat &tmp_resized,
                                        std::vector<cv::Mat> &tmp_bgr,
                                        std::vector<cv::Mat> &bgr,
                                        std::vector<float> &rescale_param) {
  cv::Mat resized;
  create_bgr_channels(bgr);
  rescale_param.clear();
  if (tmp_resized.cols != input_geometry_.width ||
      tmp_resized.rows != input_geometry_.height) {
    // TODO(fuquan.ke): need specify device id? not necessary, only for soc mode
    tmp_resized = cv::Mat::zeros(input_geometry_, CV_8UC3);
  }
  if (img.cols == input_geometry_.width && img.rows == input_geometry_.height) {
    rescale_param.push_back(1);
    rescale_param.push_back(1);
    rescale_param.push_back(0);
    rescale_param.push_back(0);
    resized = img;
    LOG(INFO) << "do not resize";
  } else if (net_param_.resize_mode == IMG_PAD_RESIZE) {
    tmp_resized.setTo(pad_value_);
    pad_resize_to_dst(img, tmp_resized, rescale_param);
    resized = tmp_resized;
    LOG(INFO) << "resize using pad_resize";
  } else {
    const cv::Mat &src_img = img;
    rescale_param.push_back(src_img.cols * 1.0 / input_geometry_.width);
    rescale_param.push_back(src_img.rows * 1.0 / input_geometry_.height);
    rescale_param.push_back(0.0f);
    rescale_param.push_back(0.0f);
    cv::resize(src_img, tmp_resized, input_geometry_, 0, 0, cv::INTER_NEAREST);
    resized = tmp_resized;
    LOG(INFO) << "resize directly";
  }

  bgr_split_scale1(resized, tmp_bgr, bgr, means_, scales_, use_rgb_);
}
void BaseModel::preprocess_opencv_base(const cv::Mat &img,
                                       std::vector<cv::Mat> &bgr) {
  preprocess_opencv_async(img, temp_resized_, temp_bgr_, bgr);
}
void BaseModel::preprocess_opencv_async(const cv::Mat &img,
                                        cv::Mat &tmp_resized,
                                        std::vector<cv::Mat> &tmp_bgr,
                                        std::vector<cv::Mat> &bgr) {
  std::vector<float> scales;
  preprocess_opencv_async(img, tmp_resized, tmp_bgr, bgr, scales);
}

bmStatus_t BaseModel::preprocess_opencv(
    std::vector<cv::Mat>::const_iterator &img_iter, int batch_size) {
  set_input_n(batch_size);
  std::shared_ptr<nncompact::Tensor> input_tensor =
      net_->get_input_tensor(input_layer_);

  batch_rescale_params_.resize(batch_size);
  for (int i = 0; i < batch_size; i++) {
    preprocess_opencv_async(*(img_iter + i), temp_resized_, temp_bgr_,
                            tmp_bgr_planar_, batch_rescale_params_[i]);
    if (use_rgb_) {
      LOG(INFO) << "BGR2RGB";
      std::swap(tmp_bgr_planar_[0], tmp_bgr_planar_[2]);
    }
    for (int j = 0; j < tmp_bgr_planar_.size(); j++)
      input_tensor->from_mat(tmp_bgr_planar_[j], i, j);
  }
  net_->update_input_tensors();
  return BM_COMMON_SUCCESS;
}

int BaseModel::get_output_index(const std::string &str_out_name) {
  int idx = -1;
  for (int i = 0; i < net_->output_names_.size(); i++) {
    if (str_out_name == std::string(net_->output_names_[i])) {
      idx = i;
      break;
    }
  }
  return idx;
}