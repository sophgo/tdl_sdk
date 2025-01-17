#ifndef FRAMEWORK_BASE_MODEL_HPP__
#define FRAMEWORK_BASE_MODEL_HPP__
#define _USE_MATH_DEFINES

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

#include "bmruntime_interface.h"
#include "common/cv_utils.hpp"
#include "common/status.hpp"
#include "common/timer.hpp"
#include "netcompact/net.hpp"

namespace NNBaseModel {
enum ModelType {
  CSSD = 0,
  MTCNN = 1,
  SSH = 2,
  DET3 = 3,
  BMFACEV03M = 4,
  BMFACER18 = 5,
  MOBILESSD = 6,
  REID = 7,
  DDFA = 8,
  OCCLUSION = 9,
  BMFACER34 = 10,
  BLACKLIST = 11,
  AGE = 12,
  GENDER = 13,
  GLASSESMASK = 14,
  GENDERHAT = 15,
  RETINA = 16,
  GENDERAGE = 17,
  EXPRESSION = 18,
  MULTITASK = 19,
  OCCLUSION_V2 = 20,
  FACERETINA = 21,
  GENDERGLASSAGE = 22,
  BMMASKFACER34 = 23,
  SOPHONFASNET = 24,
  FACERETINA_IR = 25,
  FACESPOOF = 26,
  BMMARK = 27,
  BMFACER34_V2 = 28,
  BMFACER34_V2_FP32 = 29,
  SOPHONFOD = 30,
  YOLO_V5_VEHICLE = 31,
  BMFACER34_V3 = 32,
  BMVEHICLE_R18 = 33,
  CARPLATE_LANDMARK = 34,
  CARPLATE_OCR = 35,
  BMPERSON_R18,
  YOLOX_INT8,
  YOLO_V5_HEAD,
  YOLO_V5_VEHICLEX2,
  SCRFD,
  BMFACER50_V1,
};
};

class BaseModel {
 public:
  BaseModel() {}

  virtual ~BaseModel();

  virtual bmStatus_t setup() { return BM_COMMON_SUCCESS; }

  void print() { net_->print_net_mode(); }

  int get_fit_n(const int left_size);
  int get_device_id();
  void set_pad_value(int pad_val) { pad_value_ = pad_val; }
  int get_output_index(const std::string &str_out_name);

 protected:
  void add_avail_n(const std::vector<int> &avail_n);

  void set_input_n(const int);

  int get_input_n();

  int get_max_input_n();

  void wrap_input_layer(const int, std::vector<cv::Mat> &);

  void wrap_input_layer(const int, std::vector<cv::Mat> &, bool);

  void wrap_input_layer(const int, const std::vector<float> &);

  void setup_net(stNetParam &);

  void check_mean_scale();

 protected:
  stNetParam net_param_;
  std::shared_ptr<nncompact::Net> net_;

  int input_n_;  // 当前处理的batch值
  int channel_ = 3;
  int use_rgb_ = 0;  // 是否使用rgb顺序
  std::string input_layer_ = "data";
  std::string output_layer_;
  cv::Size input_geometry_;
  std::vector<int> avail_n_;  // 支持的batch数列表，一般是1/2/4/8
  std::vector<float> means_;
  std::vector<float> scales_;
  std::map<std::string, float>
      output_scales_;  // 网络输出的scale，fp32模型为1，int8量化模型有具体值
  bool is_int8_model_ = false;
  std::vector<cv::Mat> temp_bgr_;
  TimeRecorder timer_;
  int pad_value_ = 0;

 public:
  std::vector<std::vector<float>> batch_rescale_params_;
  std::vector<cv::Mat> tmp_bgr_planar_;
  cv::Mat temp_resized_;
  bmStatus_t forward(bool syn = true);
  void preprocess_opencv_async(const cv::Mat &img, cv::Mat &tmp_resized,
                               std::vector<cv::Mat> &tmp_bgr,
                               std::vector<cv::Mat> &bgr,
                               std::vector<float> &rescale_param);
  void preprocess_opencv_async(const cv::Mat &img, cv::Mat &tmp_resized,
                               std::vector<cv::Mat> &tmp_bgr,
                               std::vector<cv::Mat> &bgr);
  bmStatus_t preprocess_opencv(std::vector<cv::Mat>::const_iterator &img_iter,
                               int batch_size);
  void create_bgr_channels(std::vector<cv::Mat> &bgr);
  void preprocess_opencv_base(const cv::Mat &img, std::vector<cv::Mat> &bgr);

  cv::Size get_process_size() { return input_geometry_; }
  std::string model_dir_ = "";
};

#endif
