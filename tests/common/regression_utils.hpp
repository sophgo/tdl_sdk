#pragma once
#include <dirent.h>
#include <pthread.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <map>
#include <mutex>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include "common/model_output_types.hpp"
#include "image/base_image.hpp"
#include "parse_logs.hpp"
#include "tdl_model_factory.hpp"
#include "utils/common_utils.hpp"
#include "utils/tdl_log.hpp"

#define DEFAULT_FPS_PERIOD 66.67f  // default fps period is 15fps
namespace cvitdl {
namespace unitest {

std::string gen_model_suffix();
std::string get_platform_str();
std::string gen_model_dir();
std::vector<std::string> get_platform_list();
std::string extractModelIdFromName(const std::string& model_name);
std::map<std::string, float> getCustomRegressionConfig(
    const std::string& model_name);
std::vector<std::string> getFileList(const std::string& dir_path,
                                     const std::string& extension);
bool matchObjects(const std::vector<std::vector<float>>& gt_objects,
                  const std::vector<std::vector<float>>& pred_objects,
                  const float iout_thresh, const float score_diff_thresh);

bool matchScore(const std::vector<float>& gt_info,
                const std::vector<float>& pred_info,
                const float score_diff_thresh);

bool matchKeypoints(const std::vector<float>& gt_keypoints_x,
                    const std::vector<float>& gt_keypoints_y,
                    const std::vector<float>& gt_keypoints_score,
                    const std::vector<float>& pred_keypoints_x,
                    const std::vector<float>& pred_keypoints_y,
                    const std::vector<float>& pred_keypoints_score,
                    const float position_thresh, const float score_diff_thresh);
bool matchSegmentation(const cv::Mat& mat1, const cv::Mat& mat2,
                       float mask_thresh);

bool time_consume_show(
    const std::unordered_map<std::string, std::vector<double>>& img_durations);
void enable_tpu_usage(const std::string& tpu_usage_path);
bool read_tpu_usage(const std::string& tpu_usage_path,
                    std::vector<double>& tpu_samples, std::mutex& tpu_mu);
bool read_cpu_line(std::vector<double>& cpu_samples, std::mutex& cpu_mu);
bool confirm_path(std::string& tpu_usage_path);
void load_show(const std::vector<double>& cpu_samples,
               const std::vector<double>& tpu_samples);
pid_t get_tid_by_name(const std::string& thread_name);
std::string fileSizeInMB(const std::string& path, bool useMiB = true);
int parse_cmd_result(std::string& cmd, const std::regex& re,
                     std::string& result);
int get_model_info(const std::string& model_path, float& infer_time_ms);
void* run_tdl_thread_tpu(void* args);
void* run_tdl_thread_cpu(void* args);
void run_performance(const std::string& model_path,
                     std::shared_ptr<BaseImage>& input_images,
                     std::shared_ptr<BaseModel> model,
                     float fps_period = DEFAULT_FPS_PERIOD);
// 结构体1 TPU循环使用的
typedef struct {
  std::string model_path;
  int count;
} RUN_TDL_THREAD_TPU_ARG_S;

// 结构体2 CPU循环使用的
typedef struct {
  std::shared_ptr<BaseImage> input_images;
  std::shared_ptr<BaseModel> model_;
  float fps_period;
  ModelType model_type;
} RUN_TDL_THREAD_CPU_ARG_S;

class ScopedSampler_tpu {
 private:
  // 成员采用引用，是为了让ScopedSampler直接控制外部的sampling、th_cpu、th_tpu。
  std::atomic<bool>& sampling;
  std::thread& th_tpu;

 public:
  ScopedSampler_tpu(std::atomic<bool>& s, std::thread& tpu)
      : sampling(s), th_tpu(tpu) {
    sampling.store(true);
  }
  ~ScopedSampler_tpu() {
    sampling.store(false);
    if (th_tpu.joinable()) th_tpu.join();
  }
};

class ScopedSampler_cpu {
 private:
  // 成员采用引用，是为了让ScopedSampler直接控制外部的sampling、th_cpu、th_tpu。
  std::atomic<bool>& sampling;
  std::thread& th_cpu;

 public:
  ScopedSampler_cpu(std::atomic<bool>& s, std::thread& cpu)
      : sampling(s), th_cpu(cpu) {
    sampling.store(false);
  }
  ~ScopedSampler_cpu() {
    sampling.store(false);
    if (th_cpu.joinable()) th_cpu.join();
  }
};

}  // namespace unitest
}  // namespace cvitdl