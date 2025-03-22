#ifndef TDL_SDK_FRAMEWORK_UTILS_QWEN_VL_HELPER_HPP
#define TDL_SDK_FRAMEWORK_UTILS_QWEN_VL_HELPER_HPP
#include <opencv2/opencv.hpp>

#include <map>
#include <string>

class QwenVLHelper {
 public:
  QwenVLHelper();
  ~QwenVLHelper();

  /**
   * @brief 从图片中提取帧
   *
   * @param image_path 图片路径
   * @param args ,resized_height,resized_width,或者min_pixels,max_pixels
   * @return r、g、b三个通道的帧列表
   */
  static std::vector<cv::Mat> fetchImage(
      const std::string &image_path, const std::map<std::string, int> &args);
  /**
   * @brief 从视频中提取帧
   *
   * @param video_path 视频路径
   * @param desired_fps 期望的帧率
   * @param desired_nframes 期望的帧数
   * @param max_video_sec
   * 最大视频时长,0表示不限制,只针对desired_nframes为0时有效
   * @return r、g、b三个通道的帧列表
   */
  static std::vector<std::vector<cv::Mat>> fetchVideo(
      const std::string &video_path, double desired_fps, int desired_nframes,
      int max_video_sec = 0);

  /**
   * @brief 测试从视频中提取帧的时间
   *
   * @param video_path 视频路径
   * @param desired_fps 期望的帧率
   * @param desired_nframes 期望的帧数
   * @param max_video_sec
   * 最大视频时长,0表示不限制,只针对desired_nframes为0时有效
   * @return r、g、b三个通道的帧列表
   */
  static std::map<std::string, float> testFetchVideoTs(
      const std::string &video_path, double desired_fps, int desired_nframes,
      int max_video_sec = 0);
};

#endif
