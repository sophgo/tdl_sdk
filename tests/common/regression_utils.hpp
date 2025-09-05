#pragma once
#include <cmath>
#include <cstdio>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "utils/tdl_log.hpp"

namespace cvitdl {
namespace unitest {

std::string gen_model_suffix();
std::string get_platform_str();
std::string gen_model_dir();
std::vector<std::string> get_platform_list();
std::string extractModelIdFromName(const std::string &model_name);
std::map<std::string, float> getCustomRegressionConfig(
    const std::string &model_name);
std::vector<std::string> getFileList(const std::string &dir_path,
                                     const std::string &extension);
bool matchObjects(const std::vector<std::vector<float>> &gt_objects,
                  const std::vector<std::vector<float>> &pred_objects,
                  const float iout_thresh, const float score_diff_thresh);

bool matchScore(const std::vector<float> &gt_info,
                const std::vector<float> &pred_info,
                const float score_diff_thresh);

bool matchKeypoints(const std::vector<float> &gt_keypoints_x,
                    const std::vector<float> &gt_keypoints_y,
                    const std::vector<float> &gt_keypoints_score,
                    const std::vector<float> &pred_keypoints_x,
                    const std::vector<float> &pred_keypoints_y,
                    const std::vector<float> &pred_keypoints_score,
                    const float position_thresh, const float score_diff_thresh);
bool matchSegmentation(const cv::Mat &mat1, const cv::Mat &mat2,
                       float mask_thresh);
}  // namespace unitest
}  // namespace cvitdl