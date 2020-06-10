// Copyright 2018 Bitmain Inc.
// License
// Author
/**
 * @file image_utils.hpp
 * @ingroup NETWORKS_UTILS
 */
#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace qnn {
namespace vision {

void NormalizeAndSplitToU8(cv::Mat &prepared, const std::vector<float> &mean,
                           const std::vector<float> &scales, std::vector<cv::Mat> &channels);

void NormalizeAndSplitToU8(cv::Mat &prepared, std::vector<cv::Mat> &channels, float MeanR,
                           float alphaR, float MeanG, float alphaG, float MeanB, float alphaB,
                           int width, int height);

void NormalizeToU8(cv::Mat *src_channels, const std::vector<float> &mean,
                   const std::vector<float> &scales, std::vector<cv::Mat> &channels);

void AverageAndSplitToF32(cv::Mat &prepared, std::vector<cv::Mat> &channels, float MeanR,
                          float MeanG, float MeanB, int width, int height);

void NormalizeAndSplitToF32(cv::Mat &prepared, std::vector<cv::Mat> &channels, float MeanR,
                            float alphaR, float MeanG, float alphaG, float MeanB, float alphaB,
                            int width, int height);

const cv::Rect RectScale(const cv::Rect &rect, const int cols, const int rows,
                         const float scale_val, const int offset_x = 0, const int offset_y = 0);
}  // namespace vision
}  // namespace qnn
