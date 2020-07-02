#ifndef _TEMPLATE_MATCHING_H_
#define _TEMPLATE_MATCHING_H_

#include <stdlib.h>
#include <opencv2/opencv.hpp>

cv::Mat template_matching(const cv::Mat &crop_rgb_frame, const cv::Mat &ir_frame, cv::Rect box);
std::vector<cv::Mat> TTA_9_cropps(cv::Mat image);

#endif
