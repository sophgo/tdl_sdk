#ifndef _TEMPLATE_MATCHING_H_
#define _TEMPLATE_MATCHING_H_

#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "core/face/cvai_face_types.h"

cv::Mat template_matching(const cv::Mat &crop_rgb_frame, const cv::Mat &ir_frame, cv::Rect box,
                          cvai_liveness_ir_position_e ir_pos);
std::vector<cv::Mat> TTA_9_cropps(cv::Mat image);

#endif
