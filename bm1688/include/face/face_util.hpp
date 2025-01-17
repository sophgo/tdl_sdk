#ifndef FACE_UTIL_HPP_
#define FACE_UTIL_HPP_
#include <vector>

#include <opencv2/opencv.hpp>

#include "face/face_common.hpp"



cv::Mat align_face(const cv::Mat &src, const FacePts facePt, int width,
                   int height);

cv::Mat align_eye(const cv::Mat &src, const FaceRect rect, const FacePts facePt,
                  int width, int height);

cv::Mat align_face_to_dest(const cv::Mat &src, cv::Mat &aligned,
                           const FacePts &facePt, int width, int height);

float calc_cosine(const std::vector<float> &feature1,
                  const std::vector<float> &feature2);

cv::Mat calc_transform_matrix(const FacePts &face_pt,
                              cv::Size dst_size = cv::Size(112, 112));

std::vector<float> face_pose_estimate(const FacePts &facept, cv::Size imsize);
cv::Mat similarTransform(cv::Mat src, cv::Mat dst);
#endif
