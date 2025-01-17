#ifndef CV_UTILS_OPT_HPP_
#define CV_UTILS_OPT_HPP_

#include <face/face_common.hpp>
#include <map>
#include <netcompact/tensor.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core_c.h>
#include <string>
#include "common/obj_det_utils.hpp"



void bgr_split_scale1(const cv::Mat &src_mat, std::vector<cv::Mat> &tmp_bgr,
                      const std::vector<cv::Mat> &input_channels,
                      const std::vector<float> &mean,
                      const std::vector<float> &scale, bool use_rgb = false);


void pad_resize_to_dst(const cv::Mat &src_img, cv::Mat &dst_img,
                       std::vector<float> &rescale_params,
                       cv::InterpolationFlags inter_flag = cv::INTER_NEAREST);
// ret,true:need pad resize,else resize directly
bool compute_pad_resize_param(cv::Size src_size, cv::Size dst_size,
                              std::vector<float> &scale_param);


std::vector<cv::Mat> wrap_img_channels(char *p_src_img, int width, int height,
                                       int c, int batch, int img_type);

void align_frame_faces_cpu(std::vector<cv::Mat> &frame_imgs,
                           std::vector<std::vector<FacePts>> &frame_landmarks,
                           std::vector<cv::Mat> &aligned_faces);

float cal_iou(const cv::Rect &box1,const cv::Rect &box2);
float box_in_ratio(const cv::Rect &box1,const cv::Rect &box2);
void normalize_feature(std::vector<float> &feature);
void align_carplate(const stObjPts &landmarks,const cv::Mat &src_mat,cv::Mat &dst_mat);

void get_score(FaceRect &detector_info);
void GetHeadPose(const FacePts &pFacial5points, FacePose &hp);
void Predict3DFacialNormal(const cv::Point& noseTip, const cv::Point& noseBase,
                          const cv::Point& midEye, const cv::Point& midMouth,
                          FacePose &hp);
float CalDistance(const cv::Point& p1, const cv::Point& p2);
float CalAngle(const cv::Point& pt1, const cv::Point& pt2);
float CalSlant(int ln, int lf, const float Rn, float theta);
template <typename T>
T Saturate(const T& val, const T& minVal, const T& maxVal);


#endif
