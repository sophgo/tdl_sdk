#include "occlusion_classification.hpp"
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <error_msg.hpp>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include "core/core/cvtdl_errno.h"
#include "core/cvi_tdl_types_mem_internal.h"
#include "core/face/cvtdl_face_types.h"
#include "core/object/cvtdl_object_types.h"
#include "core/utils/vpss_helper.h"
#include "core_utils.hpp"
#include "cvi_comm.h"
#include "misc.hpp"
#ifdef ENABLE_CVIAI_CV_UTILS
#include "cv/imgproc.hpp"
#else
#include "opencv2/imgproc.hpp"
#endif
#define OCCULUSION_CLASSIFICATION_FACTOR (float)(1 / 127.5)
#define OCCULUSION_CLASSIFICATION_MEAN (1.0)

namespace cvitdl {

// 计算均值
cv::Mat boxFilter(const cv::Mat &src, int radius) {
  cv::Mat dst;
  cv::boxFilter(src, dst, CV_32F, cv::Size(radius, radius));
  return dst;
}

// 导向滤波实现
void guidedFilter(const cv::Mat &I, const cv::Mat &p, cv::Mat &q, int r, double eps) {
  cv::Mat I_mean = boxFilter(I, r);
  cv::Mat p_mean = boxFilter(p, r);
  cv::Mat Ip_mean = boxFilter(I.mul(p), r);
  cv::Mat cov_Ip = Ip_mean - I_mean.mul(p_mean);

  cv::Mat I_var = boxFilter(I.mul(I), r) - I_mean.mul(I_mean);
  cv::Mat a = cov_Ip / (I_var + eps);  // 计算 a
  cv::Mat b = p_mean - a.mul(I_mean);  // 计算 b

  cv::Mat a_mean = boxFilter(a, r);
  cv::Mat b_mean = boxFilter(b, r);
  q = a_mean.mul(I) + b_mean;  // 计算输出
}

void customConvertTo(const cv::Mat &src, cv::Mat &dst, int dtype, int chnumber, double alpha = 1.0,
                     double beta = 0.0) {
  if (src.empty()) {
    std::cerr << "Input image is empty!" << std::endl;
    return;
  }

  dst.create(src.size(), dtype);
  for (int r = 0; r < src.rows; ++r) {
    for (int c = 0; c < src.cols; ++c) {
      if (chnumber == 3) {
        cv::Vec3b pixel = src.at<cv::Vec3b>(r, c);
        double value;
        for (int channel = 0; channel < 3; ++channel) {
          value = static_cast<double>(pixel[channel]) * alpha + beta;
          if (dtype == CV_32F) {
            dst.at<cv::Vec3f>(r, c)[channel] = static_cast<float>(value);
          } else if (dtype == CV_8U) {
            dst.at<cv::Vec3b>(r, c)[channel] =
                static_cast<uchar>(std::min(std::max(value, 0.0), 255.0));
          }
        }
      } else {
        uchar pixel = src.at<uchar>(r, c);
        double value;
        value = static_cast<double>(pixel) * alpha + beta;
        if (dtype == CV_32F) {
          dst.at<float>(r, c) = static_cast<float>(value);
        } else if (dtype == CV_8U) {
          dst.at<uchar>(r, c) = static_cast<uchar>(std::min(std::max(value, 0.0), 255.0));
        }
      }
    }
  }
}
float OcclusionClassification::Lapulasi2(cv::Mat cur_frame_gray, bool flag) {
  std::ofstream ofs2("/mnt/data/cur_frame_gray.bin", std::ios::binary);
  if (ofs2) {
    ofs2.write(reinterpret_cast<const char *>(cur_frame_gray.data),
               cur_frame_gray.elemSize() * cur_frame_gray.total());
    ofs2.close();
  }
  std::cout << "Type of cur_frame_gray image: " << cur_frame_gray.type() << std::endl;
  // 应用拉普拉斯算子
  cv::Mat _laplacian;
  cv::Laplacian(cur_frame_gray, _laplacian, CV_64F);

  cv::threshold(_laplacian, laplacianAbs, 0, 0, cv::THRESH_TRUNC);
  laplacianAbs = _laplacian - 2 * laplacianAbs;
  // cv::threshold(laplacianAbs, laplacianAbs, 255, 255, cv::THRESH_TOZERO);
  cv::Mat _laplacian_8u;
  cv::convertScaleAbs(laplacianAbs, _laplacian_8u);
  // customConvertTo(laplacianAbs,_laplacian_8u, CV_8U, 1);
  std::ofstream ofs0("/mnt/data/_laplacian_8u.bin", std::ios::binary);
  if (ofs0) {
    ofs0.write(reinterpret_cast<const char *>(_laplacian_8u.data),
               _laplacian_8u.elemSize() * _laplacian_8u.total());
    ofs0.close();
  }
  // 定义结构元素
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

  // 进行闭运算
  cv::Mat closed;
  cv::morphologyEx(_laplacian_8u, closed, cv::MORPH_CLOSE, kernel);
  std::cout << "Type of closed image: " << closed.type() << std::endl;
  cv::Mat binary_f;
  // cv::Mat binary;
  cv::Mat binary_not;
  cv::threshold(closed, binary_f, 10, 255, cv::THRESH_BINARY);
  // customConvertTo(binary_f,binary, CV_8U, 1);
  std::ofstream ofs("/mnt/data/binary.bin", std::ios::binary);
  if (ofs) {
    ofs.write(reinterpret_cast<const char *>(binary_f.data),
              binary_f.elemSize() * binary_f.total());
    ofs.close();
  }
  cv::bitwise_not(binary_f, binary_not);
  std::ofstream ofs1("/mnt/data/binary_not.bin", std::ios::binary);
  if (ofs1) {
    ofs1.write(reinterpret_cast<const char *>(binary_not.data),
               binary_not.elemSize() * binary_not.total());
    ofs1.close();
  }
  std::cout << "Type of binary image: " << binary_not.type() << std::endl;

  cv::Mat labels;
  int num_labels = cv::connectedComponents(binary_not, labels, 4);

  // 计算每个连通区域的像素数
  std::vector<int> sizes(num_labels, 0);
  for (int i = 0; i < labels.rows; i++) {
    for (int j = 0; j < labels.cols; j++) {
      sizes[labels.at<int>(i, j)]++;
    }
  }
  int max_connected_area = (num_labels > 1) ? *std::max_element(sizes.begin() + 1, sizes.end()) : 0;
  int total_pixels = labels.rows * labels.cols;
  std::cout << "Max connected area (excluding background): " << max_connected_area << std::endl;
  float occ_ratio = static_cast<float>(max_connected_area) / total_pixels;
  return occ_ratio;
}

OcclusionClassification::OcclusionClassification() : Core(CVI_MEM_DEVICE) {
  _crop_bbox.x1 = 0;
  _crop_bbox.y1 = 0;
  _crop_bbox.x2 = 1;
  _crop_bbox.y2 = 1;
  for (uint32_t i = 0; i < 3; i++) {
    m_preprocess_param[0].factor[i] = OCCULUSION_CLASSIFICATION_FACTOR;
    m_preprocess_param[0].mean[i] = OCCULUSION_CLASSIFICATION_MEAN;
  }
  m_preprocess_param[0].format = PIXEL_FORMAT_RGB_888_PLANAR;
  m_preprocess_param[0].keep_aspect_ratio = false;
  m_preprocess_param[0].use_crop = true;
#ifndef __CV186X__
  m_preprocess_param[0].resize_method = VPSS_SCALE_COEF_OPENCV_BILINEAR;
#endif
}

OcclusionClassification::~OcclusionClassification() {}

void OcclusionClassification::set_algparam(OcclusionAlgParam occ_pre_param) {
  auto_lap_dev_th = occ_pre_param.lap_dev_th;
  _ai_cv_th = occ_pre_param.ai_cv_th;
  _crop_bbox = occ_pre_param.crop_bbox;
}

int OcclusionClassification::cv_method(VIDEO_FRAME_INFO_S *frame,
                                       cvtdl_class_meta_t *occlusion_classification_meta,
                                       float ai_occ_rato) {
  frame->stVFrame.pu8VirAddr[0] =
      (CVI_U8 *)CVI_SYS_MmapCache(frame->stVFrame.u64PhyAddr[0], frame->stVFrame.u32Length[0]);

  cv::Mat cur_frame(frame->stVFrame.u32Height, frame->stVFrame.u32Width, CV_8UC3,
                    frame->stVFrame.pu8VirAddr[0], frame->stVFrame.u32Stride[0]);
  int frame_h = cur_frame.rows;
  int frame_w = cur_frame.cols;
  cv::Rect roi(int(_crop_bbox.x1 * frame_w), int(_crop_bbox.y1 * frame_h),
               int((_crop_bbox.x2 - _crop_bbox.x1) * frame_w),
               int((_crop_bbox.y2 - _crop_bbox.y1) * frame_h));
  cv::Mat sub_frame = cur_frame(roi);

  cv::Mat cur_frame_gray;

#ifdef ENABLE_CVIAI_CV_UTILS
  cviai::cvtColor(sub_frame, cur_frame_gray, COLOR_BGR2GRAY);
  std::cout << "cviai::cvtColor: " << std::endl;
#else
  cv::cvtColor(sub_frame, cur_frame_gray, cv::COLOR_BGR2GRAY);
#endif

  float la_res = Lapulasi2(cur_frame_gray, keyframe_flag);
  std::cout << "la_res: " << la_res << std::endl;
  occlusion_classification_meta->score[1] = la_res;

  if (la_res < 0.4) {
    occlusion_classification_meta->cls[0] = 0;
  } else {
    occlusion_classification_meta->cls[0] = 1;
  }

  CVI_SYS_Munmap((void *)frame->stVFrame.pu8VirAddr[0], frame->stVFrame.u32Length[0]);
  return CVI_SUCCESS;
}

int OcclusionClassification::inference(VIDEO_FRAME_INFO_S *frame,
                                       cvtdl_class_meta_t *occlusion_classification_meta) {
  int frame_h = frame->stVFrame.u32Height;
  int frame_w = frame->stVFrame.u32Width;

  // m_vpss_config[0].crop_attr.enCropCoordinate = VPSS_CROP_RATIO_COOR;
  m_vpss_config[0].crop_attr.stCropRect = {int(_crop_bbox.x1 * frame_w),
                                           int(_crop_bbox.y1 * frame_h),
                                           uint32_t((_crop_bbox.x2 - _crop_bbox.x1) * frame_w),
                                           uint32_t((_crop_bbox.y2 - _crop_bbox.y1) * frame_h)};

  std::vector<VIDEO_FRAME_INFO_S *> frames = {frame};
  int ret = run(frames);
  if (ret != CVI_SUCCESS) {
    return ret;
  }

  float *score = getOutputRawPtr<float>(0);
  double a = std::exp(score[0]);
  double b = std::exp(score[1]);

  score[0] = a / (a + b);
  score[1] = b / (a + b);
  std::cout << "this frame occ ratio is:" << score[1] << std::endl;
  occlusion_classification_meta->cls[0] = score[1] > score[0];
  occlusion_classification_meta->score[0] = score[1];

  ret = cv_method(frame, occlusion_classification_meta, score[1]);

  return ret;
}

}  // namespace cvitdl