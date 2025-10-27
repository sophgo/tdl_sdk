#include "utils/pose_helper.hpp"
#include <cstdio>

#ifdef __EDGE_PLATFORM__
#include <opencv2/core/core_c.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#else

#include <opencv2/core/version.hpp>

#if CV_VERSION_MAJOR >= 4 && CV_VERSION_MINOR >= 5
#include "opencv2/core/core_c.h"
#endif
#include <opencv2/opencv.hpp>

#endif

#ifndef PI
#define PI 3.14159265358979323846
#endif

static std::vector<cv::Point3f> modelPts = {
    {-30.f, 40.f, 30.f},   // 左眼中心 (示例)
    {+30.f, 40.f, 30.f},   // 右眼中心
    {0.f, 10.f, 30.f},     // 鼻尖
    {-25.f, -40.f, 30.f},  // 左嘴角
    {+25.f, -40.f, 30.f}   // 右嘴角
};

// x1, x2, x3, x4, x5, y1, y2, y3, y4, y5
// (x1, y1) = Left eye center
// (x2, y2) = Right eye center
// (x3, y3) = Nose tip
// (x4, y4) = Left Mouth corner
// (x5, y5) = Right mouth corner

float CalDistance(const cv::Point& p1, const cv::Point& p2) {
  float x = p1.x - p2.x;
  float y = p1.y - p2.y;
  return sqrtf(x * x + y * y);
}

float CalSlant(int ln, int lf, const float Rn, float theta) {
  float dz = 0;
  float slant = 0;
  const float m1 = ((float)ln * ln) / ((float)lf * lf);
  const float m2 = (cos(theta)) * (cos(theta));
  const float Rn_sq = Rn * Rn;

  if (m2 == 1) {
    dz = sqrt(Rn_sq / (m1 + Rn_sq));
  }
  if (m2 >= 0 && m2 < 1) {
    dz = sqrt((Rn_sq - m1 - 2 * m2 * Rn_sq +
               sqrt(((m1 - Rn_sq) * (m1 - Rn_sq)) + 4 * m1 * m2 * Rn_sq)) /
              (2 * (1 - m2) * Rn_sq));
  }
  slant = acos(dz);
  return slant;
}

template <typename T>
T Saturate(const T& val, const T& minVal, const T& maxVal) {
  return std::min(std::max(val, minVal), maxVal);
}

float CalAngle(const cv::Point& pt1, const cv::Point& pt2) {
  return 360 - cvFastArctan(pt2.y - pt1.y, pt2.x - pt1.x);
}

int Predict3DFacialNormal(const cv::Point& noseTip, const cv::Point& noseBase,
                          const cv::Point& midEye, const cv::Point& midMouth,
                          FacePose* hp) {
  float noseBase_noseTip_distance = CalDistance(noseTip, noseBase);
  float midEye_midMouth_distance = CalDistance(midEye, midMouth);

  // Angle facial middle (symmetric) line.
  float symm = CalAngle(noseBase, midEye);

  // Angle between 2D image facial normal & x-axis.
  float tilt = CalAngle(noseBase, noseTip);

  // Angle between 2D image facial normal & facial middle (symmetric) line.
  float theta = (std::abs(tilt - symm)) * (PI / 180.0);

  // Angle between 3D image facial normal & image plain normal (optical axis).
  float slant =
      CalSlant(noseBase_noseTip_distance, midEye_midMouth_distance, 0.5, theta);

  // Define a 3D vector for the facial normal
  hp->facialUnitNormalVector[0] =
      sin(slant) * (cos((360 - tilt) * (PI / 180.0)));
  hp->facialUnitNormalVector[1] =
      sin(slant) * (sin((360 - tilt) * (PI / 180.0)));
  hp->facialUnitNormalVector[2] = -cos(slant);

  return 0;
}

int32_t PoseHelper::predictFacePose(const std::vector<float>& landmark_5x,
                                    const std::vector<float>& landmark_5y,
                                    FacePose* hp) {
  cv::Point leye = cv::Point(landmark_5x[0], landmark_5y[0]);
  cv::Point reye = cv::Point(landmark_5x[1], landmark_5y[1]);
  cv::Point lmouth = cv::Point(landmark_5x[3], landmark_5y[3]);
  cv::Point rmouth = cv::Point(landmark_5x[4], landmark_5y[4]);
  cv::Point noseTip = cv::Point(landmark_5x[2], landmark_5y[2]);

  cv::Point midEye =
      cv::Point((leye.x + reye.x) * 0.5, (leye.y + reye.y) * 0.5);
  cv::Point midMouth =
      cv::Point((lmouth.x + rmouth.x) * 0.5, (lmouth.y + rmouth.y) * 0.5);
  cv::Point noseBase =
      cv::Point((midMouth.x + midEye.x) * 0.5, (midMouth.y + midEye.y) * 0.5);

  Predict3DFacialNormal(noseTip, noseBase, midEye, midMouth, hp);

  hp->yaw =
      acos((std::abs(hp->facialUnitNormalVector[2])) /
           (std::sqrt(
               hp->facialUnitNormalVector[0] * hp->facialUnitNormalVector[0] +
               hp->facialUnitNormalVector[2] * hp->facialUnitNormalVector[2])));
  if (noseTip.x < noseBase.x) hp->yaw = -hp->yaw;
  hp->yaw = Saturate(hp->yaw, -1.f, 1.f);

  hp->pitch = acos(std::sqrt(
      (hp->facialUnitNormalVector[0] * hp->facialUnitNormalVector[0] +
       hp->facialUnitNormalVector[2] * hp->facialUnitNormalVector[2]) /
      (hp->facialUnitNormalVector[0] * hp->facialUnitNormalVector[0] +
       hp->facialUnitNormalVector[1] * hp->facialUnitNormalVector[1] +
       hp->facialUnitNormalVector[2] * hp->facialUnitNormalVector[2])));
  if (noseTip.y > noseBase.y) hp->pitch = -hp->pitch;
  hp->pitch = Saturate(hp->pitch, -1.f, 1.f);

  hp->roll = CalAngle(leye, reye);
  if (hp->roll > 180) hp->roll = hp->roll - 360;
  hp->roll /= 90;
  hp->roll = Saturate(hp->roll, -1.f, 1.f);

  return 0;
}

int32_t PoseHelper::predictFacePose(const std::vector<float>& landmark_5x,
                                    const std::vector<float>& landmark_5y,
                                    const int img_width, const int img_height,
                                    FacePose* hp) {
#ifdef __EDGE_PLATFORM__

  std::vector<cv::Point2f> imagePts5 = {{landmark_5x[0], landmark_5y[0]},
                                        {landmark_5x[1], landmark_5y[1]},
                                        {landmark_5x[2], landmark_5y[2]},
                                        {landmark_5x[3], landmark_5y[3]},
                                        {landmark_5x[4], landmark_5y[4]}};

  double fx = 1.2 * std::max(img_width, img_height);
  double fy = fx;
  double cx = img_width * 0.5;
  double cy = img_height * 0.5;
  cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
  cv::Mat dist = cv::Mat::zeros(1, 5, CV_64F);  // 假设无畸变

  // 3. solvePnP 解算姿态
  cv::Mat rvec, tvec;
  if (!cv::solvePnP(modelPts, imagePts5, K, dist, rvec, tvec,
                    /*useExtrinsicGuess=*/false, cv::SOLVEPNP_ITERATIVE)) {
    return -1;
  }

  // 4. Rodrigues -> 3×3
  cv::Mat R_cv;
  cv::Rodrigues(rvec, R_cv);

  // 5. Extract Euler angles (ZYX 顺序: yaw–pitch–roll)
  double sy = std::sqrt(R_cv.at<double>(0, 0) * R_cv.at<double>(0, 0) +
                        R_cv.at<double>(1, 0) * R_cv.at<double>(1, 0));
  bool singular = sy < 1e-6;
  double x, y, z;
  if (!singular) {
    x = std::atan2(R_cv.at<double>(2, 1), R_cv.at<double>(2, 2));  // roll
    y = std::atan2(-R_cv.at<double>(2, 0), sy);                    // pitch
    z = std::atan2(R_cv.at<double>(1, 0), R_cv.at<double>(0, 0));  // yaw
  } else {
    x = std::atan2(-R_cv.at<double>(1, 2), R_cv.at<double>(1, 1));
    y = std::atan2(-R_cv.at<double>(2, 0), sy);
    z = 0;
  }

  constexpr double RAD2DEG = 180.0 / CV_PI;
  hp->roll = x * RAD2DEG;
  hp->pitch = y * RAD2DEG;
  hp->yaw = z * RAD2DEG;

  return 0;
#else
  return -1;
#endif
}
