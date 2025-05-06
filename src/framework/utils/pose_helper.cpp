#include "utils/pose_helper.hpp"

#ifdef __EDGE_PLATFORM__
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

static std::vector<cv::Point3f> modelPts = {
    {-30.f, 40.f, 30.f},   // 左眼中心 (示例)
    {+30.f, 40.f, 30.f},   // 右眼中心
    {0.f, 10.f, 30.f},     // 鼻尖
    {-25.f, -40.f, 30.f},  // 左嘴角
    {+25.f, -40.f, 30.f}   // 右嘴角
};

#endif

// x1, x2, x3, x4, x5, y1, y2, y3, y4, y5
// (x1, y1) = Left eye center
// (x2, y2) = Right eye center
// (x3, y3) = Nose tip
// (x4, y4) = Left Mouth corner
// (x5, y5) = Right mouth corner
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
