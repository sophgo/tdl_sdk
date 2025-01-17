#include "common/cv_utils.hpp"
#include "face/face_util.hpp"
#include <log/Logger.hpp>

#define INTER_RESIZE_COEF_BITS (11)
#define INTER_RESIZE_COEF_SCALE (1 << INTER_RESIZE_COEF_BITS)
#define MAX_ESIZE 16
#define PI 3.1416
#define FACE_AREA_STANDARD (112 * 112)
#define EYE_DISTANCE_STANDARD 80.
#define ABS(x) ((x) >= 0 ? (x) : (-(x)))
template <typename T>
T Saturate(const T& val, const T& minVal, const T& maxVal) {
  return std::min(std::max(val, minVal), maxVal);
}

void bgr_split_scale(const cv::Mat &src_mat, std::vector<cv::Mat> &tmp_bgr,
                     const std::vector<cv::Mat> &input_channels,
                     float input_scale) {

  // int w = src_mat.cols;
  // int h = src_mat.rows;
  // std::cout << "split begin" << std::endl;
  cv::split(src_mat, tmp_bgr); // cv::split is faster than vpp
  // std::cout << "split over" << std::endl;
  LOG(INFO) << tmp_bgr.size();
  for (int i = 0; i < tmp_bgr.size(); i++) {
    // std::cout << tmp_bgr[i].type() << " " << input_channels[i].type() <<
    // std::endl;
    tmp_bgr[i].convertTo(input_channels[i], input_channels[i].type(),
                         input_scale);
  }
  // LOG(INFO) << (int)*(input_channels[0].data + 14802);
  // std::cout << "convert bgr to channel" << std::endl;
}

void bgr_split_scale1(const cv::Mat &src_mat, std::vector<cv::Mat> &tmp_bgr,
                      const std::vector<cv::Mat> &input_channels,
                      const std::vector<float> &mean,
                      const std::vector<float> &scale,
                      bool use_rgb /*= false*/) {
  // cv::split is faster than vpp
  cv::split(src_mat, tmp_bgr);
  if (use_rgb) {
    LOG(INFO) << "use RGB,img size:" << src_mat.size();
    std::swap(tmp_bgr[0], tmp_bgr[2]);
  }
  for (int i = 0; i < tmp_bgr.size(); i++) {
    float m = 0, s = 1;
    if (mean.size() > i)
      m = mean[i];
    if (scale.size() > i)
      s = scale[i];
    tmp_bgr[i].convertTo(input_channels[i], input_channels[i].type(), s, -m);
  }
}


cv::Mat
pad_resize_img(const cv::Mat &src_img, const cv::Size dst_size,
               std::vector<float> &rescale_params,
               cv::InterpolationFlags inter_flag /*= cv::INTER_NEAREST*/) {
  bool need_pad_resize =
      compute_pad_resize_param(src_img.size(), dst_size, rescale_params);
  cv::Mat dst_img;
  if (!need_pad_resize) {
    //#ifdef BUILD_ARMV8
    //    cv::vpp::resize(src_img, dst_img, dst_size, 0, 0, inter_flag);
    //#else
    cv::resize(src_img, dst_img, dst_size, 0, 0, inter_flag);
    //#endif
  } else {
    int src_resized_w = lrint(src_img.cols / rescale_params[0]);
    int src_resized_h = lrint(src_img.rows / rescale_params[1]);
    dst_img = cv::Mat::zeros(dst_size, src_img.type());
    cv::Rect roi(rescale_params[2], rescale_params[3], src_resized_w,
                 src_resized_h);
    cv::Mat resized_img;
    LOG(INFO) << "opencv padresize,size:" << src_resized_w << ","
              << src_resized_h << ",offset:" << roi.x << "," << roi.y;
    //#ifdef BUILD_ARMV8
    //   cv::vpp::resize(src_img,
    //   resized_img,cv::Size(src_resized_w,src_resized_h), 0, 0, inter_flag);
    //#else
    cv::resize(src_img, dst_img(roi), cv::Size(src_resized_w, src_resized_h), 0,
               0, inter_flag);
    //#endif
    // resized_img.copyTo(dst_img(roi));
  }
  return dst_img;
}
void pad_resize_to_dst(
    const cv::Mat &src_img, cv::Mat &dst_img,
    std::vector<float> &rescale_params,
    cv::InterpolationFlags inter_flag /*= cv::INTER_NEAREST*/) {
  bool need_pad_resize =
      compute_pad_resize_param(src_img.size(), dst_img.size(), rescale_params);

  if (!need_pad_resize) {
    cv::resize(src_img, dst_img, dst_img.size(), 0, 0, inter_flag);
  } else {
    int src_resized_w = lrint(src_img.cols / rescale_params[0]);
    int src_resized_h = lrint(src_img.rows / rescale_params[1]);
    cv::Rect roi(rescale_params[2], rescale_params[3], src_resized_w,
                 src_resized_h);
    cv::Mat resized_img;
    LOG(INFO) << "opencv padresize,size:" << src_resized_w << ","
              << src_resized_h << ",offset:" << roi.x << "," << roi.y;
    cv::resize(src_img, dst_img(roi), cv::Size(src_resized_w, src_resized_h), 0,
               0, inter_flag);
  }
}
bool compute_pad_resize_param(cv::Size src_size, cv::Size dst_size,
                              std::vector<float> &rescale_params) {
  rescale_params.clear();
  float src_w = src_size.width;
  float src_h = src_size.height;
  float ratio_w = src_w / dst_size.width;
  float ratio_h = src_h / dst_size.height;
  float ratio = std::max(ratio_w, ratio_h);
  // LOG(INFO)<<src_size<<"->"<<dst_size<<",ratio:"<<ratio_w<<","<<ratio_h;
  rescale_params.push_back(ratio);
  rescale_params.push_back(ratio);
  cv::Mat dst_img;
  if (ratio_w != ratio_h) {
    int src_resized_w = lrint(src_w / ratio);
    int src_resized_h = lrint(src_h / ratio);
    int roi_x = (dst_size.width - src_resized_w + 1) / 2;
    int roi_y = (dst_size.height - src_resized_h + 1) / 2;
    // LOG(INFO)<<"scale:"<<ratio<<",src_size:"<<src_resized_w<<","<<src_resized_h<<",roi_xy:"<<roi_x<<","<<roi_y;
    rescale_params.push_back(roi_x);
    rescale_params.push_back(roi_y);
    return true;
  } else {

    rescale_params.push_back(0.0f);
    rescale_params.push_back(0.0f);
    return false;
  }
}

void vis_face_rects(cv::Mat &src_img, std::vector<FaceRect> &rects) {

  for (int i = 0; i < rects.size(); i++) {
    FaceRect fr = rects[i];
    cv::rectangle(src_img, cv::Point2f(fr.x1, fr.y1), cv::Point2f(fr.x2, fr.y2),
                  cv::Scalar(0, 0, 255), 1);
    for (int j = 0; j < fr.facepts.x.size(); j++) {
      cv::circle(src_img, cv::Point(fr.facepts.x[j], fr.facepts.y[j]), 1,
                 cv::Scalar(0, 255, 0));
    }
  }
}

std::vector<cv::Mat> wrap_img_channels(char *p_src_img, int width, int height,
                                       int c, int batch, int img_type) {
  std::vector<cv::Mat> channel_imgs;
  int sz = CV_ELEM_SIZE(img_type);
  char *p_img = p_src_img + width * height * c * batch * sz;
  for (int i = 0; i < c; i++) {
    cv::Mat channel(height, width, img_type, p_img);
    assert(channel.cols * sz == channel.step[0]);
    channel_imgs.push_back(channel);
    p_img += width * height * sz;
  }
  return channel_imgs;
}

void align_frame_faces_cpu(std::vector<cv::Mat> &frame_imgs,
                           std::vector<std::vector<FacePts>> &frame_landmarks,
                           std::vector<cv::Mat> &aligned_faces) {
  assert(frame_imgs.size() == frame_landmarks.size());
  cv::Size dst_size(112, 112);
  for (size_t i = 0; i < frame_imgs.size(); i++) {
    std::vector<FacePts> &cur_frame_lds = frame_landmarks[i];
    for (auto &pt : cur_frame_lds) {
      cv::Mat trans = calc_transform_matrix(pt, dst_size);
      cv::Mat aligned;
      cv::warpAffine(frame_imgs[i], aligned, trans, dst_size);
      aligned_faces.push_back(aligned);
    }
  }
}
float cal_iou(const cv::Rect &box1,const cv::Rect &box2){
  float xx1 = std::max(box1.x, box2.x);
  float yy1 = std::max(box1.y, box2.y);
  float xx2 = std::min(box1.br().x, box2.br().x);
  float yy2 = std::min(box1.br().y, box2.br().y);
  float w = std::max(float(0), xx2 - xx1 + 1);
  float h = std::max(float(0), yy2 - yy1 + 1);
  float inter = w * h;
  float iou = inter / (box1.area()+box2.area() - inter);
  return iou;
}
float box_in_ratio(const cv::Rect &box1,const cv::Rect &box2){
  float xx1 = std::max(box1.x, box2.x);
  float yy1 = std::max(box1.y, box2.y);
  float xx2 = std::min(box1.br().x, box2.br().x);
  float yy2 = std::min(box1.br().y, box2.br().y);
  int w = std::max(float(0), xx2 - xx1 + 1);
  int h = std::max(float(0), yy2 - yy1 + 1);
  float in_ratio = w*h/box1.area();
  return in_ratio;
}
//TODO: use eigen
void normalize_feature(std::vector<float> &feature){
  float sum = 0;
  for(int i = 0; i < feature.size();i++){
    sum += feature[i]*feature[i];
  }
  float sqrt_sum = sqrt(sum);
    for(int i = 0; i < feature.size();i++){
    feature[i] = feature[i]/sqrt_sum;
  }
  
  // Eigen::Map<Eigen::Matrix<float, 1, feature.size(), Eigen::RowMajor>> norm_feature(
  //   &feature[0]);
  // norm_feature.normalize();
}

void align_carplate(const stObjPts &landmarks,
                    const cv::Mat &src_mat,cv::Mat &dst_mat){
  float s=0.03;
  float rand_offset[] = {s,s,1-s,s,1-s,1-s,s,1-s};
  
  std::vector<cv::Point2f>src_pts, dst_pts;
  cv::Size dst_size = dst_mat.size();
  for(int i = 0; i < 4;i++){
    src_pts.push_back(cv::Point2f(landmarks.x[i],landmarks.y[i]));
  }
  dst_pts.push_back(cv::Point2f(dst_size.width*rand_offset[0], dst_size.height*rand_offset[1]));
  dst_pts.push_back(cv::Point2f(dst_size.width*rand_offset[2], dst_size.height*rand_offset[3]));
  dst_pts.push_back(cv::Point2f(dst_size.width*rand_offset[4], dst_size.height*rand_offset[5]));
  dst_pts.push_back(cv::Point2f(dst_size.width*rand_offset[6], dst_size.height*rand_offset[7]));
  cv::Mat trans = cv::getPerspectiveTransform(src_pts,dst_pts);
  cv::warpPerspective(src_mat,dst_mat,trans,dst_mat.size());
}

void get_score(FaceRect &detector_info){
  FaceRect &bbox = detector_info;
  FacePose pose;
  FacePts &pts_info = detector_info.facepts;
  GetHeadPose(pts_info,pose);
  float nose_x = pts_info.x[2];
  float left_max = MIN(pts_info.x[0],pts_info.x[3]);
  float right_max = MAX(pts_info.x[1],pts_info.x[4]);
  float width = bbox.x2 - bbox.x1;
  float height = bbox.y2 - bbox.y1;
  float l_ = nose_x - left_max;
  float r_ = right_max - nose_x;
  float eye_diff_x = pts_info.x[1] - pts_info.x[0];
  float eye_diff_y = pts_info.y[1] - pts_info.y[0];
  float eye_size = sqrt(eye_diff_x * eye_diff_x + eye_diff_y * eye_diff_y);
  float mouth_diff_x = pts_info.x[4] - pts_info.x[3];
  float mouth_diff_y = pts_info.y[4] - pts_info.y[3];
  float mouth_size = sqrt(mouth_diff_x * mouth_diff_x + mouth_diff_y * mouth_diff_y);
  float ydiff = pts_info.y[1] - bbox.y1;
  if(ydiff<height*0.15){
    detector_info.head_score = 0.0;
    return ;
  }
  // 如果特征点在候选框
  if(pts_info.x[1] > bbox.x2 || pts_info.x[2] > bbox.x2 || pts_info.x[4] > bbox.x2 || \
  pts_info.x[0] < bbox.x1 || pts_info.x[2] < bbox.x1 || pts_info.x[3] < bbox.x1){
    detector_info.head_score = 0.0;
  }
  // 鼻子在最左边或最右边,或者眼睛和嘴巴的宽度太小
  else if((l_ + 0.01 * width) < 0 || (r_ + 0.01 * width) < 0 || (eye_size / width)< 0.25|| (mouth_size / width) < 0.15){
    detector_info.head_score = 0.0;
  }else{
    float face_area = (bbox.y2 - bbox.y1) * (bbox.x2 - bbox.x1);
    float area_score = MIN(1.0, face_area / FACE_AREA_STANDARD);
    float pose_score = 1. - (ABS(pose.yaw) + ABS(pose.pitch) + ABS(pose.roll)) / 3.;
    float size_score = eye_size / (bbox.x2 - bbox.x1);
    size_score += mouth_size / (bbox.x2 - bbox.x1);
    detector_info.head_score = pose_score * 0.9 + 0.3 * size_score;
  }
}

void GetHeadPose(const FacePts &pFacial5points, FacePose &hp) {
  cv::Point leye = cv::Point(pFacial5points.x[0], pFacial5points.y[0]);
  cv::Point reye = cv::Point(pFacial5points.x[1], pFacial5points.y[1]);
  cv::Point lmouth = cv::Point(pFacial5points.x[3], pFacial5points.y[3]);
  cv::Point rmouth = cv::Point(pFacial5points.x[4], pFacial5points.y[4]);
  cv::Point noseTip = cv::Point(pFacial5points.x[2], pFacial5points.y[2]);
  cv::Point midEye = cv::Point((leye.x + reye.x) * 0.5, (leye.y + reye.y) * 0.5);
  cv::Point midMouth = cv::Point((lmouth.x + rmouth.x) * 0.5, (lmouth.y + rmouth.y) * 0.5);
  cv::Point noseBase = cv::Point((midMouth.x + midEye.x) * 0.5, (midMouth.y + midEye.y) * 0.5);

  Predict3DFacialNormal(noseTip, noseBase, midEye, midMouth, hp);

  hp.yaw = acos((std::abs(hp.facialUnitNormalVector[2])) /
                 (std::sqrt(hp.facialUnitNormalVector[0] * hp.facialUnitNormalVector[0] +
                            hp.facialUnitNormalVector[2] * hp.facialUnitNormalVector[2])));
  if (noseTip.x < noseBase.x) hp.yaw = -hp.yaw;
  hp.yaw = Saturate(hp.yaw, -1.f, 1.f);

  hp.pitch = acos(std::sqrt((hp.facialUnitNormalVector[0] * hp.facialUnitNormalVector[0] +
                              hp.facialUnitNormalVector[2] * hp.facialUnitNormalVector[2]) /
                             (hp.facialUnitNormalVector[0] * hp.facialUnitNormalVector[0] +
                              hp.facialUnitNormalVector[1] * hp.facialUnitNormalVector[1] +
                              hp.facialUnitNormalVector[2] * hp.facialUnitNormalVector[2])));
  if (noseTip.y > noseBase.y) hp.pitch = -hp.pitch;
  hp.pitch = Saturate(hp.pitch, -1.f, 1.f);

  hp.roll = CalAngle(leye, reye);
  if (hp.roll > 180) hp.roll = hp.roll - 360;
  hp.roll /= 90;
  hp.roll = Saturate(hp.roll, -1.f, 1.f);

}

void Predict3DFacialNormal(const cv::Point& noseTip, const cv::Point& noseBase,
                          const cv::Point& midEye, const cv::Point& midMouth,
                          FacePose &hp) {
  float noseBase_noseTip_distance = CalDistance(noseTip, noseBase);
  float midEye_midMouth_distance = CalDistance(midEye, midMouth);

  // Angle facial middle (symmetric) line.
  float symm = CalAngle(noseBase, midEye);

  // Angle between 2D image facial normal & x-axis.
  float tilt = CalAngle(noseBase, noseTip);

  // Angle between 2D image facial normal & facial middle (symmetric) line.
  float theta = (std::abs(tilt - symm)) * (PI / 180.0);

  // Angle between 3D image facial normal & image plain normal (optical axis).
  float slant = CalSlant(noseBase_noseTip_distance, midEye_midMouth_distance, 0.5, theta);

  // Define a 3D vector for the facial normal
  hp.facialUnitNormalVector[0] = sin(slant) * (cos((360 - tilt) * (PI / 180.0)));
  hp.facialUnitNormalVector[1] = sin(slant) * (sin((360 - tilt) * (PI / 180.0)));
  hp.facialUnitNormalVector[2] = -cos(slant);

}

float CalDistance(const cv::Point& p1, const cv::Point& p2) {
  float x = p1.x - p2.x;
  float y = p1.y - p2.y;
  return sqrtf(x * x + y * y);
}

float CalAngle(const cv::Point& pt1, const cv::Point& pt2) {
  return 360 - cvFastArctan(pt2.y - pt1.y, pt2.x - pt1.x);
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
    dz = sqrt(
        (Rn_sq - m1 - 2 * m2 * Rn_sq + sqrt(((m1 - Rn_sq) * (m1 - Rn_sq)) + 4 * m1 * m2 * Rn_sq)) /
        (2 * (1 - m2) * Rn_sq));
  }
  slant = acos(dz);
  return slant;
}