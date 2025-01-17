#ifndef FACE_COMMON_HPP_
#define FACE_COMMON_HPP_

#include <opencv2/opencv.hpp>
#include <vector>

typedef struct FacePts {
  FacePts() { score = 0; }

  std::vector<float> x;
  std::vector<float> y;
  float score;

  bool valid() { return x.size() == y.size() && !x.empty() && !y.empty(); }
} FacePts;

typedef struct FacePose {
  FacePose() {
    pitch = 0;
    yaw = 0;
    roll = 0;
  }

  float pitch;
  float yaw;
  float roll;
  float facialUnitNormalVector[3];
} FacePose;

/* Store the information of face bounding box
 * x1: left-top     point X-axis
 * y1: left-top     point Y-axis
 * x2: right-bottom point X-axis
 * y2: right-bottom point Y-axis
 * score: bbox prediction score
 * */
typedef struct FaceRect {
  FaceRect() {
    score = 0.;
    x1 = 0.;
    y1 = 0.;
    x2 = 0.;
    y2 = 0.;
  }

  float x1;
  float y1;
  float x2;
  float y2;
  float temp_x1;
  float temp_y1;
  float temp_x2;
  float temp_y2;
  float head_score;
  float score;
  int label; // for mask face: 1 with mask,0 no mask
  FacePts facepts;
  FacePose facepose;

  void print() {
    std::cout << "bbox:" << std::endl;
    std::cout << "x1 " << x1 << ",y1 " << y1 << ",x2 " << x2 << ",y2 " << y2
              << std::endl;
    std::cout << "point:" << std::endl;
    for (int i = 0; i < facepts.x.size(); i++) {
      std::cout << "x " << facepts.x[i] << " y " << facepts.y[i] << std::endl;
    }
  }

  cv::Vec4f regression; // only used by mtcnn
} FaceRect;

/*
 * Image transform parammeters
 * for pre-process function
 * */
struct IMGTransParam {
  int src_w = 0;
  int src_h = 0;
  float fx = 0.;
  float fy = 0.;
  int ox = 0;
  int oy = 0;

  // resize factor x,y; offset x,y
  IMGTransParam(){};

  IMGTransParam(int src_w, int src_h, float fx, float fy, int ox, int oy)
      : src_w(src_w), src_h(src_h), fx(fx), fy(fy), ox(ox), oy(oy) {}
};

#endif
