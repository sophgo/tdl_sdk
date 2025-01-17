#ifndef FACE_DETECTOR_PYTHON_HPP__
#define FACE_DETECTOR_PYTHON_HPP__
#include "face/face_detect_landmark.hpp"
#include "factory/model.hpp"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class NNFaceDetector {
 public:
  explicit NNFaceDetector(const NNFactory *factory, int device_id = 0,std::string detector_type="scrfd");
  ~NNFaceDetector();
  py::tuple predict(py::list &inputs, int detect_landmark = 1);
  py::tuple predict(py::array_t<unsigned char, py::array::c_style> &input,
                    int detect_landmark = 1);
  py::tuple predict_mask(py::list &inputs, int detect_landmark = 1);
  py::tuple predict_mask(py::array_t<unsigned char, py::array::c_style> &input,
                         int detect_landmark = 1);

  void predict_face_landmark(std::vector<cv::Mat> &images, float ld_threshold,
                             std::vector<std::vector<FaceRect>> &img_landmarks);

 private:
  FaceCSSD *detector_ = nullptr;
  FaceSCRFD *detector_scrfd_ = nullptr;
  FaceLandmark *landmark_ = nullptr;
  // FaceDetectLandmark *fdl_ = nullptr;
};
class NNFaceDetMark {
 public:
  explicit NNFaceDetMark(const NNFactory* factory, ModelType m_detector, ModelType m_landmark,int device_id=0);

  ~NNFaceDetMark();

  // py::tuple predict(py::list& inputs, py::float_& detection_thresold, py::float_& point_threshold);

  py::tuple predict(py::list& inputs);

  py::tuple predict(py::array_t<unsigned char, py::array::c_style>& input);

  void forward(std::vector<cv::Mat>& images,
               float detection_thresold,
               float point_threshold,
               std::vector<std::vector<FaceRect>>& faceRects);

 private:
  BaseModel* p_detector = nullptr;
  BaseModel* p_landmark = nullptr;
  FaceDetectLandmark* p_fdl = nullptr;
  ModelType detector_type;
  ModelType landmark_type;
};
#endif
