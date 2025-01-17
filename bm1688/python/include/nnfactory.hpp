#ifndef NNFACTORY_PYTHON_HPP__
#define NNFACTORY_PYTHON_HPP__
#include "face/face_util.hpp"
#include "face_detector.hpp"
#include "face_extractor.hpp"
#include "yolov5_detector.hpp"

class NNF {
 public:
  explicit NNF(const std::string& model_dir, int device_id = -1);

  ~NNF();

  NNFaceDetector* get_face_detector(const std::string& detector_type = "cssd");

  // model_name:BMFACEV03M,BMFACER18,BMFACER34
  NNFaceExtractor* get_face_extractor(const std::string& model_name = "BMFACEV03M");


  NNYOLOV5Detector* get_vehicle_detector();

  py::array_t<unsigned char, py::array::c_style> align(
      py::array_t<unsigned char, py::array::c_style> input, py::list bbox, py::list point);

 private:
  NNFactory* factory_ = nullptr;
  int device_id_;
};

#endif
