#ifndef YOLOV5_DETECTOR_PYTHON_HPP__
#define YOLOV5_DETECTOR_PYTHON_HPP__
#include "factory/model.hpp"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class NNYOLOV5Detector {
public:
  explicit NNYOLOV5Detector(const NNFactory *factory, int device_id = 0);
  ~NNYOLOV5Detector();
  py::tuple predict(py::list &inputs);
  py::tuple predict(py::array_t<unsigned char, py::array::c_style> &input);

private:
  YOLOV5 *detector_ = nullptr;
};

#endif
