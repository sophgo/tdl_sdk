#ifndef HEAD_POSE_PYTHON_HPP__
#define HEAD_POSE_PYTHON_HPP__
#include "factory/model.hpp"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class NNHeadPose {
public:
  explicit NNHeadPose(const NNFactory *factory, int device_id = 0);
  ~NNHeadPose();
  py::list predict(py::list &inputs, py::list &bboxes_list);
  py::list predict(py::array_t<unsigned char, py::array::c_style> &input,
                   py::list &bboxes);

private:
  FaceDDFA *headpose_ = nullptr;
};

#endif
