#ifndef FACE_EXTRACTOR_PYTHON_HPP__
#define FACE_EXTRACTOR_PYTHON_HPP__
#include "factory/model.hpp"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class NNFaceExtractor {
public:
  NNFaceExtractor(const NNFactory *factory, ModelType model_type = BMFACEV03M,
                  int device_id = 0);
  ~NNFaceExtractor();
  py::list predict(py::list &inputs, py::list &bboxes_list,
                   py::list &points_list);
  py::list predict_mask(py::list &inputs, py::list &bboxes_list,
                        py::list &points_list);
  py::list predict(py::list &inputs, py::list &bboxes_list,
                   py::list &points_list, py::dict &additional_out);
  py::list predict(py::list &inputs, py::list &bboxes_list,
                   py::list &points_list, py::list &aligned_list);
  py::list predict_mask(py::list &inputs, py::list &bboxes_list,
                        py::list &points_list, py::list &aligned_list);
  py::list predict(py::array_t<unsigned char, py::array::c_style> &input,
                   py::list &bboxes, py::list &points);
  py::list predict(py::array_t<unsigned char, py::array::c_style> &input,
                   py::list &bboxes, py::list &points,
                   py::dict &additional_out);
  py::list predict(py::array_t<unsigned char, py::array::c_style> &input,
                   py::list &bboxes, py::list &points, py::list &aligned);
  py::list predict_cropped(py::list &inputs);

private:
  FaceExtract *extractor_ = nullptr;
  FaceExtract *mask_extractor_ = nullptr;
  void normalize(std::vector<float> &feature);
};

#endif
