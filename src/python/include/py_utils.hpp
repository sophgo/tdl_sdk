#ifndef PY_UTILS_HPP_
#define PY_UTILS_HPP_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "common/common_types.hpp"
#include "common/object_type_def.hpp"
namespace py = pybind11;

inline TDLDataType py_to_tdl_data_type(py::dtype dtype) {
  if (dtype.is(py::dtype::of<uint8_t>())) {
    return TDLDataType::UINT8;
  } else if (dtype.is(py::dtype::of<int8_t>())) {
    return TDLDataType::INT8;
  } else if (dtype.is(py::dtype::of<uint16_t>())) {
    return TDLDataType::UINT16;
  } else if (dtype.is(py::dtype::of<int16_t>())) {
    return TDLDataType::INT16;
  } else if (dtype.is(py::dtype::of<uint32_t>())) {
    return TDLDataType::UINT32;
  } else if (dtype.is(py::dtype::of<int32_t>())) {
    return TDLDataType::INT32;
  } else if (dtype.is(py::dtype::of<float>())) {
    return TDLDataType::FP32;
  } else {
    throw std::runtime_error("Unsupported data type");
  }
}

inline std::string object_type_to_string(TDLObjectType type) {
  switch (type) {
    case TDLObjectType::OBJECT_TYPE_UNDEFINED:
      return "UNDEFINED";
    case TDLObjectType::OBJECT_TYPE_PERSON:
      return "PERSON";
    case TDLObjectType::OBJECT_TYPE_FACE:
      return "FACE";
    case TDLObjectType::OBJECT_TYPE_HAND:
      return "HAND";
    case TDLObjectType::OBJECT_TYPE_HEAD:
      return "HEAD";
    case TDLObjectType::OBJECT_TYPE_HEAD_SHOULDER:
      return "HEAD_SHOULDER";
    case TDLObjectType::OBJECT_TYPE_HARD_HAT:
      return "HARD_HAT";
    case TDLObjectType::OBJECT_TYPE_FACE_MASK:
      return "FACE_MASK";
    case TDLObjectType::OBJECT_TYPE_CAR:
      return "CAR";
    case TDLObjectType::OBJECT_TYPE_BUS:
      return "BUS";
    case TDLObjectType::OBJECT_TYPE_TRUCK:
      return "TRUCK";
    case TDLObjectType::OBJECT_TYPE_MOTORBIKE:
      return "MOTORBIKE";
    case TDLObjectType::OBJECT_TYPE_BICYCLE:
      return "BICYCLE";
    case TDLObjectType::OBJECT_TYPE_LICENSE_PLATE:
      return "PLATE";
    case TDLObjectType::OBJECT_TYPE_CAT:
      return "CAT";
    case TDLObjectType::OBJECT_TYPE_DOG:
      return "DOG";
    default:
      return "UNKOWN";
  }
}
#endif
