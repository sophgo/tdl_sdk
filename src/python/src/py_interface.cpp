#include "py_image.hpp"
#include "py_model.hpp"
using namespace pytdl;

// 明确指定函数类型，解决重载问题
PyImage (*read_func)(const std::string&) = &read;
void (*write_func)(const PyImage&, const std::string&) = &write;
PyImage (*resize_func)(const PyImage&, int, int) = &resize;
PyImage (*crop_func)(const PyImage&,
                     const std::tuple<int, int, int, int>&) = &crop;
PyImage (*crop_resize_func)(const PyImage&,
                            const std::tuple<int, int, int, int>&, int,
                            int) = &cropResize;

// pybind11绑定实现
PYBIND11_MODULE(tdl, m) {
  m.doc() = "tdl sdk module python binding";

  // 图像模块
  py::module image = m.def_submodule("image", "image module");
  // 绑定枚举
  py::enum_<ImageFormat>(image, "ImageFormat")
      .value("RGB_PLANAR", ImageFormat::RGB_PLANAR)
      .value("BGR_PLANAR", ImageFormat::BGR_PLANAR)
      .value("RGB_PACKED", ImageFormat::RGB_PACKED)
      .value("BGR_PACKED", ImageFormat::BGR_PACKED)
      .value("GRAY", ImageFormat::GRAY)
      .value("YUV420SP_UV", ImageFormat::YUV420SP_UV)
      .value("YUV420SP_VU", ImageFormat::YUV420SP_VU)
      //   .value("NV12", ImageFormat::YUV420SP_UV)
      //   .value("NV21", ImageFormat::YUV420SP_VU)
      .value("YUV420P_UV", ImageFormat::YUV420P_UV)
      .value("YUV420P_VU", ImageFormat::YUV420P_VU)
      .value("YUV422P_UV", ImageFormat::YUV422P_UV)
      .value("YUV422P_VU", ImageFormat::YUV422P_VU)
      .value("YUV422SP_UV", ImageFormat::YUV422SP_UV)
      .value("YUV422SP_VU", ImageFormat::YUV422SP_VU)
      .export_values();

  py::enum_<TDLDataType>(image, "TDLDataType")
      .value("UINT8", TDLDataType::UINT8)
      .value("INT8", TDLDataType::INT8)
      .value("UINT16", TDLDataType::UINT16)
      .value("INT16", TDLDataType::INT16)
      .value("UINT32", TDLDataType::UINT32)
      .value("INT32", TDLDataType::INT32)
      .value("FP32", TDLDataType::FP32)
      .export_values();

  // 绑定图像类
  py::class_<PyImage>(image, "Image")
      .def(py::init<>())
      .def_static("from_numpy", &PyImage::fromNumpy, py::arg("numpy_array"),
                  py::arg("format") = ImageFormat::BGR_PACKED)
      .def("get_size", &PyImage::getSize)
      .def("get_format", &PyImage::getFormat);

  // 绑定模块函数，使用明确类型的函数指针
  image.def("write", write_func, py::arg("image"), py::arg("path"));
  image.def("read", read_func, py::arg("path"));
  image.def("resize", resize_func, py::arg("src"), py::arg("width"),
            py::arg("height"));
  image.def("crop", crop_func, py::arg("src"), py::arg("roi"));
  image.def("crop_resize", crop_resize_func, py::arg("src"), py::arg("roi"),
            py::arg("width"), py::arg("height"));

  // 神经网络模块
  py::module nn = m.def_submodule("nn", "Neural network algorithms module");
  py::enum_<ModelType>(nn, "ModelType")
      .value("FD_SCRFD", ModelType::SCRFD_FACE)
      .value("YOLOV8N_PERSON_VEHICLE", ModelType::YOLOV8N_PERSON_VEHICLE)
      .value("YOLOV8N_HEAD_HARDHAT", ModelType::YOLOV8N_HEAD_HARDHAT)
      .value("YOLOV10_COCO80", ModelType::YOLOV10_COCO80)
      .value("YOLOV6_COCO80", ModelType::YOLOV6_COCO80)
      .value("SEG_YOLOV8_COCO80", ModelType::SEG_YOLOV8_COCO80)
      .value("YOLOV8_POSE_PERSON17", ModelType::YOLOV8_POSE_PERSON17)
      .value("IMG_FEATURE_CLIP", ModelType::IMG_FEATURE_CLIP)
      .value("TEXT_FEATURE_CLIP", ModelType::TEXT_FEATURE_CLIP)
      .export_values();
  py::class_<PyObejectDetector>(nn, "ObjectDetector")
      .def(py::init<ModelType, std::string, int>(), py::arg("model_type"),
           py::arg("model_path"), py::arg("device_id") = 0)
      .def("inference", &PyObejectDetector::inference, py::arg("image"),
           py::arg("parameters") = py::dict());

  py::class_<PyFaceDetector>(nn, "FaceDetector")
      .def(py::init<ModelType, std::string, int>(), py::arg("model_type"),
           py::arg("model_path"), py::arg("device_id") = 0)
      .def("inference", &PyFaceDetector::inference, py::arg("image"),
           py::arg("parameters") = py::dict());
}