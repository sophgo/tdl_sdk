#include "nnfactory.hpp"

#include <iostream>
#include <log/Logger.hpp>

#include "common/device_info.hpp"
namespace py = pybind11;

NNF::NNF(const std::string& model_dir, int device_id /*=-1*/) {
  factory_ = new NNFactory(model_dir);

  if (device_id == -1) {
    std::vector<int> devices = get_device_ids();
    device_id_ = devices[0];
  } else {
    device_id_ = device_id;
  }
  LOG(INFO) << "initialize nnfactory,use device:" << device_id_;
}

NNF::~NNF() { delete factory_; }

// Note: ownership of the memory allocated in C++ is transferred to python
NNFaceDetector* NNF::get_face_detector(
    const std::string& detector_type /*= "scrfd"*/) {
  return new NNFaceDetector(factory_, device_id_, detector_type);
}

NNFaceExtractor* NNF::get_face_extractor(
    const std::string& model_name /*="BMFACEV03M"*/) {
  if (model_name == "BMFACEV03M") {
    return new NNFaceExtractor(factory_, BMFACEV03M, device_id_);
  } else if (model_name == "BMFACER18") {
    return new NNFaceExtractor(factory_, BMFACER18, device_id_);
  } else if (model_name == "BMFACER34") {
    return new NNFaceExtractor(factory_, BMFACER34, device_id_);
  } else if (model_name == "BMFACER34_V2") {
    return new NNFaceExtractor(factory_, BMFACER34_V2, device_id_);
  } else if (model_name == "BMFACER34_V2_FP32") {
    return new NNFaceExtractor(factory_, BMFACER34_V2_FP32, device_id_);
  } else if (model_name == "BMFACER34_V3") {
    return new NNFaceExtractor(factory_, NNBaseModel::BMFACER34_V3, device_id_);
  } else {
    std::cout << "no current model for face extractor,use BMFACEV03M as default"
              << std::endl;
    return new NNFaceExtractor(factory_, BMFACEV03M, device_id_);
  }
}

NNYOLOV5Detector* NNF::get_vehicle_detector() {
  return new NNYOLOV5Detector(factory_, device_id_);
}

py::array_t<unsigned char, py::array::c_style> NNF::align(
    py::array_t<unsigned char, py::array::c_style> input, py::list bbox,
    py::list point) {
  py::buffer_info buf = input.request();
  cv::Mat image((int)buf.shape[0], (int)buf.shape[1], CV_8UC3, buf.ptr);
  FaceRect rect;
  rect.x1 = py::cast<float>(bbox[0]);
  rect.y1 = py::cast<float>(bbox[1]);
  rect.x2 = py::cast<float>(bbox[2]);
  rect.y2 = py::cast<float>(bbox[3]);
  for (int k = 0; k < 5; k++) {
    rect.facepts.x.push_back(py::cast<float>(point[k]));
    rect.facepts.y.push_back(py::cast<float>(point[k + 5]));
  }
  cv::Mat aligned = align_face(image, rect.facepts, 112, 112);
  auto* temp = new unsigned char[aligned.cols * aligned.rows * 3];
  for (int i = 0; i < aligned.rows; i++) {
    const unsigned char* cur = aligned.ptr<unsigned char>(i);
    unsigned int offset = 3 * aligned.cols;
    memcpy(temp + i * offset, cur, sizeof(unsigned char) * offset);
  }
  py::array_t<unsigned char, py::array::c_style> btemp(
      {aligned.cols, aligned.rows, 3}, temp);
  delete[] temp;
  return btemp;
}

PYBIND11_MODULE(nnfactory, m) {
  m.doc() = "pybind11 for NNFactory";  // optional

  py::class_<NNF> nnf(m, "NNFactory");

  nnf.def(py::init<const std::string&, int>(), py::arg("model_dir"),
          py::arg("device_id") = -1)
      //      .def("get_detector", &NNF::get_face_detector,
      //      py::return_value_policy::reference_internal)
      .def("get_detector", &NNF::get_face_detector,
           py::arg("detector_type") = "scrfd")
      .def("get_feature_extractor", &NNF::get_face_extractor,
           py::arg("model_type") = "BMFACEV03M")
      .def("get_vehicle_detector", &NNF::get_vehicle_detector);

  //     .def("get_liveness_classifier", &NNF::get_liveness_classifier);
  //     .def("get_feature_extractor_depth", &NNF::get_face_extractor)
  //     .def("get_feature_extractor_nir", &NNF::get_face_extractor)
  //     .def("get_object_detector", &NNF::get_object_detector)
  nnf.def("align", &NNF::align);

  py::class_<NNFaceDetector> cssd(m, "FaceDetector");
  cssd.def("predict",
           (py::tuple(NNFaceDetector::*)(py::list&, int))(
               &NNFaceDetector::predict),
           "predict fuction", py::arg("images"), py::arg("detect_landmark") = 1)
      .def("predict",
           (py::tuple(NNFaceDetector::*)(py::array_t<unsigned char, 1>&, int))(
               &NNFaceDetector::predict),
           "predict fuction", py::arg("image"), py::arg("detect_landmark") = 1);

  py::class_<NNFaceDetMark> det_mark(m, "FaceDetMark");
  det_mark
      .def("predict",
           (py::tuple(NNFaceDetMark::*)(py::list&))(&NNFaceDetMark::predict),
           "...", py::arg("images"))
      // .def("predict", (py::tuple (NNFaceDetMark::*)(py::list&, py::float_&,
      // py::float_&))
      //          (&NNFaceDetMark::predict), "...",
      //      py::arg("images"),
      //      py::arg("detection_threshold") = 0.6,
      //      py::arg("point_threshold") = 0.7)
      .def("predict",
           (py::tuple(NNFaceDetMark::*)(py::array_t<unsigned char, 1>&))(
               &NNFaceDetMark::predict),
           "predict fuction", py::arg("image"));

  py::class_<NNFaceExtractor> bmface(m, "FaceExtractor");
  bmface
      .def("predict",
           (py::list(NNFaceExtractor::*)(py::list&, py::list&, py::list&))(
               &NNFaceExtractor::predict),
           "predict function", py::arg("images_list"), py::arg("bboxes_list"),
           py::arg("points_list"))
      .def("predict",
           (py::list(NNFaceExtractor::*)(py::list&, py::list&, py::list&,
                                         py::dict&))(&NNFaceExtractor::predict),
           "predict function", py::arg("images_list"), py::arg("bboxes_list"),
           py::arg("points_list"), py::arg("additional_out"))
      .def("predict",
           (py::list(NNFaceExtractor::*)(py::list&, py::list&, py::list&,
                                         py::list&))(&NNFaceExtractor::predict),
           "predict function", py::arg("images_list"), py::arg("bboxes_list"),
           py::arg("points_list"), py::arg("aligned_list"))
      .def("predict",
           (py::list(NNFaceExtractor::*)(py::array_t<unsigned char, 1>&,
                                         py::list&,
                                         py::list&))(&NNFaceExtractor::predict),
           "predict function", py::arg("image"), py::arg("bboxes"),
           py::arg("points"))
      .def("predict",
           (py::list(NNFaceExtractor::*)(py::array_t<unsigned char, 1>&,
                                         py::list&, py::list&,
                                         py::dict&))(&NNFaceExtractor::predict),
           "predict function", py::arg("image"), py::arg("bboxes"),
           py::arg("points"), py::arg("additional_out"))
      .def("predict",
           (py::list(NNFaceExtractor::*)(py::array_t<unsigned char, 1>&,
                                         py::list&, py::list&,
                                         py::list&))(&NNFaceExtractor::predict),
           "predict function", py::arg("image"), py::arg("bboxes"),
           py::arg("points"), py::arg("aligned"))
      .def("predict_cropped",
           (py::list(NNFaceExtractor::*)(py::list&))(
               &NNFaceExtractor::predict_cropped),
           "predict function", py::arg("image"));

  py::class_<NNYOLOV5Detector> yolov5(m, "NNYOLOV5Detector");
  yolov5
      .def("predict",
           (py::tuple(NNYOLOV5Detector::*)(py::list&))(
               &NNYOLOV5Detector::predict),
           "predict fuction", py::arg("images"))
      .def("predict",
           (py::tuple(NNYOLOV5Detector::*)(py::array_t<unsigned char, 1>&))(
               &NNYOLOV5Detector::predict),
           "predict fuction", py::arg("image"));
}
