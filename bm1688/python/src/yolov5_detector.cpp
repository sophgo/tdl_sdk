#include "yolov5_detector.hpp"
#include <log/Logger.hpp>
NNYOLOV5Detector::NNYOLOV5Detector(const NNFactory *factory,
                                   int device_id /*=0*/) {
  detector_ =
      (YOLOV5 *)(factory->get_model(NNBaseModel::YOLO_V5_VEHICLE, device_id));
}

NNYOLOV5Detector ::~NNYOLOV5Detector() {
  if (detector_ != nullptr) {
    delete detector_;
    detector_ = nullptr;
  }
}

py::tuple NNYOLOV5Detector::predict(py::list &inputs) {
  // std::cout << "start to multi predict" << std::endl;
  std::vector<cv::Mat> images;
  for (int i = 0; i < inputs.size(); i++) {
    py::array_t<unsigned char, py::array::c_style> np_image =
        py::cast<py::array_t<unsigned char, py::array::c_style>>(inputs[i]);
    py::buffer_info buf = np_image.request();
    cv::Mat image((int)buf.shape[0], (int)buf.shape[1], CV_8UC3, buf.ptr);
    images.push_back(image);
  }
  std::vector<std::vector<ObjectBox>> detecteds;
  detector_->detect(images, 0.5, detecteds);

  py::list bboxes_list;
  py::list cls_list;
  py::list probs_list;
  for (int i = 0; i < images.size(); i++) {
    py::list res;
    py::list bboxes;
    py::list classes;
    py::list probs;
    auto &detected = detecteds[i];
    for (int j = 0; j < detected.size(); j++) {
      py::list bbox;
      auto &obj = detected[j];
      bbox.append(obj.x1);
      bbox.append(obj.y1);
      bbox.append(obj.x2);
      bbox.append(obj.y2);
      bboxes.append(bbox);
      classes.append(obj.label);
      probs.append(obj.score);
    }
    bboxes_list.append(bboxes);
    cls_list.append(classes);
    probs_list.append(probs);
  }
  py::tuple results = py::make_tuple(bboxes_list, cls_list, probs_list);
  return results;
}

py::tuple NNYOLOV5Detector::predict(
    py::array_t<unsigned char, py::array::c_style> &input) {
  // std::cout << "start to single predict" << std::endl;
  py::list inputs;
  inputs.append(input);
  py::tuple results = predict(inputs);
  py::list bboxes_list = results[0];
  py::list class_list = results[1];
  py::list probs_list = results[2];
  py::list bboxes = bboxes_list[0];
  py::list cls = class_list[0];
  py::list probs = probs_list[0];
  py::tuple result = py::make_tuple(bboxes, cls, probs);
  return result;
}
