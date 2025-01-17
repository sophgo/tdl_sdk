#include "head_pose.hpp"

NNHeadPose::NNHeadPose(const NNFactory *factory, int device_id /*=0*/) {
  headpose_ = (FaceDDFA *)(factory->get_model(DDFA, device_id));
}

NNHeadPose::~NNHeadPose() {
  std::cout << "NNHeadPose destroyed" << std::endl;
  if (headpose_ != nullptr) {
    delete headpose_;
  }
}

py::list NNHeadPose::predict(py::list &inputs, py::list &bboxes_list) {
  std::vector<cv::Mat> images;
  std::vector<std::vector<FaceRect>> detects;
  for (int i = 0; i < inputs.size(); i++) {
    py::array_t<unsigned char, py::array::c_style> np_image =
        py::cast<py::array_t<unsigned char, py::array::c_style>>(inputs[i]);
    py::buffer_info buf = np_image.request();
    cv::Mat image((int)buf.shape[0], (int)buf.shape[1], CV_8UC3, buf.ptr);
    images.push_back(image);

    py::list bboxes = bboxes_list[i];
    std::vector<FaceRect> detect;
    for (int j = 0; j < bboxes.size(); j++) {
      py::list bbox = bboxes[j];
      FaceRect rect;
      rect.x1 = py::cast<float>(bbox[0]);
      rect.y1 = py::cast<float>(bbox[1]);
      rect.x2 = py::cast<float>(bbox[2]);
      rect.y2 = py::cast<float>(bbox[3]);
      // rect.print();
      detect.push_back(rect);
    }
    detects.push_back(detect);
  }
  headpose_->detect(images, detects);
  py::list results;
  for (int i = 0; i < images.size(); i++) {
    py::list result;
    std::vector<FaceRect> &rects = detects[i];
    for (int j = 0; j < rects.size(); j++) {
      py::list angles;
      FaceRect &rect = rects[j];
      angles.append(rect.facepose.pitch);
      angles.append(rect.facepose.yaw);
      angles.append(rect.facepose.roll);
      result.append(angles);
    }
    results.append(result);
  }
  return results;
}

py::list
NNHeadPose::predict(py::array_t<unsigned char, py::array::c_style> &input,
                    py::list &bboxes) {
  py::list inputs;
  inputs.append(input);
  py::list bboxes_list;
  bboxes_list.append(bboxes);
  py::list results = predict(inputs, bboxes_list);
  return results[0];
}
