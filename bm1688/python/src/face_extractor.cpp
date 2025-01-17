#include "face_extractor.hpp"

NNFaceExtractor::NNFaceExtractor(const NNFactory *factory,
                                 ModelType model_type /*= BMFACEV03M*/,
                                 int device_id /*=0*/) {
  extractor_ = (FaceExtract *)(factory->get_model(model_type, device_id));
}

NNFaceExtractor::~NNFaceExtractor() {
  std::cout << "NNFaceExtractor destroyed" << std::endl;
  if (extractor_ != nullptr) {
    delete extractor_;
  }
}

void NNFaceExtractor::normalize(std::vector<float> &feature) {
  float norm_val = 0.00000000001;
  for (int i = 0; i < feature.size(); i++) {
    norm_val += feature[i] * feature[i];
  }
  norm_val = sqrt(norm_val);
  for (int i = 0; i < feature.size(); i++) {
    feature[i] /= norm_val;
  }
}

py::list NNFaceExtractor::predict(py::list &inputs, py::list &bboxes_list,
                                  py::list &points_list,
                                  py::list &aligned_list) {
  py::list results;
  auto *temp = new unsigned char[112 * 112 * 3];
  unsigned int offset = 336;  // 3 * 112
  for (int i = 0; i < inputs.size(); i++) {
    py::array_t<unsigned char, py::array::c_style> np_image =
        py::cast<py::array_t<unsigned char, py::array::c_style>>(inputs[i]);
    py::buffer_info buf = np_image.request();
    cv::Mat image((int)buf.shape[0], (int)buf.shape[1], CV_8UC3, buf.ptr);
    py::list bboxes = bboxes_list[i];
    py::list points = points_list[i];
    std::vector<cv::Mat> align_faces;
    py::list feas, faces;
    for (int j = 0; j < bboxes.size(); j++) {
      py::list bbox = bboxes[j];
      py::list point = points[j];
      FaceRect rect;
      rect.x1 = py::cast<float>(bbox[0]);
      rect.y1 = py::cast<float>(bbox[1]);
      rect.x2 = py::cast<float>(bbox[2]);
      rect.y2 = py::cast<float>(bbox[3]);
      for (int k = 0; k < 5; k++) {
        rect.facepts.x.push_back(py::cast<float>(point[k]));
        rect.facepts.y.push_back(py::cast<float>(point[k + 5]));
      }
      // rect.print();
      cv::imwrite("before_align_image.jpg", image);
      cv::Mat aligned_face = align_face(image, rect.facepts, 112, 112);
      cv::imwrite("align_image.jpg", aligned_face);
      align_faces.push_back(aligned_face);
      for (int i = 0; i < 112; i++) {
        const unsigned char *cur = aligned_face.ptr<unsigned char>(i);
        memcpy(temp + i * offset, cur, sizeof(unsigned char) * offset);
      }
      py::array_t<unsigned char, py::array::c_style> btemp({112, 112, 3}, temp);
      faces.append(btemp);
    }
    std::vector<std::vector<float>> features;
    extractor_->extract(align_faces, features);
    for (int j = 0; j < features.size(); j++) {
      std::vector<float> &feature = features[j];
      normalize(feature);
      feas.append(feature);
    }
    aligned_list.append(faces);
    results.append(feas);
  }
  delete[] temp;
  return results;
}

py::list NNFaceExtractor::predict(py::list &inputs, py::list &bboxes_list,
                                  py::list &points_list) {
  py::list temp;
  return predict(inputs, bboxes_list, points_list, temp);
}

py::list NNFaceExtractor::predict(py::list &inputs, py::list &bboxes_list,
                                  py::list &points_list,
                                  py::dict &additional_out) {
  return predict(inputs, bboxes_list, points_list);
}

py::list NNFaceExtractor::predict(
    py::array_t<unsigned char, py::array::c_style> &input, py::list &bboxes,
    py::list &points, py::list &aligned) {
  py::list inputs;
  inputs.append(input);
  py::list bboxes_list, points_list, aligned_list;
  bboxes_list.append(bboxes);
  points_list.append(points);
  py::list results = predict(inputs, bboxes_list, points_list, aligned_list);
  py::list pre_list = aligned_list[0];
  for (int i = 0; i < pre_list.size(); i++) {
    aligned.append(pre_list[i]);
  }
  return results[0];
}

py::list NNFaceExtractor::predict(
    py::array_t<unsigned char, py::array::c_style> &input, py::list &bboxes,
    py::list &points) {
  py::list temp;
  return predict(input, bboxes, points, temp);
}

py::list NNFaceExtractor::predict(
    py::array_t<unsigned char, py::array::c_style> &input, py::list &bboxes,
    py::list &points, py::dict &additional_out) {
  return predict(input, bboxes, points);
}

py::list NNFaceExtractor::predict_cropped(py::list &inputs) {
  std::vector<cv::Mat> align_faces;
  for (int i = 0; i < inputs.size(); i++) {
    py::array_t<unsigned char, py::array::c_style> np_image =
        py::cast<py::array_t<unsigned char, py::array::c_style>>(inputs[i]);
    py::buffer_info buf = np_image.request();
    cv::Mat image((int)buf.shape[0], (int)buf.shape[1], CV_8UC3, buf.ptr);
    align_faces.push_back(image);
  }
  std::vector<std::vector<float>> features;
  extractor_->extract(align_faces, features);
  py::list results;
  for (int j = 0; j < features.size(); j++) {
    std::vector<float> &feature = features[j];
    normalize(feature);
    results.append(feature);
  }
  return results;
}
