#include "py_model.hpp"
#include <pybind11/stl.h>
#include "py_image.hpp"
#include "py_utils.hpp"
namespace pytdl {
TDLModelFactory g_model_factory_;

PyModel::PyModel(TDL_MODEL_TYPE model_type, const std::string& model_path,
                 const int device_id) {
  model_ = g_model_factory_.getModel(model_type, model_path, device_id);
  if (model_ == nullptr) {
    throw std::runtime_error("Failed to create model");
  }
}

PyObejectDetector::PyObejectDetector(TDL_MODEL_TYPE model_type,
                                     const std::string& model_path,
                                     const int device_id)
    : PyModel(model_type, model_path, device_id) {}

py::list PyObejectDetector::inference(const PyImage& image,
                                      py::dict parameters) {
  std::vector<std::shared_ptr<BaseImage>> images;
  images.push_back(image.getImage());
  std::map<std::string, float> parameters_map;
  for (auto& item : parameters) {
    parameters_map[item.first.cast<std::string>()] = item.second.cast<float>();
  }
  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  model_->inference(images, out_datas, parameters_map);
  std::shared_ptr<ModelOutputInfo> output_info = out_datas[0];
  if (output_info->getType() != ModelOutputType::OBJECT_DETECTION) {
    throw std::runtime_error("Model output type is not OBJECT_DETECTION");
  }
  std::shared_ptr<ModelBoxInfo> box_info =
      std::dynamic_pointer_cast<ModelBoxInfo>(output_info);
  if (!box_info) {
    throw std::runtime_error("Failed to cast to ModelBoxInfo");
  }
  py::list bboxes;
  for (auto& box : box_info->bboxes) {
    py::dict box_dict;
    box_dict[py::str("class_id")] = box.class_id;
    box_dict[py::str("class_name")] = object_type_to_string(box.object_type);
    box_dict[py::str("x1")] = box.x1;
    box_dict[py::str("y1")] = box.y1;
    box_dict[py::str("x2")] = box.x2;
    box_dict[py::str("y2")] = box.y2;
    box_dict[py::str("score")] = box.score;
    bboxes.append(box_dict);
  }
  return bboxes;
}

PyFaceDetector::PyFaceDetector(TDL_MODEL_TYPE model_type,
                               const std::string& model_path,
                               const int device_id)
    : PyObejectDetector(model_type, model_path, device_id) {}

py::list PyFaceDetector::inference(const PyImage& image, py::dict parameters) {
  std::vector<std::shared_ptr<BaseImage>> images;
  images.push_back(image.getImage());
  std::map<std::string, float> parameters_map;
  for (auto& item : parameters) {
    parameters_map[item.first.cast<std::string>()] = item.second.cast<float>();
  }
  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  model_->inference(images, out_datas, parameters_map);
  std::shared_ptr<ModelOutputInfo> output_info = out_datas[0];
  if (output_info->getType() !=
      ModelOutputType::OBJECT_DETECTION_WITH_LANDMARKS) {
    throw std::runtime_error("Model output type is not OBJECT_DETECTION");
  }
  std::shared_ptr<ModelBoxLandmarkInfo> box_landmark_info =
      std::dynamic_pointer_cast<ModelBoxLandmarkInfo>(output_info);
  if (!box_landmark_info) {
    throw std::runtime_error("Failed to cast to ModelBoxLandmarkInfo");
  }
  py::list bboxes;
  for (auto& box : box_landmark_info->box_landmarks) {
    py::dict box_dict;
    box_dict[py::str("class_id")] = box.class_id;
    box_dict[py::str("class_name")] = object_type_to_string(box.object_type);
    box_dict[py::str("x1")] = box.x1;
    box_dict[py::str("y1")] = box.y1;
    box_dict[py::str("x2")] = box.x2;
    box_dict[py::str("y2")] = box.y2;
    box_dict[py::str("score")] = box.score;
    py::list landmarks;
    for (size_t i = 0; i < box.landmarks_x.size(); ++i) {
      py::list landmark;
      landmark.append(box.landmarks_x[i]);
      landmark.append(box.landmarks_y[i]);
      landmarks.append(landmark);
    }
    box_dict[py::str("landmarks")] = landmarks;
    if (box.landmarks_score.size() > 0) {
      box_dict[py::str("landmarks_score")] = box.landmarks_score[0];
    }
    bboxes.append(box_dict);
  }
  return bboxes;
}
}  // namespace pytdl