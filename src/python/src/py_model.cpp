#include "py_model.hpp"
#include <pybind11/stl.h>
#include "py_image.hpp"
#include "py_utils.hpp"
namespace pytdl {

PyModel::PyModel(ModelType model_type, const std::string& model_path,
                 const int device_id) {
  model_ = TDLModelFactory::getInstance().getModel(model_type, model_path, {},
                                                   device_id);
  if (model_ == nullptr) {
    throw std::runtime_error("Failed to create model");
  }
}

PyObjectDetector::PyObjectDetector(ModelType model_type,
                                   const std::string& model_path,
                                   const int device_id)
    : PyModel(model_type, model_path, device_id) {}

py::list PyObjectDetector::inference(const PyImage& image,
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
py::list PyObjectDetector::inference(
    const py::array_t<unsigned char, py::array::c_style>& input,
    py::dict parameters) {
  PyImage image = PyImage::fromNumpy(input);
  return inference(image, parameters);
}

PyFaceDetector::PyFaceDetector(ModelType model_type,
                               const std::string& model_path,
                               const int device_id)
    : PyObjectDetector(model_type, model_path, device_id) {}

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
py::list PyFaceDetector::inference(
    const py::array_t<unsigned char, py::array::c_style>& input,
    py::dict parameters) {
  PyImage image = PyImage::fromNumpy(input);
  return inference(image, parameters);
}

PyClassifier::PyClassifier(ModelType model_type, const std::string& model_path,
                           const int device_id)
    : PyModel(model_type, model_path, device_id) {}

py::dict PyClassifier::inference(const PyImage& image, py::dict parameters) {
  std::vector<std::shared_ptr<BaseImage>> images;
  images.push_back(image.getImage());
  std::map<std::string, float> parameters_map;
  for (auto& item : parameters) {
    parameters_map[item.first.cast<std::string>()] = item.second.cast<float>();
  }
  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  model_->inference(images, out_datas, parameters_map);
  std::shared_ptr<ModelOutputInfo> output_info = out_datas[0];
  if (output_info->getType() != ModelOutputType::CLASSIFICATION) {
    throw std::runtime_error("Model output type is not CLASSIFICATION");
  }
  std::shared_ptr<ModelClassificationInfo> classification_output =
      std::dynamic_pointer_cast<ModelClassificationInfo>(output_info);
  if (!classification_output) {
    throw std::runtime_error("Failed to cast to ModelClassificationInfo");
  }
  py::dict classification_dict;

  classification_dict[py::str("class_id")] =
      classification_output->topk_class_ids[0];
  classification_dict[py::str("score")] = classification_output->topk_scores[0];

  return classification_dict;
}
py::dict PyClassifier::inference(
    const py::array_t<unsigned char, py::array::c_style>& input,
    py::dict parameters) {
  PyImage image = PyImage::fromNumpy(input);
  return inference(image, parameters);
}

PyAttributeExtractor::PyAttributeExtractor(ModelType model_type,
                                           const std::string& model_path,
                                           const int device_id)
    : PyModel(model_type, model_path, device_id) {}

py::dict PyAttributeExtractor::inference(const PyImage& image,
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

  std::shared_ptr<ModelAttributeInfo> box_attribute_output =
      std::dynamic_pointer_cast<ModelAttributeInfo>(output_info);
  if (!box_attribute_output) {
    throw std::runtime_error("Failed to cast to ModelAttributeInfo");
  }
  py::dict face_attribute_dict;

  float mask_score =
      box_attribute_output->attributes[OBJECT_ATTRIBUTE_HUMAN_MASK];
  float gender_score =
      box_attribute_output->attributes[OBJECT_ATTRIBUTE_HUMAN_GENDER];
  float age_score =
      box_attribute_output->attributes[OBJECT_ATTRIBUTE_HUMAN_AGE];
  float glass_score =
      box_attribute_output->attributes[OBJECT_ATTRIBUTE_HUMAN_GLASSES];

  face_attribute_dict[py::str("mask_score")] = mask_score;
  face_attribute_dict[py::str("is_wearing_mask")] =
      (mask_score > 0.5) ? py::bool_(true) : py::bool_(false);

  face_attribute_dict[py::str("gender_score")] = gender_score;
  face_attribute_dict[py::str("is_male")] =
      (gender_score > 0.5) ? py::bool_(true) : py::bool_(false);

  face_attribute_dict[py::str("age_score")] = age_score;
  face_attribute_dict[py::str("age")] = int(age_score * 100);

  face_attribute_dict[py::str("glass_score")] = glass_score;
  face_attribute_dict[py::str("is_wearing_glasses")] =
      (glass_score > 0.5) ? py::bool_(true) : py::bool_(false);

  return face_attribute_dict;
}
py::dict PyAttributeExtractor::inference(
    const py::array_t<unsigned char, py::array::c_style>& input,
    py::dict parameters) {
  PyImage image = PyImage::fromNumpy(input);
  return inference(image, parameters);
}

PyFaceLandmark::PyFaceLandmark(ModelType model_type,
                               const std::string& model_path,
                               const int device_id)
    : PyModel(model_type, model_path, device_id) {}

py::dict PyFaceLandmark::inference(const PyImage& image, py::dict parameters) {
  std::vector<std::shared_ptr<BaseImage>> images;
  images.push_back(image.getImage());
  std::map<std::string, float> parameters_map;
  for (auto& item : parameters) {
    parameters_map[item.first.cast<std::string>()] = item.second.cast<float>();
  }
  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;

  model_->inference(images, out_datas, parameters_map);
  std::shared_ptr<ModelOutputInfo> output_info = out_datas[0];

  std::shared_ptr<ModelLandmarksInfo> face_lankmark_output =
      std::dynamic_pointer_cast<ModelLandmarksInfo>(output_info);
  if (!face_lankmark_output) {
    throw std::runtime_error("Failed to cast to ModelAttributeInfo");
  }
  py::dict face_lankmark_dict;

  face_lankmark_dict[py::str("score")] =
      face_lankmark_output->landmarks_score[0];
  py::list landmarks;
  for (size_t i = 0; i < face_lankmark_output->landmarks_x.size(); ++i) {
    py::list landmark;
    landmark.append(face_lankmark_output->landmarks_x[i]);
    landmark.append(face_lankmark_output->landmarks_y[i]);
    landmarks.append(landmark);
  }
  face_lankmark_dict[py::str("landmarks")] = landmarks;
  return face_lankmark_dict;
}
py::dict PyFaceLandmark::inference(
    const py::array_t<unsigned char, py::array::c_style>& input,
    py::dict parameters) {
  PyImage image = PyImage::fromNumpy(input);
  return inference(image, parameters);
}

PyKeyPointDetector::PyKeyPointDetector(ModelType model_type,
                                       const std::string& model_path,
                                       const int device_id)
    : PyModel(model_type, model_path, device_id) {}

py::dict PyKeyPointDetector::inference(const PyImage& image,
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
  if (output_info->getType() != ModelOutputType::OBJECT_LANDMARKS) {
    throw std::runtime_error("Model output type is not OBJECT_LANDMARKS");
  }
  std::shared_ptr<ModelLandmarksInfo> box_landmark_info =
      std::dynamic_pointer_cast<ModelLandmarksInfo>(output_info);
  if (!box_landmark_info) {
    throw std::runtime_error("Failed to cast to ModelLandmarkInfo");
  }
  py::dict landmark_dict;
  py::list landmarks;
  for (size_t i = 0; i < box_landmark_info->landmarks_x.size(); ++i) {
    py::list landmark;
    landmark.append(box_landmark_info->landmarks_x[i]);
    landmark.append(box_landmark_info->landmarks_y[i]);
    landmarks.append(landmark);
  }
  landmark_dict[py::str("landmarks")] = landmarks;
  if (box_landmark_info->landmarks_score.size() > 0) {
    py::list landmark_score;
    for (size_t i = 0; i < box_landmark_info->landmarks_score.size(); ++i) {
      landmark_score.append(box_landmark_info->landmarks_score[i]);
    }
    landmark_dict[py::str("landmarks_score")] = landmark_score;
  }
  return landmark_dict;
}
py::dict PyKeyPointDetector::inference(
    const py::array_t<unsigned char, py::array::c_style>& input,
    py::dict parameters) {
  PyImage image = PyImage::fromNumpy(input);
  return inference(image, parameters);
}

PySemanticSegmentation::PySemanticSegmentation(ModelType model_type,
                                               const std::string& model_path,
                                               const int device_id)
    : PyModel(model_type, model_path, device_id) {}

py::dict PySemanticSegmentation::inference(const PyImage& image,
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
  if (output_info->getType() != ModelOutputType::SEGMENTATION) {
    throw std::runtime_error("Model output type is not SEGMENTATION");
  }
  std::shared_ptr<ModelSegmentationInfo> segmentation_output =
      std::dynamic_pointer_cast<ModelSegmentationInfo>(output_info);
  if (!segmentation_output) {
    throw std::runtime_error("Failed to cast to ModelSegmentationInfo");
  }
  py::dict segmentation_dict;
  segmentation_dict[py::str("output_width")] =
      segmentation_output->output_width;
  segmentation_dict[py::str("output_height")] =
      segmentation_output->output_height;
  py::list class_id;
  py::list class_conf;
  for (size_t i = 0; i < segmentation_output->output_width *
                             segmentation_output->output_height;
       ++i) {
    class_id.append(segmentation_output->class_id[i]);
    class_conf.append(segmentation_output->class_conf[i]);
  }
  segmentation_dict[py::str("class_id")] = class_id;
  segmentation_dict[py::str("class_conf")] = class_conf;
  return segmentation_dict;
}
py::dict PySemanticSegmentation::inference(
    const py::array_t<unsigned char, py::array::c_style>& input,
    py::dict parameters) {
  PyImage image = PyImage::fromNumpy(input);
  return inference(image, parameters);
}

PyInstanceSegmentation::PyInstanceSegmentation(ModelType model_type,
                                               const std::string& model_path,
                                               const int device_id)
    : PyModel(model_type, model_path, device_id) {}

py::dict PyInstanceSegmentation::inference(const PyImage& image,
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
  if (output_info->getType() !=
      ModelOutputType::OBJECT_DETECTION_WITH_SEGMENTATION) {
    throw std::runtime_error(
        "Model output type is not OBJECT_DETECTION_WITH_SEGMENTATION");
  }
  std::shared_ptr<ModelBoxSegmentationInfo> instance_seg_output =
      std::dynamic_pointer_cast<ModelBoxSegmentationInfo>(output_info);
  if (!instance_seg_output) {
    throw std::runtime_error("Failed to cast to ModelBoxSegmentationInfo");
  }
  py::dict instance_seg_dict;
  instance_seg_dict[py::str("mask_width")] = instance_seg_output->mask_width;
  instance_seg_dict[py::str("mask_height")] = instance_seg_output->mask_height;
  py::list bboxes_seg;
  for (auto& box_seg_info : instance_seg_output->box_seg) {
    py::dict box_seg_dict;
    box_seg_dict[py::str("class_id")] = box_seg_info.class_id;
    box_seg_dict[py::str("class_name")] =
        object_type_to_string(box_seg_info.object_type);
    box_seg_dict[py::str("x1")] = box_seg_info.x1;
    box_seg_dict[py::str("y1")] = box_seg_info.y1;
    box_seg_dict[py::str("x2")] = box_seg_info.x2;
    box_seg_dict[py::str("y2")] = box_seg_info.y2;
    box_seg_dict[py::str("score")] = box_seg_info.score;
    py::list mask;

    for (int i = 0;
         i < instance_seg_output->mask_width * instance_seg_output->mask_height;
         i++) {
      mask.append(box_seg_info.mask[i]);
      std::cout << box_seg_info.mask[i] << std::endl;
    }
    box_seg_dict[py::str("mask")] = mask;
    // box_seg_dict[py::str("mask_point_size")] = box_seg_info.mask_point_size;
    // py::list mask_points;
    // for(int i; i< box_seg_info.mask_point_size; i++){
    //   py::list mask_point;
    //   mask_point.append(box_seg_info.mask_point[2*i]);
    //   mask_point.append(box_seg_info.mask_point[2*i+1]);
    //   mask_points.append(mask_point);
    // }
    bboxes_seg.append(box_seg_dict);
  }
  instance_seg_dict[py::str("bboxes_seg")] = bboxes_seg;
  return instance_seg_dict;
}
py::dict PyInstanceSegmentation::inference(
    const py::array_t<unsigned char, py::array::c_style>& input,
    py::dict parameters) {
  PyImage image = PyImage::fromNumpy(input);
  return inference(image, parameters);
}

PyLaneDetection::PyLaneDetection(ModelType model_type,
                                 const std::string& model_path,
                                 const int device_id)
    : PyModel(model_type, model_path, device_id) {}

py::list PyLaneDetection::inference(const PyImage& image, py::dict parameters) {
  std::vector<std::shared_ptr<BaseImage>> images;
  images.push_back(image.getImage());
  std::map<std::string, float> parameters_map;
  for (auto& item : parameters) {
    parameters_map[item.first.cast<std::string>()] = item.second.cast<float>();
  }
  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  model_->inference(images, out_datas, parameters_map);
  std::shared_ptr<ModelOutputInfo> output_info = out_datas[0];
  if (output_info->getType() != ModelOutputType::OBJECT_LANDMARKS) {
    throw std::runtime_error("Model output type is not OBJECT_LANDMARKS");
  }
  std::shared_ptr<ModelBoxLandmarkInfo> lane_output =
      std::dynamic_pointer_cast<ModelBoxLandmarkInfo>(output_info);
  if (!lane_output) {
    throw std::runtime_error("Failed to cast to ModelBoxLandmarkInfo");
  }
  py::list lanes_list;
  for (size_t j = 0; j < lane_output->box_landmarks.size(); j++) {
    py::list landmarks;
    for (int k = 0; k < 2; k++) {
      py::list landmark;
      landmark.append(lane_output->box_landmarks[j].landmarks_x[k]);
      landmark.append(lane_output->box_landmarks[j].landmarks_y[k]);
      landmarks.append(landmark);
    }
    lanes_list.append(landmarks);
  }
  return lanes_list;
}
py::list PyLaneDetection::inference(
    const py::array_t<unsigned char, py::array::c_style>& input,
    py::dict parameters) {
  PyImage image = PyImage::fromNumpy(input);
  return inference(image, parameters);
}

PyFeatureExtractor::PyFeatureExtractor(ModelType model_type,
                                       const std::string& model_path,
                                       const int device_id)
    : PyModel(model_type, model_path, device_id) {}

py::array_t<float> PyFeatureExtractor::inference(const PyImage& image,
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
  if (output_info->getType() != ModelOutputType::FEATURE_EMBEDDING) {
    throw std::runtime_error("Model output type is not FEATURE_EMBEDDING");
  }
  std::shared_ptr<ModelFeatureInfo> feature_output =
      std::dynamic_pointer_cast<ModelFeatureInfo>(output_info);
  if (!feature_output) {
    throw std::runtime_error("Failed to cast to ModelFeatureInfo");
  }

  ssize_t size = static_cast<ssize_t>(feature_output->embedding_num);
  switch (feature_output->embedding_type) {
    case TDLDataType::INT8: {
      int8_t* feature_ptr =
          reinterpret_cast<int8_t*>(feature_output->embedding);
      return py::array_t<int8_t>({size}, {sizeof(int8_t)}, feature_ptr);
    }
    case TDLDataType::UINT8: {
      uint8_t* feature_ptr =
          reinterpret_cast<uint8_t*>(feature_output->embedding);
      return py::array_t<uint8_t>({size}, {sizeof(uint8_t)}, feature_ptr);
    }
    case TDLDataType::FP32: {
      float* feature_ptr = reinterpret_cast<float*>(feature_output->embedding);
      return py::array_t<float>({size}, {sizeof(float)}, feature_ptr);
    }
    default:
      assert(false && "Unsupported embedding_type");
      return py::array_t<float>();
  }
}
py::array_t<float> PyFeatureExtractor::inference(
    const py::array_t<unsigned char, py::array::c_style>& input,
    py::dict parameters) {
  PyImage image = PyImage::fromNumpy(input);
  return inference(image, parameters);
}

PyCharacterRecognitor::PyCharacterRecognitor(ModelType model_type,
                                             const std::string& model_path,
                                             const int device_id)
    : PyModel(model_type, model_path, device_id) {}

py::list PyCharacterRecognitor::inference(const PyImage& image,
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
  if (output_info->getType() != ModelOutputType::OCR_INFO) {
    throw std::runtime_error("Model output type is not OCR_INFO");
  }
  std::shared_ptr<ModelOcrInfo> char_output =
      std::dynamic_pointer_cast<ModelOcrInfo>(output_info);
  if (!char_output) {
    throw std::runtime_error("Failed to cast to ModelOcrInfo");
  }
  py::list char_list;
  std::string str(char_output->text_info);
  char_list.append(str);
  return char_list;
}
py::list PyCharacterRecognitor::inference(
    const py::array_t<unsigned char, py::array::c_style>& input,
    py::dict parameters) {
  PyImage image = PyImage::fromNumpy(input);
  return inference(image, parameters);
}

PyTracker::PyTracker(TrackerType type) {
  tracker_ = TrackerFactory::createTracker(type);
}
void PyTracker::setPairConfig(
    const std::map<TDLObjectType, TDLObjectType>& object_pair_config) {
  tracker_->setPairConfig(object_pair_config);
}
void PyTracker::setTrackConfig(const TrackerConfig& track_config) {
  tracker_->setTrackConfig(track_config);
}
TrackerConfig PyTracker::getTrackConfig() { return tracker_->getTrackConfig(); }

py::list PyTracker::track(const py::list& boxes, uint64_t frame_id) {
  static std::unordered_map<size_t, ObjectBoxInfo> prev_boxes;
  std::vector<ObjectBoxInfo> box_vec;
  for (size_t i = 0; i < boxes.size(); ++i) {
    py::dict box_dict = boxes[i].cast<py::dict>();
    ObjectBoxInfo box_info;
    box_info.x1 = box_dict["x1"].cast<float>();
    box_info.y1 = box_dict["y1"].cast<float>();
    box_info.x2 = box_dict["x2"].cast<float>();
    box_info.y2 = box_dict["y2"].cast<float>();
    box_info.class_id = box_dict["class_id"].cast<int>();
    // box_info.class_name = box_dict["class_name"].cast<std::string>();
    box_info.score = box_dict["score"].cast<float>();
    box_info.object_type =
        string_to_object_type(box_dict["class_name"].cast<std::string>());
    // box_info.object_type = static_cast<TDLObjectType>(box_info.class_id);
    box_vec.push_back(box_info);
  }
  std::vector<TrackerInfo> trackers;
  tracker_->track(box_vec, frame_id, trackers);
  py::list result;
  for (auto& tracker : trackers) {
    py::dict tracker_dict;
    py::dict box_dict;
    box_dict["x1"] = tracker.box_info_.x1;
    box_dict["y1"] = tracker.box_info_.y1;
    box_dict["x2"] = tracker.box_info_.x2;
    box_dict["y2"] = tracker.box_info_.y2;
    box_dict["class_id"] = tracker.box_info_.class_id;
    // box_dict["class_name"] =
    // object_type_to_string(tracker.box_info_.object_type);
    box_dict["score"] = tracker.box_info_.score;
    tracker_dict["box_info"] = box_dict;
    tracker_dict["status"] = static_cast<int>(tracker.status_);
    tracker_dict["obj_idx"] = tracker.obj_idx_;
    tracker_dict["track_id"] = tracker.track_id_;
    float velocity_x =
        std::isinf(tracker.velocity_x_) ? 0.0f : tracker.velocity_x_;
    float velocity_y =
        std::isinf(tracker.velocity_y_) ? 0.0f : tracker.velocity_y_;
    tracker_dict["velocity_x"] = velocity_x;
    tracker_dict["velocity_y"] = velocity_y;
    result.append(tracker_dict);
    prev_boxes[tracker.track_id_] = tracker.box_info_;
  }
  return result;
}
void PyTracker::setImgSize(int width, int height) {
  tracker_->setImgSize(width, height);
}

py::dict PyModel::getPreprocessParameters() {
  PreprocessParams pre_params;
  model_->getPreprocessParameters(pre_params);
  py::dict params;
  params["mean"] = py::make_tuple(pre_params.mean[0], pre_params.mean[1],
                                  pre_params.mean[2]);
  params["scale"] = py::make_tuple(pre_params.scale[0], pre_params.scale[1],
                                   pre_params.scale[2]);
  return params;
}

void PyModel::setPreprocessParameters(const py::dict& params) {
  PreprocessParams pre_params;
  auto mean = params["mean"].cast<py::tuple>();
  auto scale = params["scale"].cast<py::tuple>();
  pre_params.mean[0] = mean[0].cast<float>();
  pre_params.mean[1] = mean[1].cast<float>();
  pre_params.mean[2] = mean[2].cast<float>();
  pre_params.scale[0] = scale[0].cast<float>();
  pre_params.scale[1] = scale[1].cast<float>();
  pre_params.scale[2] = scale[2].cast<float>();
  model_->setPreprocessParameters(pre_params);
}
}  // namespace pytdl