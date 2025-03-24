#include "py_model.hpp"
#include <pybind11/stl.h>
#include "py_image.hpp"
#include "py_utils.hpp"
namespace pytdl {
TDLModelFactory g_model_factory_;

PyModel::PyModel(ModelType model_type, const std::string& model_path,
                 const int device_id) {
  model_ = g_model_factory_.getModel(model_type, model_path, device_id);
  if (model_ == nullptr) {
    throw std::runtime_error("Failed to create model");
  }
}

PyObejectDetector::PyObejectDetector(ModelType model_type,
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

PyFaceDetector::PyFaceDetector(ModelType model_type,
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

PyClassifier::PyClassifier(ModelType model_type,
                               const std::string& model_path,
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
  if (output_info->getType() !=
      ModelOutputType::CLASSIFICATION) {
    throw std::runtime_error("Model output type is not CLASSIFICATION");
  }
  std::shared_ptr<ModelClassificationInfo> classification_output =
      std::dynamic_pointer_cast<ModelClassificationInfo>(output_info);
  if (!classification_output) {
    throw std::runtime_error("Failed to cast to ModelClassificationInfo");
  }
  py::dict classification_dict;

  classification_dict[py::str("class_id")] = classification_output->topk_class_ids[0];
  classification_dict[py::str("score")] = classification_output->topk_scores[0];

  return classification_dict;
}

PyAttributeExtractor::PyAttributeExtractor(ModelType model_type,
                               const std::string& model_path,
                               const int device_id)
    : PyModel(model_type, model_path, device_id) {}

py::dict PyAttributeExtractor::inference(const PyImage& image, py::dict parameters) {
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

  float mask_score = box_attribute_output->attributes[OBJECT_ATTRIBUTE_HUMAN_MASK];
  float gender_score = box_attribute_output->attributes[OBJECT_ATTRIBUTE_HUMAN_GENDER];
  float age_score = box_attribute_output->attributes[OBJECT_ATTRIBUTE_HUMAN_AGE];
  float glass_score = box_attribute_output->attributes[OBJECT_ATTRIBUTE_HUMAN_GLASSES];

  face_attribute_dict[py::str("mask_score")] = mask_score;
  face_attribute_dict[py::str("is_wearing_mask")] = (mask_score > 0.5) ? py::bool_(true) : py::bool_(false);

  face_attribute_dict[py::str("gender_score")] = gender_score;
  face_attribute_dict[py::str("is_male")] = (gender_score > 0.5) ? py::bool_(true) : py::bool_(false);

  face_attribute_dict[py::str("age_score")] = age_score;
  face_attribute_dict[py::str("age")] = int(age_score*100);

  face_attribute_dict[py::str("glass_score")] = glass_score;
  face_attribute_dict[py::str("is_wearing_glasses")] = (glass_score > 0.5) ? py::bool_(true) : py::bool_(false);
  
  return face_attribute_dict;
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

  face_lankmark_dict[py::str("score")] = face_lankmark_output->landmarks_score[0];
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

PyKeyPointDetector::PyKeyPointDetector(ModelType model_type,
                               const std::string& model_path,
                               const int device_id)
    : PyModel(model_type, model_path, device_id) {}

py::dict PyKeyPointDetector::inference(const PyImage& image, py::dict parameters) {
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
      ModelOutputType::OBJECT_LANDMARKS) {
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

PySemanticSegmentation::PySemanticSegmentation(ModelType model_type,
                               const std::string& model_path,
                               const int device_id)
    : PyModel(model_type, model_path, device_id) {}

py::dict PySemanticSegmentation::inference(const PyImage& image, py::dict parameters) {
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
      ModelOutputType::SEGMENTATION) {
    throw std::runtime_error("Model output type is not SEGMENTATION");
  }
  std::shared_ptr<ModelSegmentationInfo> segmentation_output =
      std::dynamic_pointer_cast<ModelSegmentationInfo>(output_info);
  if (!segmentation_output) {
    throw std::runtime_error("Failed to cast to ModelSegmentationInfo");
  }
  py::dict segmentation_dict;
  segmentation_dict[py::str("output_width")] = segmentation_output->output_width;
  segmentation_dict[py::str("output_height")] = segmentation_output->output_height;
  py::list class_id;
  py::list class_conf;  
  for (size_t i = 0; i < segmentation_output->output_width*segmentation_output->output_height; ++i) {
    class_id.append(segmentation_output->class_id[i]);
    class_conf.append(segmentation_output->class_conf[i]);
  }
  segmentation_dict[py::str("class_id")] = class_id;
  segmentation_dict[py::str("class_conf")] = class_conf;
  return segmentation_dict;
}

PyInstanceSegmentation::PyInstanceSegmentation(ModelType model_type,
                               const std::string& model_path,
                               const int device_id)
    : PyModel(model_type, model_path, device_id) {}

py::dict PyInstanceSegmentation::inference(const PyImage& image, py::dict parameters) {
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
    throw std::runtime_error("Model output type is not OBJECT_DETECTION_WITH_SEGMENTATION");
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
    box_seg_dict[py::str("class_name")] = object_type_to_string(box_seg_info.object_type);
    box_seg_dict[py::str("x1")] = box_seg_info.x1;
    box_seg_dict[py::str("y1")] = box_seg_info.y1;
    box_seg_dict[py::str("x2")] = box_seg_info.x2;
    box_seg_dict[py::str("y2")] = box_seg_info.y2;
    box_seg_dict[py::str("score")] = box_seg_info.score;
    py::list mask;

    for(int i=0; i<instance_seg_output->mask_width*instance_seg_output->mask_height; i++){
      mask.append(box_seg_info.mask[i]);
      std::cout<<box_seg_info.mask[i]<< std::endl;
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
  if (output_info->getType() !=
      ModelOutputType::OBJECT_LANDMARKS) {
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

PyFeatureExtractor::PyFeatureExtractor(ModelType model_type,
                               const std::string& model_path,
                               const int device_id)
    : PyModel(model_type, model_path, device_id) {}

py::array_t<float> PyFeatureExtractor::inference(const PyImage& image, py::dict parameters) {
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
      ModelOutputType::FEATURE_EMBEDDING) {
    throw std::runtime_error("Model output type is not FEATURE_EMBEDDING");
  }
  std::shared_ptr<ModelFeatureInfo> feature_output =
      std::dynamic_pointer_cast<ModelFeatureInfo>(output_info);
  if (!feature_output) {
    throw std::runtime_error("Failed to cast to ModelFeatureInfo");
  }
  float* feature_ptr = reinterpret_cast<float*>(feature_output->embedding);
  
  return py::array_t<float>(feature_output->embedding_num,feature_ptr);
}
}  // namespace pytdl