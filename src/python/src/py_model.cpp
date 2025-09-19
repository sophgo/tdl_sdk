#include "py_model.hpp"
#include <pybind11/stl.h>
#include "py_image.hpp"
#include "py_utils.hpp"
namespace pytdl {

PyModel::PyModel(std::shared_ptr<BaseModel>& model) : model_(model) {}

py::list PyModel::inference(const PyImage& image) {
  std::vector<std::shared_ptr<BaseImage>> images;
  images.push_back(image.getImage());
  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  model_->inference(images, out_datas);
  return outputParse(out_datas);
}

py::list PyModel::inference(
    const py::array_t<unsigned char, py::array::c_style>& input) {
  PyImage image = PyImage::fromNumpy(input);
  return inference(image);
}

py::list PyModel::outputParse(
    const std::vector<std::shared_ptr<ModelOutputInfo>>& out_datas) {
  std::shared_ptr<ModelOutputInfo> output_info = out_datas[0];
  if (output_info->getType() == ModelOutputType::OBJECT_DETECTION) {
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
  } else if (output_info->getType() ==
             ModelOutputType::OBJECT_DETECTION_WITH_LANDMARKS) {
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
  } else if (output_info->getType() == ModelOutputType::CLASSIFICATION) {
    std::shared_ptr<ModelClassificationInfo> classification_output =
        std::dynamic_pointer_cast<ModelClassificationInfo>(output_info);
    if (!classification_output) {
      throw std::runtime_error("Failed to cast to ModelClassificationInfo");
    }
    py::dict classification_dict;

    classification_dict[py::str("class_id")] =
        classification_output->topk_class_ids[0];
    classification_dict[py::str("score")] =
        classification_output->topk_scores[0];

    py::list result;
    result.append(classification_dict);
    return result;
  } else if (output_info->getType() == ModelOutputType::CLS_ATTRIBUTE) {
    std::shared_ptr<ModelAttributeInfo> attribute_output =
        std::dynamic_pointer_cast<ModelAttributeInfo>(output_info);
    if (!attribute_output) {
      throw std::runtime_error("Failed to cast to ModelAttributeInfo");
    }
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
    py::list result;
    result.append(face_attribute_dict);
    return result;
  } else if (output_info->getType() == ModelOutputType::OBJECT_LANDMARKS) {
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
    py::list result;
    result.append(landmark_dict);
    return result;

  } else if (output_info->getType() == ModelOutputType::SEGMENTATION) {
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
    py::list result;
    result.append(segmentation_dict);
    return result;
  } else if (output_info->getType() ==
             ModelOutputType::OBJECT_DETECTION_WITH_SEGMENTATION) {
    std::shared_ptr<ModelBoxSegmentationInfo> instance_seg_output =
        std::dynamic_pointer_cast<ModelBoxSegmentationInfo>(output_info);
    if (!instance_seg_output) {
      throw std::runtime_error("Failed to cast to ModelBoxSegmentationInfo");
    }
    py::dict instance_seg_dict;
    instance_seg_dict[py::str("mask_width")] = instance_seg_output->mask_width;
    instance_seg_dict[py::str("mask_height")] =
        instance_seg_output->mask_height;
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

      for (int i = 0; i < instance_seg_output->mask_width *
                              instance_seg_output->mask_height;
           i++) {
        mask.append(box_seg_info.mask[i]);
        std::cout << box_seg_info.mask[i] << std::endl;
      }
      box_seg_dict[py::str("mask")] = mask;
      // box_seg_dict[py::str("mask_point_size")] =
      // box_seg_info.mask_point_size; py::list mask_points; for(int i; i<
      // box_seg_info.mask_point_size; i++){
      //   py::list mask_point;
      //   mask_point.append(box_seg_info.mask_point[2*i]);
      //   mask_point.append(box_seg_info.mask_point[2*i+1]);
      //   mask_points.append(mask_point);
      // }
      bboxes_seg.append(box_seg_dict);
    }
    instance_seg_dict[py::str("bboxes_seg")] = bboxes_seg;
    py::list result;
    result.append(instance_seg_dict);
    return result;
  } else if (output_info->getType() == ModelOutputType::FEATURE_EMBEDDING) {
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
        float* feature_ptr =
            reinterpret_cast<float*>(feature_output->embedding);
        return py::array_t<float>({size}, {sizeof(float)}, feature_ptr);
      }
      default:
        assert(false && "Unsupported embedding_type");
        return py::array_t<float>();
    }
  } else if (output_info->getType() == ModelOutputType::OCR_INFO) {
    std::shared_ptr<ModelOcrInfo> char_output =
        std::dynamic_pointer_cast<ModelOcrInfo>(output_info);
    if (!char_output) {
      throw std::runtime_error("Failed to cast to ModelOcrInfo");
    }
    py::list char_list;
    std::string str(char_output->text_info);
    char_list.append(str);
    return char_list;
  } else if (output_info->getType() ==
             ModelOutputType::OBJECT_DETECTION_WITH_LANDMARKS) {
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
    py::list result;
    result.append(lanes_list);
    return result;
  } else {
    throw std::runtime_error("Model output type is not supported");
  }
}

void getModelConfig(const py::dict& model_config,
                    ModelConfig& model_config_cpp) {
  if (model_config.contains("mean") && model_config.contains("scale")) {
    model_config_cpp.mean.resize(3);
    model_config_cpp.std.resize(3);
    auto mean = model_config["mean"].cast<py::tuple>();
    auto scale = model_config["scale"].cast<py::tuple>();
    model_config_cpp.mean[0] = mean[0].cast<float>();
    model_config_cpp.mean[1] = mean[1].cast<float>();
    model_config_cpp.mean[2] = mean[2].cast<float>();
    model_config_cpp.std[0] = scale[0].cast<float>();
    model_config_cpp.std[1] = scale[1].cast<float>();
    model_config_cpp.std[2] = scale[2].cast<float>();
    if (model_config.contains("rgb_order")) {
      model_config_cpp.rgb_order =
          model_config["rgb_order"].cast<std::string>();
    }
    if (model_config.contains("types")) {
      model_config_cpp.types =
          model_config["types"].cast<std::vector<std::string>>();
    }
    if (model_config.contains("net_name")) {
      model_config_cpp.net_name = model_config["net_name"].cast<std::string>();
    }
    if (model_config.contains("file_name")) {
      model_config_cpp.file_name =
          model_config["file_name"].cast<std::string>();
    }
    if (model_config.contains("comment")) {
      model_config_cpp.comment = model_config["comment"].cast<std::string>();
    }
    if (model_config.contains("custom_config_str")) {
      model_config_cpp.custom_config_str =
          model_config["custom_config_str"]
              .cast<std::map<std::string, std::string>>();
    }
  } else {
    model_config_cpp = {};
  }
}

py::dict PyModel::getPreprocessParameters() {
  PreprocessParams pre_params;
  model_->getPreprocessParameters(pre_params);
  py::dict model_config;
  model_config["mean"] = py::make_tuple(pre_params.mean[0], pre_params.mean[1],
                                        pre_params.mean[2]);
  model_config["scale"] = py::make_tuple(
      pre_params.scale[0], pre_params.scale[1], pre_params.scale[2]);
  return model_config;
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

PyModel get_model_from_dir(const ModelType model_type,
                           const std::string& model_dir, const int device_id) {
  TDLModelFactory& model_factory = TDLModelFactory::getInstance();
  model_factory.setModelDir(model_dir);
  auto model = model_factory.getModel(model_type, device_id);
  if (model == nullptr) {
    throw std::runtime_error("Failed to create model");
  }
  return PyModel(model);
};

PyModel get_model(ModelType model_type, const std::string& model_path,
                  const py::dict& model_config, const int device_id) {
  ModelConfig model_config_cpp;
  std::shared_ptr<BaseModel> model;
  if (model_config.empty()) {
    model = TDLModelFactory::getInstance().getModel(model_type, model_path,
                                                    device_id);
  } else {
    getModelConfig(model_config, model_config_cpp);
    model = TDLModelFactory::getInstance().getModel(
        model_type, model_path, model_config_cpp, device_id);
  }
  if (model == nullptr) {
    throw std::runtime_error("Failed to create model");
  }
  return PyModel(model);
}
}  // namespace pytdl