#ifndef TDL_FRAMEWORK_COMMON_MODEL_OUTPUT_TYPES_HPP
#define TDL_FRAMEWORK_COMMON_MODEL_OUTPUT_TYPES_HPP

#include <map>
#include <vector>
#include "common/common_types.hpp"
#include "common/object_type_def.hpp"
enum class ModelOutputType {
  OBJECT_DETECTION,
  OBJECT_DETECTION_WITH_LANDMARKS,
  OBJECT_DETECTION_WITH_SEGMENTATION,
  OBJECT_LANDMARKS,
  FEATURE_EMBEDDING,
  CLASSIFICATION,
  ATTRIBUTE,
  SEGMENTATION,
  UNKOWN
};

class ModelOutputInfo {
 public:
  virtual ~ModelOutputInfo() = default;
  virtual ModelOutputType getType() const = 0;
};
class ObjectBoxInfo {
 public:
  ObjectBoxInfo() = default;
  ObjectBoxInfo(int32_t class_id, float score, float x1, float y1, float x2,
                float y2)
      : class_id(class_id), score(score), x1(x1), y1(y1), x2(x2), y2(y2) {}

  int32_t class_id;
  TDLObjectType object_type = OBJECT_TYPE_UNDEFINED;
  float score;
  float x1, y1, x2, y2;
};

class ModelBoxInfo : public ModelOutputInfo {
 public:
  ModelBoxInfo() = default;
  ~ModelBoxInfo() = default;
  ModelOutputType getType() const override {
    return ModelOutputType::OBJECT_DETECTION;
  }

  uint32_t image_width;
  uint32_t image_height;
  std::vector<ObjectBoxInfo> bboxes;
};

class ObjectBoxLandmarkInfo {
 public:
  ObjectBoxLandmarkInfo() = default;

  int32_t class_id;
  TDLObjectType object_type = OBJECT_TYPE_UNDEFINED;
  float score;
  float x1, y1, x2, y2;
  std::vector<float> landmarks_x;
  std::vector<float> landmarks_y;
  std::vector<float> landmarks_score;
};

class ObjectBoxSegmentationInfo {
 public:
  ObjectBoxSegmentationInfo()     
      : class_id(0),
      object_type(OBJECT_TYPE_UNDEFINED),
      score(0.0f),
      x1(0.0f),
      y1(0.0f),
      x2(0.0f),
      y2(0.0f),
      mask(nullptr),
      mask_point(nullptr),
      mask_point_size(0) {}
  ~ObjectBoxSegmentationInfo() {
      if (mask != nullptr) {
          delete[] mask;          
          mask = nullptr;         
      }
      if (mask_point != nullptr) {
          delete[] mask_point;   
          mask_point = nullptr;   
      }
  }
  int32_t class_id;
  TDLObjectType object_type = OBJECT_TYPE_UNDEFINED;
  float score;
  float x1, y1, x2, y2;
  uint8_t *mask;
  float *mask_point;
  uint32_t mask_point_size;
};
class ModelBoxSegmentationInfo : public ModelOutputInfo {
 public:
  ~ModelBoxSegmentationInfo() = default;
  ModelOutputType getType() const override {
    return ModelOutputType::OBJECT_DETECTION_WITH_SEGMENTATION;
  }
  uint32_t image_width;
  uint32_t image_height;
  uint32_t mask_width;
  uint32_t mask_height;
  std::vector<ObjectBoxSegmentationInfo> box_seg;
};

class ModelBoxLandmarkInfo : public ModelOutputInfo {
 public:
  ~ModelBoxLandmarkInfo() = default;
  ModelOutputType getType() const override {
    return ModelOutputType::OBJECT_DETECTION_WITH_LANDMARKS;
  }
  uint32_t image_width;
  uint32_t image_height;
  std::vector<ObjectBoxLandmarkInfo> box_landmarks;
};

class ModelLandmarksInfo : public ModelOutputInfo {
 public:
  ~ModelLandmarksInfo() = default;
  ModelOutputType getType() const override {
    return ModelOutputType::OBJECT_LANDMARKS;
  }
  uint32_t image_width;
  uint32_t image_height;
  TDLObjectType object_type = OBJECT_TYPE_UNDEFINED;
  std::vector<float> landmarks_x;
  std::vector<float> landmarks_y;
  std::vector<float> landmarks_score;
  std::map<TDLObjectAttributeType, float> attributes;
};

class ModelFeatureInfo : public ModelOutputInfo {
 public:
  ~ModelFeatureInfo();
  ModelOutputType getType() const override {
    return ModelOutputType::FEATURE_EMBEDDING;
  }
  uint8_t* embedding = nullptr;
  int32_t embedding_num;
  TDLDataType embedding_type;
};

class ModelClipFeatureInfo : public ModelOutputInfo {
 public:
  ~ModelClipFeatureInfo();
  ModelOutputType getType() const override {
    return ModelOutputType::FEATURE_EMBEDDING;
  }
  float* embedding = nullptr;
  int32_t embedding_num;
  TDLDataType embedding_type;
};

class ModelClassificationInfo : public ModelOutputInfo {
 public:
  ~ModelClassificationInfo() = default;
  ModelOutputType getType() const override {
    return ModelOutputType::CLASSIFICATION;
  }
  std::vector<float> topk_scores;
  std::vector<int32_t> topk_class_ids;
  std::vector<TDLObjectType> topk_object_types;
};

class ModelAttributeInfo : public ModelOutputInfo {
 public:
  ~ModelAttributeInfo() = default;
  ModelOutputType getType() const override {
    return ModelOutputType::ATTRIBUTE;
  }
  std::map<TDLObjectAttributeType, float> attributes;
};

class ModelSegmentationInfo : public ModelOutputInfo {
 public:
  ~ModelSegmentationInfo() = default;
  ModelOutputType getType() const override {
    return ModelOutputType::SEGMENTATION;
  }
};

#endif
