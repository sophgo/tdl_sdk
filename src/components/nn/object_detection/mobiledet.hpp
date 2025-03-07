#pragma once
#include <bitset>
#include <limits>
#include <memory>
#include <vector>

#include "anchors.hpp"
#include "model/base_model.hpp"

class MobileDetV2Detection final : public BaseModel {
 public:
  enum class Category {
    coco80,          // COCO 80 classes
    vehicle,         // CAR, TRUCK, MOTOCYCLE
    pedestrian,      // Pedestrian
    person_pets,     // PERSON, DOG, and CAT
    person_vehicle,  // PERSON, CAR, BICYCLE, MOTOCYCLE, BUS, TRUCK
  };

  struct CvimodelInfo {
    MobileDetV2Detection::Category category;
    size_t min_level;
    size_t max_level;
    size_t num_scales;
    std::vector<std::pair<float, float>> aspect_ratios;
    float anchor_scale;

    size_t image_width;
    size_t image_height;
    int num_classes;
    std::vector<int> strides;
    std::map<int, std::string> class_out_names;
    std::map<int, std::string> bbox_out_names;
    std::map<int, std::string> obj_max_names;
    std::map<int, float> class_dequant_thresh;
    std::map<int, float> bbox_dequant_thresh;
    float default_score_threshold;

    typedef int (*ClassMapFunc)(int);
    ClassMapFunc class_id_map;
    static CvimodelInfo create_config(MobileDetV2Detection::Category model);
  };

  MobileDetV2Detection();
  explicit MobileDetV2Detection(MobileDetV2Detection::Category category,
                                float iou_thresh = 0.5);
  ~MobileDetV2Detection();

  int32_t outputParse(
      const std::vector<std::shared_ptr<BaseImage>>& images,
      std::vector<std::shared_ptr<ModelOutputInfo>>& out_datas) override;
  int32_t onModelOpened() override;

  void setModelThreshold(const float& threshold);

 private:
  void get_raw_outputs(
      std::vector<std::pair<int8_t*, size_t>>* cls_tensor_ptr,
      std::vector<std::pair<int8_t*, size_t>>* objectness_tensor_ptr,
      std::vector<std::pair<int8_t*, size_t>>* bbox_tensor_ptr);
  void generate_dets_for_each_stride(
      std::map<int, std::vector<ObjectBoxInfo>>& det_vec);
  void generate_dets_for_tensor(
      std::map<int, std::vector<ObjectBoxInfo>>& det_vec,
      float class_dequant_thresh, float bbox_dequant_thresh,
      int8_t quant_thresh, const int8_t* logits, const int8_t* objectness,
      int8_t* bboxes, size_t class_tensor_size,
      const std::vector<AnchorBox>& anchors);

  std::vector<std::vector<AnchorBox>> m_anchors;
  CvimodelInfo m_model_config;
  float m_iou_threshold;
  float m_model_threshold = 0.5;

  std::vector<int8_t> m_quant_inverse_score_threshold;
  // std::bitset<CVI_TDL_DET_TYPE_END> m_filter;
};