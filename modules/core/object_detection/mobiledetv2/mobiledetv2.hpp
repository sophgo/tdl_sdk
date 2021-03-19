#pragma once
#include <bitset>
#include <limits>
#include <memory>
#include <vector>

#include "anchors.hpp"
#include "core.hpp"
#include "core/object/cvai_object_types.h"

namespace cviai {

class MobileDetV2 final : public Core {
 public:
  // TODO: remove duplicate struct
  struct object_detect_rect_t {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
  };

  // TODO: define in common header
  typedef std::shared_ptr<object_detect_rect_t> PtrDectRect;
  typedef std::vector<PtrDectRect> Detections;

  enum class Model {
    d0,             // MobileDetV2-D0
    d1,             // MobileDetV2-D1
    d2,             // MobileDetV2-D2
    lite,           // MobileDetV2-Lite
    vehicle_d0,     // MobileDetV2-Vehicle-D0
    pedestrian_d0,  // MobileDetV2-Pedestrian-D0
  };

  struct CvimodelInfo {
    MobileDetV2::Model model;
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
    static CvimodelInfo create_config(MobileDetV2::Model model);
  };

  explicit MobileDetV2(MobileDetV2::Model model, float iou_thresh = 0.5);
  virtual ~MobileDetV2();
  int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
  int inference(VIDEO_FRAME_INFO_S *frame, cvai_object_t *meta, cvai_obj_det_type_e det_type);
  virtual void setModelThreshold(float threshold) override;
  virtual bool allowExportChannelAttribute() const override { return true; }
  virtual int onModelOpened() override;
  void select_classes(const std::vector<uint32_t> &selected_classes);

 private:
  int vpssPreprocess(const std::vector<VIDEO_FRAME_INFO_S *> &srcFrames,
                     std::vector<std::shared_ptr<VIDEO_FRAME_INFO_S>> *dstFrames) override;
  void get_raw_outputs(std::vector<std::pair<int8_t *, size_t>> *cls_tensor_ptr,
                       std::vector<std::pair<int8_t *, size_t>> *objectness_tensor_ptr,
                       std::vector<std::pair<int8_t *, size_t>> *bbox_tensor_ptr);
  void generate_dets_for_each_stride(Detections *det_vec);
  void generate_dets_for_tensor(Detections *det_vec, float class_dequant_thresh,
                                float bbox_dequant_thresh, int8_t quant_thresh,
                                const int8_t *logits, const int8_t *objectness, int8_t *bboxes,
                                size_t class_tensor_size, const std::vector<AnchorBox> &anchors);
  void filter_dets(Detections &dets);

  std::vector<std::vector<AnchorBox>> m_anchors;
  CvimodelInfo m_model_config;
  float m_iou_threshold;

  // score threshold for quantized inverse threshold
  std::vector<int8_t> m_quant_inverse_score_threshold;
  std::bitset<CVI_AI_DET_TYPE_END> m_filter;
};

}  // namespace cviai
