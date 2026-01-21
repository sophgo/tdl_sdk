#pragma once
#include <bitset>
#include <map>
#include <string>
#include <vector>

#include "model/base_model.hpp"

class Yolo26Detection final : public BaseModel {
 public:
  Yolo26Detection(const int num_cls = 0);
  Yolo26Detection(std::pair<int, int> yolov8_pair);
  ~Yolo26Detection();

  virtual int32_t outputParse(
      const std::vector<std::shared_ptr<BaseImage>> &images,
      std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) override;
  virtual int32_t onModelOpened() override;

 private:
  void makeAnchors(int input_h, int input_w);

  void getAllClassScores(int batch_idx, std::vector<float> &all_scores);

  void getAllRegValues(int batch_idx, std::vector<float> &all_reg);

  void getTopkIndex(const std::vector<float> &scores, int num_anchors,
                    int num_cls, int max_det, std::vector<float> &top_scores,
                    std::vector<int> &top_cls_idx,
                    std::vector<int> &top_anchor_idx);

  void dist2bbox(const float *reg_vals, int anchor_idx,
                 std::vector<float> &bbox);

  // 输出分支信息
  std::vector<int> strides;
  std::map<int, std::string> class_out_names;
  std::map<int, std::string> bbox_out_names;

  // 锚点信息
  std::vector<float> anchor_points_x_;  // 锚点 x 坐标 (total_anchors,)
  std::vector<float> anchor_points_y_;  // 锚点 y 坐标 (total_anchors,)
  std::vector<float> stride_tensor_;  // 每个锚点对应的步长 (total_anchors,)
  int total_anchors_ = 0;

  // 每个 stride 对应的锚点起始索引和数量
  std::map<int, int> stride_anchor_start_;  // stride -> 起始索引
  std::map<int, int> stride_anchor_count_;  // stride -> 锚点数量

  int num_box_channel_ = 4;  // yolo26 输出 4 通道距离值 (ltrb)
  int num_cls_ = 0;
  int max_det_ = 300;  // 最大检测数

  // 官方代码推理是设置model_threshold_为0.25
};
