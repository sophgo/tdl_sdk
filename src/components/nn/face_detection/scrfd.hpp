#ifndef SCRFD_HPP
#define SCRFD_HPP

#include "image/base_image.hpp"
#include "model/base_model.hpp"
class SCRFD : public BaseModel {
 public:
  SCRFD();
  ~SCRFD();

  // TODO:add support for int8 output tensor decode
  virtual int32_t outputParse(
      const std::vector<std::shared_ptr<BaseImage>>& images,
      std::vector<std::shared_ptr<ModelOutputInfo>>& out_datas) override;
  virtual int onModelOpened() override;

 private:
  std::vector<int> m_feat_stride_fpn;
  // std::map<std::string, std::vector<anchor_box>> m_anchors;
  // std::map<std::string, int> m_num_anchors;
  std::map<int, std::vector<std::vector<float>>> fpn_anchors_;
  std::map<int, std::map<std::string, std::string>>
      fpn_out_nodes_;  //{stride:{"box":"xxxx","score":"xxx","landmark":"xxxx"}}
  std::map<int, int> fpn_grid_anchor_num_;
  float iou_threshold_ = 0.5;
};
#endif
