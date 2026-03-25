#pragma once

#include <vector>

#include "model/base_model.hpp"

class TopformerSegMotion final : public BaseModel {
 public:
  TopformerSegMotion();
  ~TopformerSegMotion();

  virtual int32_t inference(
      const std::shared_ptr<BaseImage>& image,
      std::shared_ptr<ModelOutputInfo>& out_data,
      const std::map<std::string, float>& parameters = {}) override;

  virtual int32_t outputParse(
      const std::shared_ptr<BaseImage>& image,
      std::shared_ptr<ModelOutputInfo>& out_data) override;

 private:
  void* ccl_instance_;
  std::vector<int8_t> cached_input_0_;
  std::vector<int8_t> cached_input_1_;
  int cached_frame_count_;
  int min_area_thresh_;
  bool with_mask_;
};
