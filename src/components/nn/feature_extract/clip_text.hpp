#pragma once
#include <bitset>

#include "model/base_model.hpp"

class Clip_Text final : public BaseModel {
 public:
  Clip_Text();
  ~Clip_Text();
  int32_t inference(
      const std::vector<std::shared_ptr<BaseImage>> &images,
      std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas,
      const std::map<std::string, float> &parameters = {}) override;
  virtual int32_t outputParse(
      const std::vector<std::shared_ptr<BaseImage>> &images,
      std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) override;
  virtual int32_t onModelOpened() override;
};
