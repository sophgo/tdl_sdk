#pragma once

#include "model/base_model.hpp"

class HandKeypointClassification final : public BaseModel {
 public:
  HandKeypointClassification();
  ~HandKeypointClassification();

  virtual int32_t inference(
      const std::vector<std::shared_ptr<BaseImage>> &images,
      std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas,
      const std::map<std::string, float>& parameters = {}) override;

  virtual int32_t outputParse(
      const std::vector<std::shared_ptr<BaseImage>> &images,
      std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) override;
  virtual int32_t onModelOpened() override;

 private:
};

