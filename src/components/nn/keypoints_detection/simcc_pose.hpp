#pragma once
#include <bitset>

#include "model/base_model.hpp"

class SimccPose final : public BaseModel {
 public:
  SimccPose();
  ~SimccPose();
  virtual int32_t outputParse(
      const std::vector<std::shared_ptr<BaseImage>> &images,
      std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) override;

 private:
};
