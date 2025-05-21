#pragma once
#include <bitset>

#include "model/base_model.hpp"

class LicensePlateRecognition final : public BaseModel {
 public:
  LicensePlateRecognition();
  std::string greedy_decode(float *prebs);
  virtual int32_t outputParse(
      const std::vector<std::shared_ptr<BaseImage>> &images,
      std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) override;
};
