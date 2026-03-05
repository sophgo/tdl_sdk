#pragma once
#include <bitset>

#include "model/base_model.hpp"

class Stereo final : public BaseModel {
 public:
  Stereo();
  ~Stereo();

  virtual int32_t outputParse(
      const std::vector<std::vector<std::shared_ptr<BaseImage>>>& images,
      std::vector<std::shared_ptr<ModelOutputInfo>>& out_datas) override;
  virtual int32_t onModelOpened() override;

 private:
  int w_;
  int h_;
};
