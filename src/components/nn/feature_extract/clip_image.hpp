#pragma once

#include "image/base_image.hpp"
#include "model/base_model.hpp"

class Clip_Image final : public BaseModel {
 public:
  Clip_Image();
  ~Clip_Image();
  virtual int32_t outputParse(
      const std::vector<std::shared_ptr<BaseImage>> &images,
      std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) override;
  virtual int32_t onModelOpened() override;

 private:
};
