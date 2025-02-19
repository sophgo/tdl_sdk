#pragma once

#include "image/base_image.hpp"
#include "model/base_model.hpp"

class FeatureExtraction : public BaseModel {
 public:
  FeatureExtraction();
  ~FeatureExtraction();

  virtual int32_t outputParse(
      const std::vector<std::shared_ptr<BaseImage>>& images,
      std::vector<void*>& out_datas) override;
  virtual int onModelOpened() override;

 private:
};
