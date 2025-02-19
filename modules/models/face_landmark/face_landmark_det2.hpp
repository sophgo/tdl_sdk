#pragma once
#include <bitset>

#include "model/base_model.hpp"

class FaceLandmarkerDet2 final : public BaseModel {
 public:
  FaceLandmarkerDet2();
  ~FaceLandmarkerDet2();

  virtual int32_t outputParse(
      const std::vector<std::shared_ptr<BaseImage>> &images,
      std::vector<void *> &out_datas) override;
  virtual int32_t onModelOpened() override;

 private:
  std::map<std::string, std::string> out_names_;
};
