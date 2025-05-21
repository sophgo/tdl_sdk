#pragma once
#include <bitset>

#include "model/base_model.hpp"

class FaceLandmarkerDet2 final : public BaseModel {
 public:
  FaceLandmarkerDet2();
  ~FaceLandmarkerDet2();

  virtual int32_t inference(
      const std::shared_ptr<BaseImage>& image,
      const std::shared_ptr<ModelOutputInfo>& model_object_infos,
      std::vector<std::shared_ptr<ModelOutputInfo>>& out_datas,
      const std::map<std::string, float>& parameters = {}) override;

  virtual int32_t outputParse(
      const std::vector<std::shared_ptr<BaseImage>>& images,
      std::vector<std::shared_ptr<ModelOutputInfo>>& out_datas) override;
  virtual int32_t onModelOpened() override;

 private:
  std::map<std::string, std::string> out_names_;
};
