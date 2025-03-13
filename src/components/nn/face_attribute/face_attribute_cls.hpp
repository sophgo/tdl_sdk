#pragma once
#include <bitset>

#include "model/base_model.hpp"

class FaceAttribute_CLS final : public BaseModel {
 public:
  FaceAttribute_CLS();
  ~FaceAttribute_CLS();

  virtual int32_t outputParse(
      const std::vector<std::shared_ptr<BaseImage>>& images,
      std::vector<std::shared_ptr<ModelOutputInfo>>& out_datas) override;
  virtual int32_t inference(
      const std::shared_ptr<BaseImage>& image,
      const std::shared_ptr<ModelOutputInfo>& model_object_infos,
      std::vector<std::shared_ptr<ModelOutputInfo>>& out_datas,
      const std::map<std::string, float>& parameters = {}) override;
  virtual int32_t onModelOpened() override;

 private:
  std::string gender_name;
  std::string age_name;
  std::string glass_name;
  std::string mask_name;
};
