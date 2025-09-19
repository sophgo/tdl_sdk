#pragma once
#include <bitset>

#include "model/base_model.hpp"

enum class FaceAttributeModel {
  GENDER_AGE_GLASS = 0,
  GENDER_AGE_GLASS_MASK = 1,
  GENDER_AGE_GLASS_EMOTION = 2,
};

class FaceAttribute_CLS final : public BaseModel {
 public:
  FaceAttribute_CLS(FaceAttributeModel model_name);
  ~FaceAttribute_CLS();

  virtual int32_t outputParse(
      const std::vector<std::shared_ptr<BaseImage>>& images,
      std::vector<std::shared_ptr<ModelOutputInfo>>& out_datas) override;
  int32_t outputParseGenderAgeGlass(
      const std::vector<std::shared_ptr<BaseImage>>& images,
      std::vector<std::shared_ptr<ModelOutputInfo>>& out_datas);
  int32_t outputParseGenderAgeGlassMask(
      const std::vector<std::shared_ptr<BaseImage>>& images,
      std::vector<std::shared_ptr<ModelOutputInfo>>& out_datas);
  int32_t outputParseGenderAgeGlassEmotion(
      const std::vector<std::shared_ptr<BaseImage>>& images,
      std::vector<std::shared_ptr<ModelOutputInfo>>& out_datas);

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
  std::string emotion_name;
  FaceAttributeModel face_attribute_model_;
};
