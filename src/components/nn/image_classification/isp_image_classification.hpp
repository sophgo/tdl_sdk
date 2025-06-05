#ifndef ISP_IMAGE_CLASSIFICATION_HPP
#define ISP_IMAGE_CLASSIFICATION_HPP

#include "model/base_model.hpp"

class IspImageClassification final : public BaseModel {
 public:
  IspImageClassification();
  ~IspImageClassification();
  int32_t inference(
      const std::vector<std::shared_ptr<BaseImage>> &images,
      std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas,
      const std::map<std::string, float> &parameters = {}) override;
  virtual int32_t outputParse(
      const std::vector<std::shared_ptr<BaseImage>> &images,
      std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) override;
  virtual int32_t onModelOpened() override;
};

#endif