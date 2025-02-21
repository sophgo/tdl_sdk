#ifndef RGB_IMAGE_CLASSIFICATION_HPP
#define RGB_IMAGE_CLASSIFICATION_HPP

#include "model/base_model.hpp"

class RgbImageClassification final : public BaseModel {
 public:
  RgbImageClassification();
  ~RgbImageClassification();

  virtual int32_t outputParse(
      const std::vector<std::shared_ptr<BaseImage>> &images,
      std::vector<void *> &out_datas) override;
  virtual int32_t onModelOpened() override;

 private:
};

#endif
