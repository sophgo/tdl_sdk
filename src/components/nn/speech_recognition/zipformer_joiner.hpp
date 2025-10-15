#ifndef ZIPFORMER_JOINER_HPP
#define ZIPFORMER_JOINER_HPP
#include "model/base_model.hpp"

class ZipformerJoiner : public BaseModel {
 public:
  ZipformerJoiner();
  virtual ~ZipformerJoiner();
  int32_t setupNetwork(NetParam& net_param) override;
  virtual int32_t inference(
      const std::shared_ptr<BaseImage>& image,
      std::shared_ptr<ModelOutputInfo>& out_data,
      const std::map<std::string, float>& parameters = {}) override;
  virtual int32_t outputParse(
      const std::shared_ptr<BaseImage>& image,
      std::shared_ptr<ModelOutputInfo>& out_data) override;
  virtual int32_t onModelOpened() override;

 private:
  int feature_size_ = 0;
};

#endif
