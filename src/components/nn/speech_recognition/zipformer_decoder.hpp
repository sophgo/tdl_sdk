#ifndef ZIPFORMER_DECODER_HPP
#define ZIPFORMER_DECODER_HPP
#include "model/base_model.hpp"

class ZipformerDecoder : public BaseModel {
 public:
  ZipformerDecoder();
  virtual ~ZipformerDecoder();
  int32_t setupNetwork(NetParam& net_param) override;
  virtual int32_t inference(
      const std::shared_ptr<BaseImage>& image,
      std::shared_ptr<ModelOutputInfo>& out_data,
      const std::map<std::string, float>& parameters = {}) override;
  virtual int32_t outputParse(
      const std::shared_ptr<BaseImage>& image,
      std::shared_ptr<ModelOutputInfo>& out_data) override;
  virtual int32_t onModelOpened() override;
};

#endif
