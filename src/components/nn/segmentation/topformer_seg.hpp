#pragma once
#include <bitset>

#include "model/base_model.hpp"

class TopformerSeg final : public BaseModel {
 public:
  TopformerSeg();
  TopformerSeg(int down_rato);
  ~TopformerSeg();

  virtual int32_t outputParse(
      const std::vector<std::shared_ptr<BaseImage>> &images,
      std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) override;

 private:

  int oriW, oriH;
  int outW, outH;
  int preW, preH;
  int outShapeH, outShapeW;
  int downRato;
};
