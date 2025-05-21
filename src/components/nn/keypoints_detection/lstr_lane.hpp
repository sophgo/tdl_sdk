#pragma once
#include <bitset>

#include "model/base_model.hpp"

class LstrLane final : public BaseModel {
 public:
  LstrLane();
  ~LstrLane();

  virtual int32_t outputParse(
      const std::vector<std::shared_ptr<BaseImage>> &images,
      std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) override;

 private:
  float gen_x_by_y(float ys, std::vector<float> &point_line);
};
