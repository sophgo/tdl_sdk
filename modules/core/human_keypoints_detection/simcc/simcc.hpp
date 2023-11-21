#pragma once
#include "core.hpp"
#include "core/face/cvtdl_face_types.h"

namespace cvitdl {

class Simcc final : public Core {
 public:
  Simcc();
  virtual ~Simcc();
  int inference(VIDEO_FRAME_INFO_S *stOutFrame, cvtdl_object_t *obj_meta);

 private:
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
  void outputParser(const float nn_width, const float nn_height, const int frame_width,
                    const int frame_height, cvtdl_object_t *obj, std::vector<float> &box,
                    int index);
};
}  // namespace cvitdl