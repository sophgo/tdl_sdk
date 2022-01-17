#pragma once
#include "core.hpp"
#include "core/object/cvai_object_types.h"

#include "opencv2/opencv.hpp"

namespace cviai {

class ThermalPerson final : public Core {
 public:
  ThermalPerson();
  virtual ~ThermalPerson();
  int inference(VIDEO_FRAME_INFO_S *srcFrame, cvai_object_t *obj);
  virtual bool allowExportChannelAttribute() const override { return true; }

 private:
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
  void outputParser(const int image_width, const int image_height, const int frame_width,
                    const int frame_height, cvai_object_t *obj);
};
}  // namespace cviai