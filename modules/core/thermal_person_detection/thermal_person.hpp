#pragma once
#ifdef ATHENA2
#include "core_a2.hpp"
#else
#include "core.hpp"
#endif
#include "core/object/cvtdl_object_types.h"

#include "opencv2/opencv.hpp"

namespace cvitdl {

class ThermalPerson final : public Core {
 public:
  ThermalPerson();
  virtual ~ThermalPerson();
  int inference(VIDEO_FRAME_INFO_S *srcFrame, cvtdl_object_t *obj);
  virtual bool allowExportChannelAttribute() const override { return true; }

 private:
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
  void outputParser(const int image_width, const int image_height, const int frame_width,
                    const int frame_height, cvtdl_object_t *obj);
};
}  // namespace cvitdl