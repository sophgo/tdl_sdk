#pragma once
#include <cvi_comm_vb.h>
#include "core.hpp"
#include "core/object/cvai_object_types.h"

namespace cviai {

/* WPODNet */
class LicensePlateRecognitionV2 final : public Core {
 public:
  LicensePlateRecognitionV2();

  virtual ~LicensePlateRecognitionV2();
  int inference(VIDEO_FRAME_INFO_S *frame, cvai_object_t *vehicle_meta);
  int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
  void greedy_decode(float *prebs);
  virtual bool allowExportChannelAttribute() const override { return true; }
};
}  // namespace cviai
