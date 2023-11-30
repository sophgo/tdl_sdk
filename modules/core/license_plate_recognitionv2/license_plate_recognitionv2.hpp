#pragma once
#include <cvi_comm_vb.h>
#include <cvi_sys.h>
#ifdef ATHENA2
#include "core_a2.hpp"
#else
#include "core.hpp"
#endif
#include "core/object/cvtdl_object_types.h"

namespace cvitdl {

/* WPODNet */
class LicensePlateRecognitionV2 final : public Core {
 public:
  LicensePlateRecognitionV2();

  virtual ~LicensePlateRecognitionV2();
  int inference(VIDEO_FRAME_INFO_S *frame, cvtdl_object_t *vehicle_meta);
  int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
  void greedy_decode(float *prebs);
  virtual bool allowExportChannelAttribute() const override { return true; }
};
}  // namespace cvitdl
