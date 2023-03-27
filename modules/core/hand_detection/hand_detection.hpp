#pragma once
#include <bitset>
#include "core.hpp"
#include "core/object/cvai_object_types.h"

namespace cviai {

class HandDetection final : public Core {
 public:
  HandDetection();
  virtual ~HandDetection();
  int inference(VIDEO_FRAME_INFO_S *srcFrame, cvai_object_t *obj_meta);
  virtual bool allowExportChannelAttribute() const override { return true; }

 private:
  int vpssPreprocess(VIDEO_FRAME_INFO_S *srcFrame, VIDEO_FRAME_INFO_S *dstFrame,
                     VPSSConfig &vpss_config);
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
  void outputParser(const int image_width, const int image_height, const int frame_width,
                    const int frame_height, cvai_object_t *obj_meta);

  std::bitset<CVI_AI_DET_TYPE_END> m_filter;
};
}  // namespace cviai
