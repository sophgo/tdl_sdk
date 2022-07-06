#pragma once
#include <bitset>
#include "core.hpp"
#include "core/object/cvai_object_types.h"

namespace cviai {

class YoloX final : public Core {
 public:
  YoloX();
  virtual ~YoloX();
  int inference(VIDEO_FRAME_INFO_S *srcFrame, cvai_object_t *obj_meta);
  virtual bool allowExportChannelAttribute() const override { return true; }
  void select_classes(const std::vector<uint32_t> &selected_classes);

 private:
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
  void outputParser(const int image_width, const int image_height, const int frame_width,
                    const int frame_height, cvai_object_t *obj_meta);

  std::bitset<CVI_AI_DET_TYPE_END> m_filter;
};
}  // namespace cviai