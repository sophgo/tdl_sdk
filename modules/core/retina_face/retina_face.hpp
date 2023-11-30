#pragma once
#ifdef ATHENA2
#include "core_a2.hpp"
#else
#include "core.hpp"
#endif
#include "core/face/cvtdl_face_types.h"

#include "anchor_generator.h"

namespace cvitdl {

class RetinaFace final : public Core {
 public:
  RetinaFace(PROCESS process);
  virtual ~RetinaFace();
  int inference(VIDEO_FRAME_INFO_S *srcFrame, cvtdl_face_t *meta);
  virtual bool allowExportChannelAttribute() const override { return true; }
  virtual void setModelThreshold(float threshold) override;

 private:
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
  virtual int onModelOpened() override;
  void outputParser(int image_width, int image_height, int frame_width, int frame_height,
                    cvtdl_face_t *meta);
  std::vector<int> m_feat_stride_fpn;
  std::map<std::string, std::vector<anchor_box>> m_anchors;
  std::map<std::string, int> m_num_anchors;
  PROCESS process;
};
}  // namespace cvitdl