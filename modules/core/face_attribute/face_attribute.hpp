#pragma once
#include <cvi_comm_vb.h>
#include <cvi_sys.h>
#ifdef ATHENA2
#include "core_a2.hpp"
#else
#include "core.hpp"
#endif
#include "core/face/cvtdl_face_types.h"

namespace cvitdl {

class FaceAttribute final : public Core {
 public:
  explicit FaceAttribute(bool with_attribute);
  virtual ~FaceAttribute();
  int inference(VIDEO_FRAME_INFO_S *stOutFrame, cvtdl_face_t *meta, int face_idx = -1);
  void setHardwareGDC(bool use_wrap_hw);
  int extract_face_feature(const uint8_t *p_rgb_pack, uint32_t width, uint32_t height,
                           uint32_t stride, cvtdl_face_info_t *p_face_info);

 private:
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
  virtual int onModelOpened() override;
  virtual int onModelClosed() override;
  void outputParser(cvtdl_face_info_t *face_info);
  int dump_bgr_pack(const char *p_img_file, VIDEO_FRAME_INFO_S *p_img_frm);
  CVI_S32 allocateION();
  void releaseION();

  bool m_use_wrap_hw;
  const bool m_with_attribute;
  float *attribute_buffer = nullptr;
  VIDEO_FRAME_INFO_S m_wrap_frame;
};
}  // namespace cvitdl