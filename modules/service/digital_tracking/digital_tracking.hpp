#pragma once
#include "core/face/cvai_face_types.h"
#include "core/object/cvai_object_types.h"
#include "vpss_engine.hpp"

namespace cviai {
namespace service {

struct Rect {
  Rect() {}
  Rect(float l, float r, float t, float b) : l(l), r(r), t(t), b(b) {}

  void add_padding(float ratio) {
    float w = (r - l) * ratio;
    float h = (t - b) * ratio;
    l -= w;
    r += w;
    t -= h;
    b += h;
  }

  bool is_valid() { return (l != -1); }

  float l = -1;
  float r = -1;
  float t = -1;
  float b = -1;
};

class DigitalTracking {
 public:
  int setVpssEngine(VpssEngine *engine);
  template <typename T>
  int run(const VIDEO_FRAME_INFO_S *srcFrame, const T *meta, VIDEO_FRAME_INFO_S *dstFrame,
          const float face_skip_ratio = 0.05f, const float trans_ratio = 0.1f);

 private:
  inline void transformRect(const float trans_ratio, const Rect &prev_rect, Rect *curr_rect);
  inline void fitRatio(const float width, const float height, Rect *rect);
  inline void fitFrame(const float width, const float height, Rect *rect);

  float m_padding_ratio = 0.3f;
  Rect m_prev_rect;
  VpssEngine *mp_vpss_inst = nullptr;
};
}  // namespace service
}  // namespace cviai
