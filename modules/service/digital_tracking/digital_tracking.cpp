#include "digital_tracking.hpp"
#include "core/utils/vpss_helper.h"
#include "cviai_log.hpp"

#include <algorithm>

#define DEFAULT_DT_ZOOM_TRANS_RATIO 0.1f

namespace cviai {
namespace service {

int DigitalTracking::setVpssEngine(VpssEngine *engine) {
  mp_vpss_inst = engine;
  return CVI_SUCCESS;
}

template <typename T>
int DigitalTracking::run(const VIDEO_FRAME_INFO_S *srcFrame, const T *meta,
                         VIDEO_FRAME_INFO_S *dstFrame, const float face_skip_ratio,
                         const float trans_ratio) {
  if (mp_vpss_inst == nullptr) {
    LOGE("vpss_inst not set.\n");
    return CVI_FAILURE;
  }
  uint32_t width = srcFrame->stVFrame.u32Width;
  uint32_t height = srcFrame->stVFrame.u32Height;
  Rect rect;

  if (0 == meta->size) {
    rect = Rect(0, width - 1, 0, height - 1);
  } else {
    float ratio_x = float(width) / meta->width;
    float bbox_y_height = meta->height * height / width;
    float ratio_y = float(height) / bbox_y_height;
    float bbox_padding_top = (meta->height - bbox_y_height) / 2;

    rect = Rect(width, 0, height, 0);
    const float total_size = width * height;
    for (uint32_t i = 0; i < meta->size; ++i) {
      cvai_bbox_t bbox = meta->info[i].bbox;
      const float &&ww = bbox.x2 - bbox.x1;
      const float &&hh = bbox.y2 - bbox.y1;
      if (ww < 4 || hh < 4) {
        continue;
      }
      const float &&box_size = ww * hh;
      if (box_size / total_size < face_skip_ratio) {
        continue;
      }
      float x1 = bbox.x1 * ratio_x;
      float x2 = bbox.x2 * ratio_x;
      float y1 = (bbox.y1 - bbox_padding_top) * ratio_y;
      float y2 = (bbox.y2 - bbox_padding_top) * ratio_y;

      rect.l = std::min(rect.l, x1);
      rect.r = std::max(rect.r, x2);
      rect.t = std::min(rect.t, y1);
      rect.b = std::max(rect.b, y2);
    }
    if (rect.l > rect.r) {
      std::swap(rect.l, rect.r);
    }
    if (std::abs(rect.l - rect.r) < 4) {
      rect.l = 0;
      rect.r = width;
    }
    if (rect.t > rect.b) {
      std::swap(rect.t, rect.b);
    }
    if (std::abs(rect.t - rect.b) < 4) {
      rect.t = 0;
      rect.b = height;
    }
    rect.add_padding(m_padding_ratio);
    fitRatio(width, height, &rect);
  }

  if (!m_prev_rect.is_valid()) {
    m_prev_rect = Rect(0, width - 1, 0, height - 1);
  }
  transformRect(trans_ratio, m_prev_rect, &rect);
  fitFrame(width, height, &rect);

  VPSS_CHN_ATTR_S chnAttr;
  VPSS_CHN_DEFAULT_HELPER(&chnAttr, width, height, srcFrame->stVFrame.enPixelFormat, true);
  VPSS_CROP_INFO_S cropAttr;
  cropAttr.bEnable = true;
  cropAttr.enCropCoordinate = VPSS_CROP_RATIO_COOR;
  cropAttr.stCropRect = {(int)rect.l, (int)rect.t, (uint32_t)(rect.r - rect.l),
                         (uint32_t)(rect.b - rect.t)};
  mp_vpss_inst->sendCropChnFrame(srcFrame, &cropAttr, &chnAttr, 1);
  mp_vpss_inst->getFrame(dstFrame, 0);

  m_prev_rect = rect;
  return CVI_SUCCESS;
}

void DigitalTracking::transformRect(const float trans_ratio, const Rect &prev_rect,
                                    Rect *curr_rect) {
  curr_rect->l = prev_rect.l * (1.0 - trans_ratio) + curr_rect->l * trans_ratio;
  curr_rect->r = prev_rect.r * (1.0 - trans_ratio) + curr_rect->r * trans_ratio;
  curr_rect->t = prev_rect.t * (1.0 - trans_ratio) + curr_rect->t * trans_ratio;
  curr_rect->b = prev_rect.b * (1.0 - trans_ratio) + curr_rect->b * trans_ratio;
}

void DigitalTracking::fitRatio(const float width, const float height, Rect *rect) {
  float origin_ratio = (double)height / width;
  float curr_ratio = (rect->b - rect->t) / (rect->r - rect->l);

  if (curr_ratio > origin_ratio) {
    float new_w = (rect->b - rect->t) / origin_ratio;
    float w_centor = (rect->l + rect->r) / 2;
    rect->l = w_centor - new_w / 2;
    rect->r = rect->l + new_w;
  } else {
    float new_h = (rect->r - rect->l) * origin_ratio;
    float h_centor = (rect->t + rect->b) / 2;
    rect->t = h_centor - new_h / 2;
    rect->b = rect->t + new_h;
  }
}

void DigitalTracking::fitFrame(const float width, const float height, Rect *rect) {
  if (((rect->r - rect->l + 1) > width) || ((rect->b - rect->t + 1) > height)) {
    rect->l = 0;
    rect->r = width - 1;
    rect->t = 0;
    rect->b = height - 1;
  } else {
    if (rect->l < 0) {
      rect->r -= rect->l;
      rect->l = 0;
    } else if (rect->r >= width) {
      rect->l -= (rect->r - width);
      rect->r = width;
    }

    if (rect->t < 0) {
      rect->b -= rect->t;
      rect->t = 0;
    } else if (rect->b >= height) {
      rect->t -= (rect->b - height);
      rect->b = height;
    }
  }
}

template int DigitalTracking::run<cvai_face_t>(const VIDEO_FRAME_INFO_S *srcFrame,
                                               const cvai_face_t *meta,
                                               VIDEO_FRAME_INFO_S *dstFrame,
                                               const float face_skip_ratio,
                                               const float trans_ratio);
template int DigitalTracking::run<cvai_object_t>(const VIDEO_FRAME_INFO_S *srcFrame,
                                                 const cvai_object_t *meta,
                                                 VIDEO_FRAME_INFO_S *dstFrame,
                                                 const float face_skip_ratio,
                                                 const float trans_ratio);
}  // namespace service
}  // namespace cviai