#include "deeplabv3.hpp"

#include "core/cviai_types_mem.h"
#include "core/utils/vpss_helper.h"
#include "core_utils.hpp"

#include "cvi_sys.h"
#include "opencv2/opencv.hpp"

#define SCALE (1 / 127.5)
#define MEAN 1.f
#define NAME_SCORE "Conv_160_dequant"

namespace cviai {

Deeplabv3::Deeplabv3() : Core(CVI_MEM_DEVICE) {}

Deeplabv3::~Deeplabv3() {
  if (m_gdc_blk != (VB_BLK)-1) {
    CVI_SYS_Munmap((void *)m_label_frame.stVFrame.pu8VirAddr[0],
                   m_label_frame.stVFrame.u32Length[0]);
    m_label_frame.stVFrame.pu8VirAddr[0] = NULL;
    CVI_VB_ReleaseBlock(m_gdc_blk);
  }
}

int Deeplabv3::setupInputPreprocess(std::vector<InputPreprecessSetup> *data) {
  if (data->size() != 1) {
    LOGE("Face quality only has 1 input.\n");
    return CVI_FAILURE;
  }

  for (uint32_t i = 0; i < 3; i++) {
    (*data)[0].factor[i] = SCALE;
    (*data)[0].mean[i] = MEAN;
  }
  (*data)[0].use_quantize_scale = true;
  (*data)[0].keep_aspect_ratio = false;
  return CVI_SUCCESS;
}

int Deeplabv3::onModelOpened() {
  CVI_SHAPE shape = getOutputShape(NAME_SCORE);
  if (CREATE_VBFRAME_HELPER(&m_gdc_blk, &m_label_frame, shape.dim[3], shape.dim[2],
                            PIXEL_FORMAT_YUV_400) != CVI_SUCCESS) {
    return CVI_FAILURE;
  }

  m_label_frame.stVFrame.pu8VirAddr[0] = (CVI_U8 *)CVI_SYS_MmapCache(
      m_label_frame.stVFrame.u64PhyAddr[0], m_label_frame.stVFrame.u32Length[0]);
  return CVI_SUCCESS;
}

int Deeplabv3::inference(VIDEO_FRAME_INFO_S *frame, VIDEO_FRAME_INFO_S *out_frame,
                         cvai_class_filter_t *filter) {
  if (frame->stVFrame.enPixelFormat != PIXEL_FORMAT_RGB_888) {
    LOGE("Error: pixel format not match PIXEL_FORMAT_RGB_888.\n");
    return CVI_FAILURE;
  }

  std::vector<VIDEO_FRAME_INFO_S *> frames = {frame};
  run(frames);

  outputParser(filter);

  VPSS_CHN_ATTR_S vpssChnAttr;
  vpssChnAttr.u32Width = frame->stVFrame.u32Width;
  vpssChnAttr.u32Height = frame->stVFrame.u32Height;
  vpssChnAttr.enVideoFormat = VIDEO_FORMAT_LINEAR;
  vpssChnAttr.enPixelFormat = PIXEL_FORMAT_YUV_400;
  vpssChnAttr.stFrameRate.s32SrcFrameRate = -1;
  vpssChnAttr.stFrameRate.s32DstFrameRate = -1;
  vpssChnAttr.u32Depth = 1;
  vpssChnAttr.bMirror = CVI_FALSE;
  vpssChnAttr.bFlip = CVI_FALSE;
  vpssChnAttr.stAspectRatio.enMode = ASPECT_RATIO_NONE;
  vpssChnAttr.stNormalize.bEnable = CVI_FALSE;

  VPSS_SCALE_COEF_E enCoef = VPSS_SCALE_COEF_NEAREST;
  mp_vpss_inst->sendFrame(&m_label_frame, &vpssChnAttr, &enCoef, 1);
  if (CVI_SUCCESS != mp_vpss_inst->getFrame(out_frame, 0)) {
    LOGE("Deeplabv3 resized output label failed.");
    return CVI_FAILURE;
  }

  return CVI_SUCCESS;
}

inline bool is_preserved_class(cvai_class_filter_t *filter, int32_t c) {
  if (c == 0) return true;  // "unlabled class" should be preserved

  if (filter->num_preserved_classes > 0) {
    for (uint32_t j = 0; j < filter->num_preserved_classes; j++) {
      if (filter->preserved_class_ids[j] == (uint32_t)c) {
        return true;
      }
    }
  }

  return false;
}

int Deeplabv3::outputParser(cvai_class_filter_t *filter) {
  float *out = getOutputRawPtr<float>(NAME_SCORE);

  CVI_SHAPE output_shape = getOutputShape(NAME_SCORE);

  int size = output_shape.dim[2] * output_shape.dim[3];
  std::vector<float> max_prob(size, 0);

  for (int32_t c = 0; c < output_shape.dim[1]; ++c) {
    int size_offset = c * size;
    for (int32_t h = 0; h < output_shape.dim[2]; ++h) {
      int width_offset = h * output_shape.dim[3];
      int frame_offset = h * m_label_frame.stVFrame.u32Stride[0];

      for (int32_t w = 0; w < output_shape.dim[3]; ++w) {
        if (out[size_offset + width_offset + w] > max_prob[width_offset + w]) {
          m_label_frame.stVFrame.pu8VirAddr[0][frame_offset + w] = (int8_t)c;
          max_prob[width_offset + w] = out[size_offset + width_offset + w];
        }
      }
    }
  }

  if (filter) {
    for (int32_t h = 0; h < output_shape.dim[2]; ++h) {
      int frame_offset = h * m_label_frame.stVFrame.u32Stride[0];

      for (int32_t w = 0; w < output_shape.dim[3]; ++w) {
        if (!is_preserved_class(filter, m_label_frame.stVFrame.pu8VirAddr[0][frame_offset + w])) {
          m_label_frame.stVFrame.pu8VirAddr[0][frame_offset + w] = 0;
        }
      }
    }
  }

  CVI_SYS_IonFlushCache(m_label_frame.stVFrame.u64PhyAddr[0], m_label_frame.stVFrame.pu8VirAddr[0],
                        m_label_frame.stVFrame.u32Length[0]);

  return CVI_SUCCESS;
}

}  // namespace cviai
