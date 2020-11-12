#include "deeplabv3.hpp"

#include "core/cviai_types_mem.h"
#include "core/utils/vpss_helper.h"
#include "core_utils.hpp"

#include "cvi_sys.h"
#include "opencv2/opencv.hpp"

#define SCALE (1 / 127.5)
#define MEAN 1.f
#define NAME_SCORE "score_Softmax"

namespace cviai {

Deeplabv3::Deeplabv3() {
  mp_config = std::make_unique<ModelConfig>();
  mp_config->input_mem_type = CVI_MEM_DEVICE;
}

Deeplabv3::~Deeplabv3() {
  if (m_gdc_blk != (VB_BLK)-1) {
    CVI_SYS_Munmap((void *)m_label_frame.stVFrame.pu8VirAddr[0],
                   m_label_frame.stVFrame.u32Length[0]);
    m_label_frame.stVFrame.pu8VirAddr[0] = NULL;
    CVI_VB_ReleaseBlock(m_gdc_blk);
  }
}

int Deeplabv3::initAfterModelOpened(std::vector<initSetup> *data) {
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
  m_export_chn_attr = true;

  CVI_TENSOR *input = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, mp_input_tensors, m_input_num);
  if (CREATE_VBFRAME_HELPER(&m_gdc_blk, &m_label_frame, input->shape.dim[3], input->shape.dim[2],
                            PIXEL_FORMAT_UINT8_C1) != CVI_SUCCESS) {
    return -1;
  }

  m_label_frame.stVFrame.pu8VirAddr[0] = (CVI_U8 *)CVI_SYS_MmapCache(
      m_label_frame.stVFrame.u64PhyAddr[0], m_label_frame.stVFrame.u32Length[0]);

  return 0;
}

int Deeplabv3::inference(VIDEO_FRAME_INFO_S *frame, VIDEO_FRAME_INFO_S *out_frame) {
  if (frame->stVFrame.enPixelFormat != PIXEL_FORMAT_RGB_888) {
    LOGE("Error: pixel format not match PIXEL_FORMAT_RGB_888.\n");
    return CVI_FAILURE;
  }

  std::vector<VIDEO_FRAME_INFO_S *> frames = {frame};
  run(frames);

  outputParser();

  VPSS_CHN_ATTR_S vpssChnAttr;
  vpssChnAttr.u32Width = frame->stVFrame.u32Width;
  vpssChnAttr.u32Height = frame->stVFrame.u32Height;
  vpssChnAttr.enVideoFormat = VIDEO_FORMAT_LINEAR;
  vpssChnAttr.enPixelFormat = frame->stVFrame.enPixelFormat;
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

int Deeplabv3::outputParser() {
  CVI_TENSOR *tensor = CVI_NN_GetTensorByName(NAME_SCORE, mp_output_tensors, m_output_num);
  float *out = (float *)CVI_NN_TensorPtr(tensor);
  CVI_SHAPE output_shape = CVI_NN_TensorShape(tensor);
  int size = output_shape.dim[2] * output_shape.dim[3];
  float *max_prob = (float *)malloc(size * sizeof(float));
  memset(max_prob, 0, size * sizeof(float));

  for (int32_t c = 0; c < output_shape.dim[1]; ++c) {
    int size_offset = c * size;

    for (int32_t h = 0; h < output_shape.dim[2]; ++h) {
      int width_offset = h * output_shape.dim[3];

      for (int32_t w = 0; w < output_shape.dim[3]; ++w) {
        if (out[size_offset + width_offset + w] > max_prob[width_offset + w]) {
          m_label_frame.stVFrame.pu8VirAddr[0][width_offset + w] = (int8_t)c;
          max_prob[width_offset + w] = out[size_offset + width_offset + w];
        }
      }
    }
  }

  free(max_prob);

  return CVI_SUCCESS;
}

}  // namespace cviai
