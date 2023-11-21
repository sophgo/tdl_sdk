#include "raii.hpp"
#include <cvi_vb.h>
#include <cvi_vpss.h>
#include "core/utils/vpss_helper.h"
#include "cvi_comm.h"
#include "cvi_tdl_media.h"
#include "gtest.h"

namespace cvitdl {
namespace unitest {

// Image
////////////////////////////////
Image::Image(const std::string &file, PIXEL_FORMAT_E format)
    : m_format(format), m_filepath(file), m_opened(false) {}

Image::~Image() {
  if (m_opened) {
    CVI_TDL_ReleaseImage(&m_frame);
  }
}

Image::Image(PIXEL_FORMAT_E format, uint32_t width, uint32_t height)
    : m_format(format), m_width(width), m_height(height), m_opened(false) {}

bool Image::open() {
  if (m_opened) return true;

  if (m_filepath.empty()) {
    if (!createEmpty()) {
      return false;
    }
  } else {
    if (CVI_TDL_ReadImage(m_filepath.c_str(), &m_frame, m_format) != CVI_SUCCESS) {
      m_width = m_frame.stVFrame.u32Width;
      m_height = m_frame.stVFrame.u32Height;
      return false;
    }
  }

  m_opened = true;
  return true;
}

bool Image::createEmpty() {
  return CREATE_ION_HELPER(&m_frame, m_width, m_height, m_format, "cvitdl/image") ==
         CVI_TDL_SUCCESS;
}

// TDLModelHandler
///////////////////////////////////
TDLModelHandler::TDLModelHandler(cvitdl_handle_t handle, CVI_TDL_SUPPORTED_MODEL_E index,
                                 const std::string &model_path, bool skip_vpsspreprocess)
    : m_is_model_opened(false),
      m_handle(handle),
      m_model_index(index),
      m_model_path(model_path),
      m_skip_vpsspreprocess(skip_vpsspreprocess) {}

TDLModelHandler::~TDLModelHandler() { close(); }

void TDLModelHandler::close() {
  if (m_is_model_opened) {
    ASSERT_EQ(CVI_TDL_CloseModel(m_handle, m_model_index), CVI_TDL_SUCCESS);
  }
}

void TDLModelHandler::open() {
  if (!m_is_model_opened) {
    ASSERT_EQ(CVI_TDL_OpenModel(m_handle, m_model_index, m_model_path.c_str()), CVI_TDL_SUCCESS)
        << "failed to set model path: " << m_model_path;
    ASSERT_EQ(CVI_TDL_SetSkipVpssPreprocess(m_handle, m_model_index, m_skip_vpsspreprocess),
              CVI_TDL_SUCCESS);
    m_is_model_opened = true;
  }
}

// VpssPreprocessor
///////////////////////////////////
VpssPreprocessor::VpssPreprocessor(VPSS_GRP grp, VPSS_CHN chn, uint32_t width, uint32_t height,
                                   PIXEL_FORMAT_E format)
    : m_grp_id(grp), m_chn_id(chn), m_grp_width(width), m_grp_height(height), m_format(format) {
  float factor[3] = {1.0, 1.0, 1.0};
  float mean[3] = {0.0, 0.0, 0.0};
  VPSS_CHN_SQ_HELPER(&m_vpss_chn_config.chn_attr, width, height, PIXEL_FORMAT_RGB_888, factor, mean,
                     false);
  m_vpss_chn_config.chn_coeff = VPSS_SCALE_COEF_BICUBIC;
}

VpssPreprocessor::VpssPreprocessor(VPSS_GRP grp, VPSS_CHN chn, const VIDEO_FRAME_INFO_S *frame)
    : VpssPreprocessor(grp, chn, frame->stVFrame.u32Width, frame->stVFrame.u32Height,
                       frame->stVFrame.enPixelFormat) {}

VpssPreprocessor::VpssPreprocessor(VPSS_GRP grp, VPSS_CHN chn, const Image &image)
    : VpssPreprocessor(grp, chn, image.getFrame()->stVFrame.u32Width,
                       image.getFrame()->stVFrame.u32Height,
                       image.getFrame()->stVFrame.enPixelFormat) {}

VpssPreprocessor::~VpssPreprocessor() { close(); }

void VpssPreprocessor::open() {
  VPSS_GRP_ATTR_S vpss_grp_attr;
  VPSS_GRP_DEFAULT_HELPER2(&vpss_grp_attr, m_grp_width, m_grp_height, m_format, 0);

  ASSERT_EQ(CVI_VPSS_CreateGrp(m_grp_id, &vpss_grp_attr), CVI_SUCCESS);
  ASSERT_EQ(CVI_VPSS_ResetGrp(m_grp_id), CVI_SUCCESS);
  ASSERT_EQ(CVI_VPSS_SetChnScaleCoefLevel(m_grp_id, m_chn_id, m_vpss_chn_config.chn_coeff),
            CVI_SUCCESS);
  ASSERT_EQ(CVI_VPSS_SetChnAttr(m_grp_id, m_chn_id, &m_vpss_chn_config.chn_attr), CVI_SUCCESS);
  ASSERT_EQ(CVI_VPSS_EnableChn(m_grp_id, m_chn_id), CVI_SUCCESS);
  ASSERT_EQ(CVI_VPSS_StartGrp(m_grp_id), CVI_SUCCESS);
}

void VpssPreprocessor::close() {
  ASSERT_EQ(CVI_VPSS_DisableChn(m_grp_id, m_chn_id), CVI_SUCCESS);
  ASSERT_EQ(CVI_VPSS_StopGrp(m_grp_id), CVI_SUCCESS);
  ASSERT_EQ(CVI_VPSS_DestroyGrp(m_grp_id), CVI_SUCCESS);
}

void VpssPreprocessor::preprocess(const VIDEO_FRAME_INFO_S *input_frame,
                                  VIDEO_FRAME_INFO_S *output_frame) {
  ASSERT_EQ(CVI_VPSS_SendFrame(m_grp_id, input_frame, 1000), CVI_SUCCESS);
  ASSERT_EQ(CVI_VPSS_GetChnFrame(m_grp_id, m_chn_id, output_frame, 1000), CVI_SUCCESS);
}

void VpssPreprocessor::setGrpConfig(uint32_t width, uint32_t height, PIXEL_FORMAT_E format) {
  m_grp_width = width;
  m_grp_height = height;
  m_format = format;

  VPSS_GRP_ATTR_S vpss_grp_attr;
  VPSS_GRP_DEFAULT_HELPER2(&vpss_grp_attr, width, height, format, 0);
  ASSERT_EQ(CVI_VPSS_SetGrpAttr(m_grp_id, &vpss_grp_attr), CVI_SUCCESS);
}

void VpssPreprocessor::setChnConfig(const cvtdl_vpssconfig_t &chn_config) {
  m_vpss_chn_config = chn_config;
}

void VpssPreprocessor::setChnConfig(const VPSS_CHN_ATTR_S &chn_config) {
  m_vpss_chn_config.chn_attr = chn_config;
}

void VpssPreprocessor::resetVpss(uint32_t width, uint32_t height, PIXEL_FORMAT_E format,
                                 const VPSS_CHN_ATTR_S &chn_config) {
  m_grp_width = width;
  m_grp_height = height;
  m_format = format;
  m_vpss_chn_config.chn_attr = chn_config;
  VPSS_GRP_ATTR_S vpss_grp_attr;
  VPSS_GRP_DEFAULT_HELPER2(&vpss_grp_attr, m_grp_width, m_grp_height, m_format, 0);
  ASSERT_EQ(CVI_VPSS_SetChnScaleCoefLevel(m_grp_id, m_chn_id, m_vpss_chn_config.chn_coeff),
            CVI_SUCCESS);
  ASSERT_EQ(CVI_VPSS_SetGrpAttr(m_grp_id, &vpss_grp_attr), CVI_SUCCESS);
  ASSERT_EQ(CVI_VPSS_SetChnAttr(m_grp_id, m_chn_id, &m_vpss_chn_config.chn_attr), CVI_SUCCESS);
}

void VpssPreprocessor::resetVpss(const Image &image, const cvtdl_vpssconfig_t &chn_config) {
  m_grp_width = image.getFrame()->stVFrame.u32Width;
  m_grp_height = image.getFrame()->stVFrame.u32Height;
  m_format = image.getFrame()->stVFrame.enPixelFormat;
  m_vpss_chn_config = chn_config;
  VPSS_GRP_ATTR_S vpss_grp_attr;
  VPSS_GRP_DEFAULT_HELPER2(&vpss_grp_attr, m_grp_width, m_grp_height, m_format, 0);
  ASSERT_EQ(CVI_VPSS_SetChnScaleCoefLevel(m_grp_id, m_chn_id, m_vpss_chn_config.chn_coeff),
            CVI_SUCCESS);
  ASSERT_EQ(CVI_VPSS_SetGrpAttr(m_grp_id, &vpss_grp_attr), CVI_SUCCESS);
  ASSERT_EQ(CVI_VPSS_SetChnAttr(m_grp_id, m_chn_id, &m_vpss_chn_config.chn_attr), CVI_SUCCESS);
}

}  // namespace unitest
}  // namespace cvitdl
