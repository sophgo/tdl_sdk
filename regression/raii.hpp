#pragma once
#include <memory>
#include <string>
#include "cvi_comm.h"
#include "cvi_tdl.h"
#include "cvi_tdl_media.h"

namespace cvitdl {
namespace unitest {

// Convenience class for loading image from file.
class Image {
 public:
  Image(PIXEL_FORMAT_E format, uint32_t width, uint32_t height);
  Image(const std::string &file, PIXEL_FORMAT_E format);
  Image(const Image &) = delete;
  ~Image();

  bool open();

  VIDEO_FRAME_INFO_S *getFrame() { return &m_frame; }

  const VIDEO_FRAME_INFO_S *getFrame() const { return &m_frame; }

  Image &operator=(const Image &) = delete;

 private:
  bool createEmpty();

  imgprocess_t img_handle;
  VIDEO_FRAME_INFO_S m_frame;
  PIXEL_FORMAT_E m_format;
  std::string m_filepath;
  uint32_t m_width;
  uint32_t m_height;
  bool m_opened;
};

// A class which setup and destroy TDL model
class VpssPreprocessor {
 public:
  VpssPreprocessor(VPSS_GRP grp, VPSS_CHN chn, uint32_t width, uint32_t height,
                   PIXEL_FORMAT_E format);
  VpssPreprocessor(VPSS_GRP grp, VPSS_CHN chn, const VIDEO_FRAME_INFO_S *frame);
  VpssPreprocessor(VPSS_GRP grp, VPSS_CHN chn, const Image &image);
  ~VpssPreprocessor();

  VpssPreprocessor(const VpssPreprocessor &) = delete;
  VpssPreprocessor &operator=(const VpssPreprocessor &) = delete;

  void open();
  void close();
  void setChnConfig(const cvtdl_vpssconfig_t &chn_config);
  void setChnConfig(const VPSS_CHN_ATTR_S &chn_config);
  void setGrpConfig(uint32_t width, uint32_t height, PIXEL_FORMAT_E format);
  void preprocess(const VIDEO_FRAME_INFO_S *input_frame, VIDEO_FRAME_INFO_S *output_frame);
  void resetVpss(uint32_t width, uint32_t height, PIXEL_FORMAT_E format,
                 const VPSS_CHN_ATTR_S &chn_config);
  void resetVpss(const Image &image, const cvtdl_vpssconfig_t &chn_config);

 private:
  VPSS_GRP m_grp_id;
  VPSS_CHN m_chn_id;
  uint32_t m_grp_width;
  uint32_t m_grp_height;
  PIXEL_FORMAT_E m_format;
  cvtdl_vpssconfig_t m_vpss_chn_config;
};

// A class which setup and destroy TDL model
class TDLModelHandler {
 public:
  TDLModelHandler(cvitdl_handle_t handle, CVI_TDL_SUPPORTED_MODEL_E index,
                  const std::string &model_path, bool skip_vpsspreprocess = false);

  ~TDLModelHandler();
  TDLModelHandler(const TDLModelHandler &) = delete;
  TDLModelHandler &operator=(const TDLModelHandler &) = delete;

  void open();
  void close();

 protected:
  bool m_is_model_opened;
  cvitdl_handle_t m_handle;
  const CVI_TDL_SUPPORTED_MODEL_E m_model_index;
  const std::string m_model_path;
  const bool m_skip_vpsspreprocess;
};

// A class which manage TDL Object life cycle
template <typename T>
class TDLObject {
 public:
  TDLObject() {
    ptr_obj = (T *)malloc(sizeof(T));
    memset(ptr_obj, 0, sizeof(T));
  }

  ~TDLObject() { this->release(); }

  TDLObject(const TDLObject &) = delete;
  TDLObject &operator=(const TDLObject &) = delete;

  void release() {
    if (ptr_obj) {
      CVI_TDL_Free(ptr_obj);
      free(ptr_obj);
      ptr_obj = NULL;
    }
  }

  operator T *() { return ptr_obj; }
  T *operator->() const { return ptr_obj; }
  T &operator*() const { return *ptr_obj; }

 private:
  T *ptr_obj;
};

}  // namespace unitest
}  // namespace cvitdl
