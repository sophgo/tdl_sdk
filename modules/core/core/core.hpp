#pragma once
#include "core/core/cvai_core_types.h"
#include "cviai_log.hpp"
#include "ive/ive.h"
#include "vpss_engine.hpp"

#include <cviruntime.h>
#include <memory>
#include <vector>

#define DEFAULT_MODEL_THRESHOLD 0.5

namespace cviai {

/*
 * OPTION_BATCH_SIZE               = 1,
 * OPTION_PREPARE_BUF_FOR_INPUTS   = 2,  // Deprecated
 * OPTION_PREPARE_BUF_FOR_OUTPUTS  = 3,  // Deprecated
 * OPTION_OUTPUT_ALL_TENSORS       = 4,
 * OPTION_SKIP_PREPROCESS          = 5,
 * OPTION_SKIP_POSTPROCESS         = 6,
 */
struct ModelConfig {
  // FIXME: something strange...
  int32_t batch_size = 0;
  bool debug_mode = false;
  bool skip_postprocess = false;
  int input_mem_type = 1;
};

struct initSetup {
  float factor[3] = {0};
  float mean[3] = {0};
  meta_rescale_type_e rescale_type = RESCALE_CENTER;
  bool pad_reverse = false;
  bool keep_aspect_ratio = true;
  bool use_quantize_scale = false;
  bool use_crop = false;
};

struct VPSSConfig {
  meta_rescale_type_e rescale_type = RESCALE_CENTER;
  VPSS_CROP_INFO_S crop_attr;
  VPSS_CHN_ATTR_S chn_attr;
};

class Core {
 public:
  virtual ~Core() = default;
  int modelOpen(const char *filepath);
  int modelClose();
  int setIveInstance(IVE_HANDLE handle);
  int setVpssEngine(VpssEngine *engine);
  void skipVpssPreprocess(bool skip);
  virtual int getChnAttribute(const uint32_t width, const uint32_t height, const uint32_t idx,
                              VPSS_CHN_ATTR_S *attr);
  virtual void setModelThreshold(float threshold);
  float getModelThreshold();
  bool isInitialized();

 protected:
  virtual int initAfterModelOpened(std::vector<initSetup> *data);
  virtual int vpssPreprocess(const std::vector<VIDEO_FRAME_INFO_S *> &srcFrames,
                             std::vector<std::shared_ptr<VIDEO_FRAME_INFO_S>> *dstFrames);
  int run(std::vector<VIDEO_FRAME_INFO_S *> &frames);
  CVI_TENSOR *getInputTensor(int idx);
  CVI_TENSOR *getOutputTensor(int idx);

  // Class settings
  std::unique_ptr<ModelConfig> mp_config;
  // cvimodel related
  CVI_TENSOR *mp_input_tensors = nullptr;
  CVI_TENSOR *mp_output_tensors = nullptr;
  int32_t m_input_num = 0;
  int32_t m_output_num = 0;
  // Preprocessing & post processing related
  bool m_skip_vpss_preprocess = false;
  bool m_export_chn_attr = false;
  float m_model_threshold = DEFAULT_MODEL_THRESHOLD;
  std::vector<VPSSConfig> m_vpss_config;

  // Handle
  CVI_MODEL_HANDLE mp_model_handle = nullptr;
  IVE_HANDLE ive_handle = NULL;
  VpssEngine *mp_vpss_inst = nullptr;

 private:
  template <typename T>
  inline int __attribute__((always_inline)) registerFrame2Tensor(std::vector<T> &frames);
};
}  // namespace cviai