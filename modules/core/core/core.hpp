#pragma once
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
  bool skip_preprocess = false;
  bool skip_postprocess = false;
  int input_mem_type = 1;
  int output_mem_type = 1;
};

class Core {
 public:
  virtual ~Core() = default;
  int modelOpen(const char *filepath);
  int modelClose();
  int setIveInstance(IVE_HANDLE handle);
  int setVpssEngine(VpssEngine *engine);
  void skipVpssPreprocess(bool skip);
  virtual void setModelThreshold(float threshold);
  float getModelThreshold();
  bool isInitialized();

 protected:
  virtual int initAfterModelOpened(float *factor, float *mean, bool &pad_reverse,
                                   bool &keep_aspect_ratio, bool &use_model_threshold);
  virtual int vpssPreprocess(const VIDEO_FRAME_INFO_S *srcFrame, VIDEO_FRAME_INFO_S *dstFrame);
  int run(VIDEO_FRAME_INFO_S *srcFrame);
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
  bool m_use_vpss_crop = false;
  float m_model_threshold = DEFAULT_MODEL_THRESHOLD;
  VPSS_CROP_INFO_S m_crop_attr;

  // Handle
  CVI_MODEL_HANDLE mp_model_handle = nullptr;
  IVE_HANDLE ive_handle = NULL;
  VpssEngine *mp_vpss_inst = nullptr;

 private:
  inline int __attribute__((always_inline)) runVideoForward(VIDEO_FRAME_INFO_S *srcFrame);
  bool m_reverse_device_mem = false;

  std::vector<VPSS_CHN_ATTR_S> m_vpss_chn_attr;
};
}  // namespace cviai