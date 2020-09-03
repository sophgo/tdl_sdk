#pragma once
#include "cvi_comm_video.h"
#include "cvi_comm_vpss.h"
#include "cviruntime.h"
#include "ive/ive.h"
#include "vpss_engine.hpp"

#include <memory>
#include <vector>

namespace cviai {

/*
 * OPTION_BATCH_SIZE               = 1,
 * OPTION_PREPARE_BUF_FOR_INPUTS   = 2,
 * OPTION_PREPARE_BUF_FOR_OUTPUTS  = 3,
 * OPTION_OUTPUT_ALL_TENSORS       = 4,
 * OPTION_SKIP_PREPROCESS          = 5,
 * OPTION_SKIP_POSTPROCESS         = 6,
 */
struct ModelConfig {
  // FIXME: something strange...
  int32_t batch_size = 0;
  bool init_input_buffer = true;
  bool init_output_buffer = true;
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
  float getInputScale();
  void skipVpssPreprocess(bool skip);
  void setModelThreshold(float threshold);

 protected:
  virtual int initAfterModelOpened() { return CVI_SUCCESS; }
  virtual int vpssPreprocess(const VIDEO_FRAME_INFO_S *srcFrame, VIDEO_FRAME_INFO_S *dstFrame);
  int run(VIDEO_FRAME_INFO_S *srcFrame);
  CVI_TENSOR *getInputTensor(int idx);
  CVI_TENSOR *getOutputTensor(int idx);

  // Class settings
  std::unique_ptr<ModelConfig> mp_config;
  // Runtime related
  CVI_MODEL_HANDLE mp_model_handle = nullptr;
  CVI_TENSOR *mp_input_tensors = nullptr;
  CVI_TENSOR *mp_output_tensors = nullptr;
  int32_t m_input_num = 0;
  int32_t m_output_num = 0;
  bool m_skip_vpss_preprocess = false;
  float m_model_threshold = 0.5;

  IVE_HANDLE ive_handle = NULL;
  VpssEngine *mp_vpss_inst = nullptr;
  std::vector<VPSS_CHN_ATTR_S> m_vpss_chn_attr;

 private:
  inline int __attribute__((always_inline)) runVideoForward(VIDEO_FRAME_INFO_S *srcFrame);
  bool m_reverse_device_mem = false;
};
}  // namespace cviai