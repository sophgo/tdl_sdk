#pragma once
#include "cviruntime.h"

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
  // int32_t batch_size = 0;
  bool init_input_buffer = true;
  bool init_output_buffer = true;
  bool debug_mode = false;
  bool skip_preprocess = false;
  bool skip_postprocess = false;
};

class core {
public:
  bool modelOpen(const char *filepath, ModelConfig *config = nullptr);
  bool modelClose();
private:
  CVI_MODEL_HANDLE mp_model_handle = nullptr;
  CVI_TENSOR *mp_input_tensors = nullptr;
  CVI_TENSOR *mp_output_tensors = nullptr;
  int32_t m_input_num = 0;
  int32_t m_output_num = 0;
};
}