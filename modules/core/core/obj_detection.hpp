#pragma once
#include <bitset>
#include "core/object/cvtdl_object_types.h"
#include "core_internel.hpp"
#define DEFAULT_MODEL_THRESHOLD 0.5
#define DEFAULT_MODEL_NMS_THRESHOLD 0.5

namespace cvitdl {

class DetectionBase : public Core {
 public:
  DetectionBase();
  virtual ~DetectionBase(){};
  virtual int inference(VIDEO_FRAME_INFO_S *srcFrame, cvtdl_object_t *obj_meta) {
    LOGE("inference function not implement!\n");
    return 0;
  }

  virtual const cvtdl_pre_param_t &get_preparam() { return preprocess_param_; }
  virtual void set_preparam(const cvtdl_pre_param_t &pre_param) { preprocess_param_ = pre_param; }
  virtual const cvtdl_det_algo_param_t &get_algparam() { return alg_param_; }
  virtual void set_algparam(const cvtdl_det_algo_param_t &alg_param);

 private:
  virtual int onModelOpened() override {
    LOGE("onModelOpened function not implement!\n");
    return 0;
  };

  virtual int onModelClosed() override { return CVI_TDL_SUCCESS; }

  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;

  virtual int vpssPreprocess(VIDEO_FRAME_INFO_S *srcFrame, VIDEO_FRAME_INFO_S *dstFrame,
                             VPSSConfig &vpss_config) override;

 protected:
  cvtdl_pre_param_t preprocess_param_;
  cvtdl_det_algo_param_t alg_param_;
};
}  // namespace cvitdl
