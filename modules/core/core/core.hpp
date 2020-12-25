#pragma once
#include "core/core/cvai_core_types.h"
#include "core/core/cvai_vpss_types.h"
#include "cviai_log.hpp"
#include "ive/ive.h"
#include "vpss_engine.hpp"

#include <cviruntime.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

#define DEFAULT_MODEL_THRESHOLD 0.5

namespace cviai {

struct CvimodelConfig {
  // FIXME: something strange...
  int32_t batch_size = 0;
  bool debug_mode = false;
  bool skip_postprocess = false;
  int input_mem_type = CVI_MEM_SYSTEM;
};

struct CvimodelPair {
  CVI_TENSOR *tensors = nullptr;
  int32_t num = 0;
};

struct CvimodelInfo {
  CvimodelConfig conf;
  CVI_MODEL_HANDLE handle = nullptr;
  CvimodelPair in;
  CvimodelPair out;
};

struct TensorInfo {
  std::string tensor_name;
  void *raw_pointer;
  CVI_SHAPE shape;
  CVI_TENSOR *tensor_handle;

  // Tensor size = (number of tensor elements) * typeof(tensor type))
  size_t tensor_size;

  // number of tensor elements
  size_t tensor_elem;
  template <typename DataType>
  DataType *get() const {
    return static_cast<DataType *>(raw_pointer);
  }
};

struct InputPreprecessSetup {
  float factor[3] = {0};
  float mean[3] = {0};
  meta_rescale_type_e rescale_type = RESCALE_CENTER;
  bool pad_reverse = false;
  bool keep_aspect_ratio = true;
  bool use_quantize_scale = false;
  bool use_crop = false;
  VPSS_SCALE_COEF_E resize_method = VPSS_SCALE_COEF_BICUBIC;
};

struct VPSSConfig {
  meta_rescale_type_e rescale_type = RESCALE_CENTER;
  VPSS_CROP_INFO_S crop_attr;
  VPSS_SCALE_COEF_E chn_coeff = VPSS_SCALE_COEF_BICUBIC;
  VPSS_CHN_ATTR_S chn_attr;
  CVI_FRAME_TYPE frame_type = CVI_FRAME_PLANAR;
};

class Core {
 public:
  Core(CVI_MEM_TYPE_E input_mem_type, bool skip_postprocess = false, int32_t batch_size = 0);
  Core();
  Core(const Core &) = delete;
  Core &operator=(const Core &) = delete;

  virtual ~Core() = default;
  int modelOpen(const char *filepath);
  int modelClose();
  int setIveInstance(IVE_HANDLE handle);
  int setVpssEngine(VpssEngine *engine);
  void skipVpssPreprocess(bool skip);
  bool hasSkippedVpssPreprocess() const { return m_skip_vpss_preprocess; }
  virtual int getChnConfig(const uint32_t width, const uint32_t height, const uint32_t idx,
                           cvai_vpssconfig_t *chn_config);
  virtual void setModelThreshold(float threshold);
  float getModelThreshold();
  bool isInitialized();

 protected:
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data);
  virtual int vpssPreprocess(const std::vector<VIDEO_FRAME_INFO_S *> &srcFrames,
                             std::vector<std::shared_ptr<VIDEO_FRAME_INFO_S>> *dstFrames);
  int run(std::vector<VIDEO_FRAME_INFO_S *> &frames);

  /*
   * Input/Output getter functions
   */
  CVI_TENSOR *getInputTensor(int idx);
  CVI_TENSOR *getOutputTensor(int idx);

  const TensorInfo &getOutputTensorInfo(const std::string &name);
  const TensorInfo &getInputTensorInfo(const std::string &name);

  const TensorInfo &getOutputTensorInfo(size_t index);
  const TensorInfo &getInputTensorInfo(size_t index);

  size_t getNumInputTensor() const;
  size_t getNumOutputTensor() const;

  CVI_SHAPE getInputShape(size_t index);
  CVI_SHAPE getOutputShape(size_t index);
  CVI_SHAPE getInputShape(const std::string &name);
  CVI_SHAPE getOutputShape(const std::string &name);

  size_t getOutputTensorElem(size_t index);
  size_t getOutputTensorElem(const std::string &name);
  size_t getInputTensorElem(size_t index);
  size_t getInputTensorElem(const std::string &name);

  size_t getOutputTensorSize(size_t index);
  size_t getOutputTensorSize(const std::string &name);
  size_t getInputTensorSize(size_t index);
  size_t getInputTensorSize(const std::string &name);

  float getInputQuantScale(size_t index);
  float getInputQuantScale(const std::string &name);

  template <typename DataType>
  DataType *getInputRawPtr(size_t index) {
    return getInputTensorInfo(index).get<DataType>();
  }

  template <typename DataType>
  DataType *getOutputRawPtr(size_t index) {
    return getOutputTensorInfo(index).get<DataType>();
  }

  template <typename DataType>
  DataType *getInputRawPtr(const std::string &name) {
    return getInputTensorInfo(name).get<DataType>();
  }

  template <typename DataType>
  DataType *getOutputRawPtr(const std::string &name) {
    return getOutputTensorInfo(name).get<DataType>();
  }
  ////////////////////////////////////////////////////

  virtual int onModelOpened() { return CVI_SUCCESS; }
  virtual bool allowExportChannelAttribute() const { return false; }

  void setInputMemType(CVI_MEM_TYPE_E type) { mp_mi->conf.input_mem_type = type; }
  std::vector<VPSSConfig> m_vpss_config;

  // Post processing related control
  float m_model_threshold = DEFAULT_MODEL_THRESHOLD;

  // External handle
  IVE_HANDLE ive_handle = NULL;
  VpssEngine *mp_vpss_inst = nullptr;

 private:
  template <typename T>
  inline int __attribute__((always_inline)) registerFrame2Tensor(std::vector<T> &frames);

  void setupTensorInfo(CVI_TENSOR *tensor, int32_t num_tensors,
                       std::map<std::string, TensorInfo> *tensor_info);

  std::map<std::string, TensorInfo> m_input_tensor_info;
  std::map<std::string, TensorInfo> m_output_tensor_info;

  // Preprocessing related control
  bool m_skip_vpss_preprocess = false;

  // Cvimodel related
  std::unique_ptr<CvimodelInfo> mp_mi;
};
}  // namespace cviai