#ifndef INCLUDE_BASE_MODEL_H_
#define INCLUDE_BASE_MODEL_H_

#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "common/model_output_types.hpp"
#include "image/base_image.hpp"
#include "net/base_net.hpp"
#include "preprocess/base_preprocessor.hpp"

#include "utils/profiler.hpp"

class BaseModel {
 public:
  BaseModel();
  virtual ~BaseModel() = default;
  int32_t modelOpen(const int device_id = 0);

  NetParam& getNetParam() { return net_param_; }
  void setNetParam(const NetParam& net_param) { net_param_ = net_param; }

  int getDeviceId() const;
  virtual int32_t inference(
      const std::shared_ptr<BaseImage>& image,
      std::shared_ptr<ModelOutputInfo>& out_data,
      const std::map<std::string, float>& parameters = {});
  /*
   * @brief 推理接口
   * @param images 输入图像，可以是原始图像，也可以是预处理后的图像
   * @param out_datas 输出数据
   * @return 0 成功，其他 失败
   */
  virtual int32_t inference(
      const std::vector<std::shared_ptr<BaseImage>>& images,
      std::vector<std::shared_ptr<ModelOutputInfo>>& out_datas,
      const std::map<std::string, float>& parameters = {});

  /*
   * @brief 推理接口
   * @param images 输入图像。外层 vector 表示 batch 维度，内层 vector
   * 表示模型多输入的图像数据
   * @param out_datas 输出数据
   * @return 0 成功，其他 失败
   */
  virtual int32_t inference(
      const std::vector<std::vector<std::shared_ptr<BaseImage>>>& images,
      std::vector<std::shared_ptr<ModelOutputInfo>>& out_datas,
      const std::map<std::string, float>& parameters = {});

  virtual int32_t inference(
      const std::shared_ptr<BaseImage>& image,
      const std::shared_ptr<ModelOutputInfo>& model_object_infos,
      std::vector<std::shared_ptr<ModelOutputInfo>>& out_datas,
      const std::map<std::string, float>& parameters = {});

  virtual int32_t outputParse(const std::shared_ptr<BaseImage>& image,
                              std::shared_ptr<ModelOutputInfo>& out_data);

  virtual int32_t outputParse(
      const std::vector<std::shared_ptr<BaseImage>>& images,
      std::vector<std::shared_ptr<ModelOutputInfo>>& out_datas);

  virtual int32_t outputParse(
      const std::vector<std::vector<std::shared_ptr<BaseImage>>>& images,
      std::vector<std::shared_ptr<ModelOutputInfo>>& out_datas);

  int32_t setPreprocessor(std::shared_ptr<BasePreprocessor> preprocessor);
  std::shared_ptr<BasePreprocessor> getPreprocessor() { return preprocessor_; }

  virtual int32_t onModelOpened() { return 0; }
  virtual int32_t onModelClosed() { return 0; }

  void setTypeMapping(const std::map<int, TDLObjectType>& type_mapping);
  virtual void setModelThreshold(float threshold);
  virtual void setExportFeature(int flag);
  virtual float getModelThreshold() const { return model_threshold_; }

  int32_t getPreprocessParameters(PreprocessParams& pre_param,
                                  const std::string& input_name = "");

  int32_t setPreprocessParameters(const PreprocessParams& pre_param,
                                  const std::string& input_name = "");
  virtual int32_t setupNetwork(NetParam& net_param);

  const std::vector<std::string>& getInputNames() const;
  const std::vector<std::string>& getOutputNames() const;
  int32_t getTensorInfo(const std::string& name, TensorInfo& info);
  uint32_t getIOTensorBytes() { return net_->getIOTensorBytes(); }
  int32_t setIOTensorMemory(uint64_t phy_addr, uint8_t* sys_mem,
                            uint32_t size) {
    return net_->setIOTensorMemory(phy_addr, sys_mem, size);
  }

 private:
  int getFitBatchSize(int left_size) const;
  void setInputBatchSize(const std::string& layer_name, int batch_size);

 protected:
  // Network and parameters
  NetParam net_param_;
  bool keep_aspect_ratio_ = false;

  std::shared_ptr<BaseNet> net_;
  std::shared_ptr<BasePreprocessor> preprocessor_;
  std::map<std::string, PreprocessParams> preprocess_params_;

  // Input and output configurations
  int input_batch_size_ = 0;

  std::string output_layer_;
  float model_threshold_ = 0.5;
  int export_feature = 0;

  std::map<std::string, std::vector<std::vector<float>>> batch_rescale_params_;

  std::vector<std::shared_ptr<BaseImage>> tmp_preprocess_images_;
  std::map<int, TDLObjectType> type_mapping_;

  Timer model_timer_;
};

#endif  // INCLUDE_BASE_MODEL_H_
// INCLUDE_BASE_MODEL_H_
