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
class BaseModel {
 public:
  BaseModel();
  virtual ~BaseModel() = default;
  int32_t modelOpen(const std::string& model_path);
  int32_t setupNetwork(NetParam& net_param);

  int getDeviceId() const;

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

  virtual int32_t inference(
      const std::shared_ptr<BaseImage>& image,
      const std::shared_ptr<ModelOutputInfo>& model_object_infos,
      std::vector<std::shared_ptr<ModelOutputInfo>>& out_datas,
      const std::map<std::string, float>& parameters = {});

  virtual int32_t outputParse(
      const std::vector<std::shared_ptr<BaseImage>>& images,
      std::vector<std::shared_ptr<ModelOutputInfo>>& out_datas) = 0;
  int32_t setPreprocessor(std::shared_ptr<BasePreprocessor> preprocessor);
  std::shared_ptr<BasePreprocessor> getPreprocessor() { return preprocessor_; }
  virtual int32_t onModelOpened() { return 0; }
  virtual int32_t onModelClosed() { return 0; }
  void setTypeMapping(const std::map<int, TDLObjectType>& type_mapping);
  void setModelThreshold(float threshold);
  float getModelThreshold() const { return model_threshold_; }

  virtual int32_t setParameters(
      const std::map<std::string, float>& parameters) {
    return 0;
  }
  virtual int32_t getParameters(std::map<std::string, float>& parameters) {
    return 0;
  }

 private:
  int getFitBatchSize(int left_size) const;
  void setInputBatchSize(const std::string& layer_name, int batch_size);

 protected:
  // Network and parameters
  NetParam net_param_;
  std::shared_ptr<BaseNet> net_;
  std::shared_ptr<BasePreprocessor> preprocessor_;
  std::map<std::string, PreprocessParams> preprocess_params_;

  // Input and output configurations
  int input_batch_size_ = 0;

  std::string output_layer_;
  float model_threshold_ = 0.5;

  std::vector<std::vector<float>> batch_rescale_params_;

  std::vector<std::shared_ptr<BaseImage>> tmp_preprocess_images_;
  std::map<int, TDLObjectType> type_mapping_;
};

#endif  // INCLUDE_BASE_MODEL_H_
