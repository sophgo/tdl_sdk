#ifndef INCLUDE_BASE_MODEL_H_
#define INCLUDE_BASE_MODEL_H_

#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "image/base_image.hpp"
#include "net/base_net.hpp"
#include "preprocess/base_preprocessor.hpp"
class BaseModel {
 public:
  BaseModel();
  virtual ~BaseModel() = default;
  int32_t modelOpen(const char* filepath);
  int32_t setupNetwork(NetParam& net_param);

  int getDeviceId() const;

  /*
   * @brief 推理接口
   * @param images 输入图像，可以是原始图像，也可以是预处理后的图像
   * @param out_datas 输出数据
   * @param src_width 原始图像输入宽，假如为0,则使用输入图像的宽
   * @param src_height 原始图像输入高，假如为0,则使用输入图像的高
   * @return 0 成功，其他 失败
   */
  virtual int32_t inference(
      const std::vector<std::shared_ptr<BaseImage>>& images,
      std::vector<void*>& out_datas, const int src_width = 0,
      const int src_height = 0);
  virtual int32_t outputParse(
      const std::vector<std::shared_ptr<BaseImage>>& images,
      std::vector<void*>& out_datas) = 0;
  int32_t setPreprocessor(std::shared_ptr<BasePreprocessor> preprocessor);

  virtual int32_t onModelOpened() { return 0; }
  virtual int32_t onModelClosed() { return 0; }

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
  std::string input_layer_ = "data";
  std::string output_layer_;
  float model_threshold_ = 0.5;

  std::vector<std::vector<float>> batch_rescale_params_;

  std::vector<std::shared_ptr<BaseImage>> tmp_preprocess_images_;
};

#endif  // INCLUDE_BASE_MODEL_H_
