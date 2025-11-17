#include <gtest/gtest.h>

#include <fstream>
#include <string>
#include <unordered_map>

#include "cvi_tdl_test.hpp"
#include "json.hpp"
#include "preprocess/base_preprocessor.hpp"
#include "regression_utils.hpp"
#include "utils/tdl_log.hpp"

namespace fs = std::experimental::filesystem;
namespace cvitdl {
namespace unitest {

class OcrTestSuite : public CVI_TDLModelTestSuite {
 public:
  OcrTestSuite() : CVI_TDLModelTestSuite() {}

  virtual ~OcrTestSuite() = default;

  std::shared_ptr<BaseModel> model_;

 protected:
  virtual void SetUp() {
    // 主要执行的函数：loadModelConfig——ret, setModelDir——m_model_dir, getModel
    // 读取模型配置
    int32_t ret = TDLModelFactory::getInstance().loadModelConfig();
    if (ret != 0) {
      LOGE("load model config failed");
      return;
    }
    // 设置模型查找的目录，m_model_dir是输出的目录
    TDLModelFactory::getInstance().setModelDir(m_model_dir);
    // 读取模型ID和路径，其中，m_json_object是CVI_TDLTestContext类中的成员变量，用于存储测试相关的JSON配置
    std::string model_id = std::string(m_json_object["model_id"]);
    std::string model_path =
        m_model_dir.string() + "/" + gen_model_dir() + "/" +
        m_json_object["model_name"].get<std::string>() + gen_model_suffix();
    // 获取BaseModel模型实例（通过继承BaseModel类）
    model_ = TDLModelFactory::getInstance().getModel(
        model_id, model_path);  // One model id may correspond to multiple
                                // models with different sizes
    // 检查模型是否加载成功
    ASSERT_NE(model_, nullptr);
  }

  virtual void TearDown() {}
};

TEST_F(OcrTestSuite, accuracy) {
  // 获取图片路径
  std::string image_dir = (m_image_dir / m_json_object["image_dir"]).string();
  // 获取当前运行的平台名称，比如"cv181x"
  std::string platform = get_platform_str();
  // 获取上下文，包括：测试标志，JSON路径，测试相关的全局状态和配置
  CVI_TDLTestContext& context = CVI_TDLTestContext::getInstance();
  // 读取上下文的测试模式枚举。
  TestFlag test_flag = context.getTestFlag();
  // 用于存储处理结果的JSON对象
  nlohmann::ordered_json results;
  // 检查是否需要获取处理结果，并且填充数据到 results
  if (!checkToGetProcessResult(test_flag, platform, results)) {
    return;
  }
  // 遍历result中的每一项，结构为："image_name": {expected_results}
  // 也就是键值对。
  for (auto iter = results.begin(); iter != results.end(); iter++) {
    // 为当前条目生成完整的图像路径
    std::string image_path =
        (m_image_dir / m_json_object["image_dir"] / iter.key()).string();
    // 日志记录
    LOGIP("image_path: %s", image_path.c_str());
    // 将当前JSON条目的值，拷贝到expected_results变量中
    auto expected_results = iter.value();
    // 读取图像文件，返回BaseImage的智能指针
    std::shared_ptr<BaseImage> frame =
        ImageFactory::readImage(image_path, ImageFormat::RGB_PACKED);

    ASSERT_NE(frame, nullptr);
    // 构造输入图像的向量容器
    std::vector<std::shared_ptr<BaseImage>> input_images;
    // 将当前帧图像添加到输入图像向量中
    input_images.push_back(frame);
    // 存储模型输出信息的向量
    std::vector<std::shared_ptr<ModelOutputInfo>> out_data;
    // 执行模型推理，参数是输入图像和输出数据
    model_->inference(input_images, out_data);
    // ocr_meta存储模型输出的OCR信息
    std::shared_ptr<ModelOcrInfo> ocr_meta =
        std::static_pointer_cast<ModelOcrInfo>(out_data[0]);

    // 如果当前处于生成基线模式，那么更新JSON文件中的条目期望值，然后遍历下一个条目
    if (test_flag == TestFlag::GENERATE_FUNCTION_RES) {
      iter.value()["characters"] = ocr_meta->text_info;
      continue;
    }

    std::string gt_str = std::string(expected_results["characters"]);

    std::string pred_str = std::string(ocr_meta->text_info);
    ASSERT_TRUE(gt_str == pred_str);
  }  // end for

  // 如果当前处于生成基线模式，那么将更新后的JSON对象写回到文件中
  if (test_flag == TestFlag::GENERATE_FUNCTION_RES) {
    m_json_object[platform] = results;
    // 调用函数将JSON写到磁盘
    writeJsonFile(context.getJsonFilePath().string(), m_json_object);
  }
}  // end of TEST_F

// 这个是线程执行函数，需要的参数有：模型名称。

// 用来测试耗时、CPU/TPU使用率
TEST_F(OcrTestSuite, performance) {
  std::string model_path = m_model_dir.string() + "/" + gen_model_dir() + "/" +
                           m_json_object["model_name"].get<std::string>() +
                           gen_model_suffix();

  std::string platform = get_platform_str();
  CVI_TDLTestContext& context = CVI_TDLTestContext::getInstance();
  TestFlag test_flag = context.getTestFlag();
  nlohmann::ordered_json results;
  if (!checkToGetProcessResult(test_flag, platform, results)) {
    return;
  }
  if (results.empty()) {
    LOGIP("performance: no images available, skip");
    return;
  }

  auto iter = results.begin();
  std::string image_path =
      (m_image_dir / m_json_object["image_dir"] / iter.key()).string();
  auto frame = ImageFactory::readImage(image_path, ImageFormat::RGB_PACKED);
  if (!frame) {
    LOGE("performance: failed to read image %s", image_path.c_str());
  }

  run_performance(model_path, frame, model_);

}  // end of TEST_F
}  // namespace unitest
}  // namespace cvitdl
