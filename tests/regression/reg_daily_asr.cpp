#include <gtest/gtest.h>

#include <fstream>
#include <string>
#include <unordered_map>

#include "cvi_tdl_test.hpp"
#include "image/base_image.hpp"
#include "json.hpp"
#include "preprocess/base_preprocessor.hpp"
#include "regression_utils.hpp"
#include "speech_recognition/zipformer_encoder.hpp"
#include "tdl_model_factory.hpp"
#include "utils/common_utils.hpp"
#include "utils/tdl_log.hpp"

namespace fs = std::experimental::filesystem;
namespace cvitdl {
namespace unitest {

class AsrTestSuite : public CVI_TDLModelTestSuite {
 public:
  AsrTestSuite() : CVI_TDLModelTestSuite() {}

  virtual ~AsrTestSuite() = default;

  std::shared_ptr<BaseModel> encoder_model_;
  std::shared_ptr<BaseModel> decoder_model_;
  std::shared_ptr<BaseModel> joiner_model_;

  std::shared_ptr<ZipformerEncoder> zipformer_model_;

  int sample_rate_;

 protected:
  virtual void SetUp() {
    int32_t ret = TDLModelFactory::getInstance().loadModelConfig();
    if (ret != 0) {
      LOGE("load model config failed");
      return;
    }
    TDLModelFactory::getInstance().setModelDir(m_model_dir);

    std::string model_id_encoder =
        std::string(m_json_object["model_id_encoder"]);
    std::string model_id_decoder =
        std::string(m_json_object["model_id_decoder"]);
    std::string model_id_joiner = std::string(m_json_object["model_id_joiner"]);
    std::string model_path_encoder =
        m_model_dir.string() + "/" + gen_model_dir() + "/" +
        m_json_object["encoder_model_name"].get<std::string>() +
        gen_model_suffix();

    std::string model_path_decoder =
        m_model_dir.string() + "/" + gen_model_dir() + "/" +
        m_json_object["decoder_model_name"].get<std::string>() +
        gen_model_suffix();

    std::string model_path_joiner =
        m_model_dir.string() + "/" + gen_model_dir() + "/" +
        m_json_object["joiner_model_name"].get<std::string>() +
        gen_model_suffix();

    encoder_model_ = TDLModelFactory::getInstance().getModel(
        model_id_encoder,
        model_path_encoder);  // One model id may correspond to multiple
                              // models with different sizes
    decoder_model_ = TDLModelFactory::getInstance().getModel(
        model_id_decoder,
        model_path_decoder);  // One model id may correspond to multiple
                              // models with different sizes
    joiner_model_ = TDLModelFactory::getInstance().getModel(
        model_id_joiner,
        model_path_joiner);  // One model id may correspond to multiple
                             // models with different sizes

    ASSERT_NE(encoder_model_, nullptr);
    ASSERT_NE(decoder_model_, nullptr);
    ASSERT_NE(joiner_model_, nullptr);

    std::string tokens_path =
        m_image_dir.string() + "/" + std::string(m_json_object["attachment"]);

    if (model_id_encoder == "RECOGNITION_SPEECH_ZIPFORMER_ENCODER") {
      zipformer_model_ =
          std::dynamic_pointer_cast<ZipformerEncoder>(encoder_model_);
      zipformer_model_->setTokensPath(tokens_path);
      zipformer_model_->setModel(decoder_model_, joiner_model_);
    } else {  // other asr model
      LOGE(" Unsupport model_id_encoder : %s", model_id_encoder);
      ASSERT_TRUE(false);
    }

    sample_rate_ = m_json_object["sample_rate"];
  }

  virtual void TearDown() {}
};

TEST_F(AsrTestSuite, accuracy) {
  std::string image_dir = (m_image_dir / m_json_object["image_dir"]).string();
  std::string platform = get_platform_str();
  CVI_TDLTestContext& context = CVI_TDLTestContext::getInstance();
  TestFlag test_flag = context.getTestFlag();
  nlohmann::ordered_json results;
  if (!checkToGetProcessResult(test_flag, platform, results)) {
    return;
  }
  for (auto iter = results.begin(); iter != results.end(); iter++) {
    std::string image_path =
        (m_image_dir / m_json_object["image_dir"] / iter.key()).string();
    LOGIP("image_path: %s\n", image_path.c_str());

    auto expected_results = iter.value();

    std::vector<uint8_t> buffer;
    ASSERT_TRUE(CommonUtils::readBinaryFile(image_path, buffer));
    for (int i = 0; i < sample_rate_ * 0.5 * 2;
         i++) {  // pad 0.5s, *2 for uint8_t to int16
      buffer.push_back(0);
    }

    int frame_size = buffer.size();

    std::shared_ptr<BaseImage> bin_data = ImageFactory::createImage(
        frame_size, 1, ImageFormat::GRAY, TDLDataType::UINT8, true);
    uint8_t* data_buffer = bin_data->getVirtualAddress()[0];
    memcpy(data_buffer, buffer.data(), frame_size * sizeof(uint8_t));

    std::shared_ptr<ModelASRInfo> asr_meta = std::make_shared<ModelASRInfo>();
    asr_meta->input_finished = true;
    std::shared_ptr<ModelOutputInfo> output_data =
        std::static_pointer_cast<ModelOutputInfo>(asr_meta);

    if (zipformer_model_) {
      zipformer_model_->inference(bin_data, output_data);
    } else {  // other asr model
      LOGE("asr model is not initialized");
      ASSERT_TRUE(false);
    }

    if (test_flag == TestFlag::GENERATE_FUNCTION_RES) {
      iter.value()["characters"] = asr_meta->text_info;
      continue;
    }
    std::string gt_str = std::string(expected_results["characters"]);

    std::string pred_str = std::string(asr_meta->text_info);
    ASSERT_TRUE(gt_str == pred_str);
  }  // end for

  if (test_flag == TestFlag::GENERATE_FUNCTION_RES) {
    m_json_object[platform] = results;
    writeJsonFile(context.getJsonFilePath().string(), m_json_object);
  }
}  // end of TEST_F

TEST_F(AsrTestSuite, performance) {
  std::string model_path_encoder =
      m_model_dir.string() + "/" + gen_model_dir() + "/" +
      m_json_object["encoder_model_name"].get<std::string>() +
      gen_model_suffix();

  std::string image_dir = (m_image_dir / m_json_object["image_dir"]).string();
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
  LOGIP("image_path: %s\n", image_path.c_str());

  // 创建缓冲区并读取音频文件的二进制内容，如果读取失败则断言失败
  std::vector<uint8_t> buffer;
  ASSERT_TRUE(CommonUtils::readBinaryFile(image_path, buffer));
  for (int i = 0; i < sample_rate_ * 0.5 * 2;
       i++) {  // pad 0.5s, *2 for uint8_t to int16
    buffer.push_back(0);
  }
  // 获取填充后的音频数据总大小
  int frame_size = buffer.size();
  // 创建一个BaseImage对象，用来承载音频数据
  std::shared_ptr<BaseImage> bin_data = ImageFactory::createImage(
      frame_size, 1, ImageFormat::GRAY, TDLDataType::UINT8, true);
  if (!bin_data) {
    LOGE("performance: failed to create image %s", image_path.c_str());
  }
  // 获取图像对象的第一个通道的虚拟地址阵阵
  uint8_t* data_buffer = bin_data->getVirtualAddress()[0];
  // 将音频数据拷贝到图像对象的内存中
  memcpy(data_buffer, buffer.data(), frame_size * sizeof(uint8_t));
  // images.push_back(ImgItem{iter.key(), image_path, bin_data});

  std::string model_id_encoder = std::string(m_json_object["model_id_encoder"]);

  if (model_id_encoder == "RECOGNITION_SPEECH_ZIPFORMER_ENCODER") {
    run_performance(model_path_encoder, bin_data, zipformer_model_, 1000.0);
  }

}  // end of TEST_F
}  // namespace unitest
}  // namespace cvitdl
