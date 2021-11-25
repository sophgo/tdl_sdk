#include <fstream>
#include <string>
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "cviai_test.hpp"
#include "evaluation/cviai_evaluation.h"
#include "evaluation/cviai_media.h"
#include "raii.hpp"
#include "regression_utils.hpp"

namespace cviai {
namespace unitest {

class ESCTestSuite : public CVIAIModelTestSuite {
 public:
  typedef CVI_S32 (*InferenceFunc)(const cviai_handle_t, VIDEO_FRAME_INFO_S *, int *);
  struct ModelInfo {
    InferenceFunc inference;
    CVI_AI_SUPPORTED_MODEL_E index;
    std::string model_path;
  };

  ESCTestSuite()
      : CVIAIModelTestSuite("reg_daily_es_classification.json", "reg_daily_es_classification") {}

  virtual ~ESCTestSuite() = default;

 protected:
  virtual void SetUp() {
    m_ai_handle = NULL;
    ASSERT_EQ(CVI_AI_CreateHandle2(&m_ai_handle, 1, 0), CVIAI_SUCCESS);
  }

  virtual void TearDown() {
    CVI_AI_DestroyHandle(m_ai_handle);
    m_ai_handle = NULL;
  }

  ModelInfo getModel(const std::string &model_name);
};

ESCTestSuite::ModelInfo ESCTestSuite::getModel(const std::string &model_name) {
  ModelInfo model_info;
  std::string model_path = (m_model_dir / model_name).string();
  model_info.index = CVI_AI_SUPPORTED_MODEL_SOUNDCLASSIFICATION;
  model_info.inference = CVI_AI_SoundClassification;
  model_info.model_path = model_path;
  return model_info;
}

TEST_F(ESCTestSuite, open_close_model) {
  for (size_t test_idx = 0; test_idx < m_json_object.size(); test_idx++) {
    auto test_config = m_json_object[test_idx];
    std::string model_name = std::string(std::string(test_config["model_name"]).c_str());

    ModelInfo model_info = getModel(model_name);
    ASSERT_LT(model_info.index, CVI_AI_SUPPORTED_MODEL_END);

    AIModelHandler aimodel(m_ai_handle, model_info.index, model_info.model_path.c_str(), false);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());

    const char *model_path_get = CVI_AI_GetModelPath(m_ai_handle, model_info.index);

    EXPECT_PRED2([](auto s1, auto s2) { return s1 == s2; }, model_info.model_path,
                 std::string(model_path_get));
  }
}

TEST_F(ESCTestSuite, inference_and_accuracy) {
  for (size_t test_idx = 0; test_idx < m_json_object.size(); test_idx++) {
    auto test_config = m_json_object[test_idx];
    std::string model_name = std::string(std::string(test_config["model_name"]).c_str());

    ModelInfo model_info = getModel(model_name);
    ASSERT_LT(model_info.index, CVI_AI_SUPPORTED_MODEL_END);

    AIModelHandler aimodel(m_ai_handle, model_info.index, model_info.model_path.c_str(), false);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());

    for (size_t audio_idx = 0; audio_idx < test_config["test_audios"].size(); audio_idx++) {
      std::string audio_path =
          (m_image_dir / std::string(test_config["test_audios"][audio_idx])).string();
      int expected_res = int(test_config["expected_results"][audio_idx]);

      FILE *fp = fopen(audio_path.c_str(), "rb");
      ASSERT_TRUE(fp);
      fseek(fp, 0, SEEK_END);
      int size = (int)ftell(fp) * sizeof(char);
      CVI_U8 *temp = (CVI_U8 *)malloc(size);
      fseek(fp, 0, SEEK_SET);
      fread(temp, 1, size, fp);
      fclose(fp);
      VIDEO_FRAME_INFO_S frame;
      frame.stVFrame.pu8VirAddr[0] = temp;
      frame.stVFrame.u32Height = 1;
      frame.stVFrame.u32Width = size;
      int index = -1;
      ASSERT_EQ(model_info.inference(m_ai_handle, &frame, &index), CVIAI_SUCCESS);
      EXPECT_EQ(index, expected_res);
      free(temp);
    }
  }
}
}  // namespace unitest
}  // namespace cviai
