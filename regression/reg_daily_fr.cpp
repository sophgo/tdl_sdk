#include <fstream>
#include <string>
#include <unordered_map>
#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_evaluation.h"
#include "cvi_tdl_media.h"
#include "cvi_tdl_test.hpp"
#include "json.hpp"
#include "raii.hpp"
#include "regression_utils.hpp"

namespace fs = std::experimental::filesystem;
namespace cvitdl {
namespace unitest {

class FaceRecognitionTestSuite : public CVI_TDLModelTestSuite {
 public:
  FaceRecognitionTestSuite() : CVI_TDLModelTestSuite("reg_daily_FR.json", "reg_daily_fr") {}

  virtual ~FaceRecognitionTestSuite() = default;

  std::string m_model_path;

 protected:
  virtual void SetUp() {
    m_tdl_handle = NULL;
    ASSERT_EQ(CVI_TDL_CreateHandle2(&m_tdl_handle, 1, 0), CVI_TDL_SUCCESS);
    ASSERT_EQ(CVI_TDL_SetVpssTimeout(m_tdl_handle, 1000), CVI_TDL_SUCCESS);
  }

  virtual void TearDown() {
    CVI_TDL_DestroyHandle(m_tdl_handle);
    m_tdl_handle = NULL;
  }
};

TEST_F(FaceRecognitionTestSuite, open_close_model) {
  for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
    std::string model_name = std::string(m_json_object[test_index]["model_name"]);
    m_model_path = (m_model_dir / fs::path(model_name)).string();
    std::cout << m_model_path << std::endl;
    ASSERT_EQ(CVI_TDL_OpenModel(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_FACERECOGNITION,
                                m_model_path.c_str()),
              CVI_TDL_SUCCESS)
        << "failed to set model path: " << m_model_path.c_str();
    ASSERT_EQ(CVI_TDL_CloseModel(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_FACERECOGNITION),
              CVI_TDL_SUCCESS);
  }
}

TEST_F(FaceRecognitionTestSuite, inference) {
  for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
    std::string model_name = std::string(m_json_object[test_index]["model_name"]);
    m_model_path = (m_model_dir / fs::path(model_name)).string();

    TDLModelHandler tdlmodel(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_FACERECOGNITION,
                             m_model_path.c_str(), false);
    ASSERT_NO_FATAL_FAILURE(tdlmodel.open());
    std::string image_path =
        (m_image_dir / std::string(m_json_object[test_index]["same_pairs"][0][0])).string();

    // test inference for PIXEL_FORMAT_RGB_888 format.
    {
      Image image_rgb(image_path, PIXEL_FORMAT_RGB_888);
      ASSERT_TRUE(image_rgb.open());

      // test 1 face
      {
        TDLObject<cvtdl_face_t> face_meta;
        init_face_meta(face_meta, 1);
        ASSERT_EQ(CVI_TDL_FaceRecognition(m_tdl_handle, image_rgb.getFrame(), face_meta),
                  CVI_TDL_SUCCESS);
      }

      // test 10 faces
      {
        TDLObject<cvtdl_face_t> face_meta;
        init_face_meta(face_meta, 10);
        ASSERT_EQ(CVI_TDL_FaceRecognition(m_tdl_handle, image_rgb.getFrame(), face_meta),
                  CVI_TDL_SUCCESS);
      }
    }

    // inference for PIXEL_FORMAT_RGB_888_PLANAR format.
    {
      Image image_rgb(image_path, PIXEL_FORMAT_RGB_888_PLANAR);
      ASSERT_TRUE(image_rgb.open());
      TDLObject<cvtdl_face_t> face_meta;
      init_face_meta(face_meta, 1);

      ASSERT_EQ(CVI_TDL_FaceRecognition(m_tdl_handle, image_rgb.getFrame(), face_meta),
                CVI_TDL_SUCCESS);
    }
  }
}

TEST_F(FaceRecognitionTestSuite, accuracy) {
  for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
    std::string model_name = std::string(m_json_object[test_index]["model_name"]);
    m_model_path = (m_model_dir / fs::path(model_name)).string();

    ASSERT_EQ(CVI_TDL_OpenModel(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_FACERECOGNITION,
                                m_model_path.c_str()),
              CVI_TDL_SUCCESS);

    std::vector<std::pair<std::string, std::string>> pair_info = {
        {"same_pairs", "same_scores"},
        {"diff_pairs", "diff_scores"},
    };
    auto test_config = m_json_object[test_index];
    for (auto pair_test : pair_info) {
      for (size_t pair_idx = 0; pair_idx < test_config[pair_test.first].size(); pair_idx++) {
        auto pair = test_config[pair_test.first][pair_idx];
        float expected_score = test_config[pair_test.second][pair_idx];

        std::string image_path1 = (m_image_dir / std::string(pair[0])).string();
        std::string image_path2 = (m_image_dir / std::string(pair[1])).string();

        TDLObject<cvtdl_face_t> face_meta1;
        {
          Image image(image_path1, PIXEL_FORMAT_RGB_888);
          ASSERT_NO_FATAL_FAILURE(image.open());
          init_face_meta(face_meta1, 1);
          ASSERT_EQ(CVI_TDL_FaceRecognition(m_tdl_handle, image.getFrame(), face_meta1),
                    CVI_TDL_SUCCESS);
        }

        TDLObject<cvtdl_face_t> face_meta2;
        {
          Image image(image_path2, PIXEL_FORMAT_RGB_888);
          ASSERT_NO_FATAL_FAILURE(image.open());
          init_face_meta(face_meta2, 1);
          ASSERT_EQ(CVI_TDL_FaceRecognition(m_tdl_handle, image.getFrame(), face_meta2),
                    CVI_TDL_SUCCESS);
        }

        float score = 0.0;
        cvitdl_service_handle_t m_service_handle;
        CVI_TDL_Service_CreateHandle(&m_service_handle, m_tdl_handle);
        CVI_TDL_Service_CalculateSimilarity(m_service_handle, &face_meta1->info[0].feature,
                                            &face_meta2->info[0].feature, &score);

        EXPECT_LT(std::abs(score - expected_score), 0.9);
        CVI_TDL_FreeCpp(face_meta1);
        CVI_TDL_FreeCpp(face_meta2);
      }
    }
  }
}
}  // namespace unitest
}  // namespace cvitdl
