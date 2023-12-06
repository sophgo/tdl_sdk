#include <fstream>
#include <string>
#include <unordered_map>

#include <gtest.h>
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

class PersonPet_DetectionTestSuite : public CVI_TDLModelTestSuite {
 public:
  PersonPet_DetectionTestSuite() : CVI_TDLModelTestSuite("daily_reg_PET.json", "reg_daily_pet") {}

  virtual ~PersonPet_DetectionTestSuite() = default;

  std::string m_model_path;

 protected:
  virtual void SetUp() {
    std::string model_name = std::string(m_json_object["model_name"]);
    m_model_path = (m_model_dir / fs::path(model_name)).string();

    m_tdl_handle = NULL;
    ASSERT_EQ(CVI_TDL_CreateHandle2(&m_tdl_handle, 1, 0), CVI_TDL_SUCCESS);
    ASSERT_EQ(CVI_TDL_SetVpssTimeout(m_tdl_handle, 1000), CVI_TDL_SUCCESS);
  }

  virtual void TearDown() {
    CVI_TDL_DestroyHandle(m_tdl_handle);
    m_tdl_handle = NULL;
  }
};

TEST_F(PersonPet_DetectionTestSuite, open_close_model) {
  ASSERT_EQ(CVI_TDL_OpenModel(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_PERSON_PETS_DETECTION,
                              m_model_path.c_str()),
            CVI_TDL_SUCCESS)
      << "failed to set model path: " << m_model_path.c_str();

  const char *model_path_get =
      CVI_TDL_GetModelPath(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_PERSON_PETS_DETECTION);

  EXPECT_PRED2([](auto s1, auto s2) { return s1 == s2; }, m_model_path,
               std::string(model_path_get));

  ASSERT_EQ(CVI_TDL_CloseModel(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_PERSON_PETS_DETECTION),
            CVI_TDL_SUCCESS);
}

TEST_F(PersonPet_DetectionTestSuite, accuracy) {
  ASSERT_EQ(CVI_TDL_OpenModel(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_PERSON_PETS_DETECTION,
                              m_model_path.c_str()),
            CVI_TDL_SUCCESS);

  int img_num = int(m_json_object["image_num"]);
  auto results = m_json_object["results"];
  for (nlohmann::json::iterator iter = results.begin(); iter != results.end(); iter++) {
    std::string image_path = (m_image_dir / iter.key()).string();
    Image image(image_path, PIXEL_FORMAT_RGB_888);
    ASSERT_TRUE(image.open());
    VIDEO_FRAME_INFO_S *vframe = image.getFrame();
    TDLObject<cvtdl_object_t> vehicle_meta;
    init_obj_meta(vehicle_meta, 1, vframe->stVFrame.u32Height, vframe->stVFrame.u32Width, 0);
    ASSERT_EQ(CVI_TDL_PersonPet_Detection(m_tdl_handle, vframe, vehicle_meta), CVI_TDL_SUCCESS);
    printf("boxes===================================\n");
    for (uint32_t i = 0; i < vehicle_meta->size; i++) {
      printf("bbox.x1 = %f\n", vehicle_meta->info[i].bbox.x1);
      printf("bbox.y1 = %f\n", vehicle_meta->info[i].bbox.y1);
      printf("bbox.x2 = %f\n", vehicle_meta->info[i].bbox.x2);
      printf("bbox.y2 = %f\n", vehicle_meta->info[i].bbox.y2);
      printf("bbox.classes = %d\n", vehicle_meta->info[i].classes);
      printf("bbox.score = %f\n", vehicle_meta->info[i].bbox.score);

      // ss << "[" << vehicle_meta->info[i].bbox.x1 << "," << vehicle_meta->info[i].bbox.y1 << ","
      //    << vehicle_meta->info[i].bbox.x2 << "," << vehicle_meta->info[i].bbox.y2 << ","
      //    << vehicle_meta->info[i].classes << "," << vehicle_meta->info[i].bbox.score << "],";
    }
    printf("boxes===================================\n");
    CVI_TDL_FreeCpp(vehicle_meta);
    // delete expected_res;
  }
}

}  // namespace unitest
}  // namespace cvitdl
