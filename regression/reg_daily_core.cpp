#include <gtest.h>
#include <vector>
#include "cvi_vpss.h"
#include "cviai.h"
#include "cviai_core_internal.hpp"
#include "cviai_test.hpp"
#include "raii.hpp"

namespace fs = std::experimental::filesystem;

namespace cviai {
namespace unitest {

class CoreTestSuite : public CVIAITestSuite {
 public:
  CoreTestSuite() {}

  virtual ~CoreTestSuite() = default;

 protected:
  virtual void SetUp() {}

  virtual void TearDown() {}
};

TEST_F(CoreTestSuite, create_handle) {
  // create handle with default vpss group id
  cviai_handle_t ai_handle = NULL;
  EXPECT_EQ(CVI_AI_CreateHandle(&ai_handle), CVIAI_SUCCESS);
  ASSERT_TRUE(ai_handle != NULL);
  VPSS_GRP *groups;
  uint32_t num_vpss_used;
  CVI_AI_GetVpssGrpIds(ai_handle, &groups, &num_vpss_used);
  EXPECT_EQ(num_vpss_used, (uint32_t)1);
  EXPECT_EQ(groups[0], (VPSS_GRP)0);
  free(groups);
  EXPECT_EQ(CVI_AI_DestroyHandle(ai_handle), CVIAI_SUCCESS);

  // create handle with specific vpss group id
  ai_handle = NULL;
  groups = NULL;
  num_vpss_used = 0;
  EXPECT_EQ(CVI_AI_CreateHandle2(&ai_handle, 1, 0), CVIAI_SUCCESS);
  ASSERT_TRUE(ai_handle != NULL);
  CVI_AI_GetVpssGrpIds(ai_handle, &groups, &num_vpss_used);
  EXPECT_EQ(num_vpss_used, (uint32_t)1);
  EXPECT_EQ(groups[0], (VPSS_GRP)1);
  free(groups);
  EXPECT_EQ(CVI_AI_DestroyHandle(ai_handle), CVIAI_SUCCESS);

  // create handle with invalid vpss group id
  ai_handle = NULL;
  EXPECT_EQ(CVI_AI_CreateHandle2(&ai_handle, VPSS_MAX_GRP_NUM, 0), CVIAI_ERR_INIT_VPSS);
  EXPECT_TRUE(ai_handle == NULL);

  // create handle with occurpied vpss group id
  {
    ai_handle = NULL;
    VpssPreprocessor occurpied_vpss(0, 0, 100, 100, PIXEL_FORMAT_BGR_888);
    ASSERT_NO_FATAL_FAILURE(occurpied_vpss.open());
    EXPECT_EQ(CVI_AI_CreateHandle2(&ai_handle, 0, 0), CVIAI_ERR_INIT_VPSS);
    EXPECT_TRUE(ai_handle == NULL);
  }

  // create handle with VPSS_MODE_SINGLE
  ai_handle = NULL;
  CVI_SYS_SetVPSSMode(VPSS_MODE_SINGLE);
  EXPECT_EQ(CVI_AI_CreateHandle2(&ai_handle, 0, 1), CVIAI_SUCCESS);
  VPSS_GRP_ATTR_S grp_attr;
  CVI_VPSS_GetGrpAttr(0, &grp_attr);
  EXPECT_EQ(grp_attr.u8VpssDev, (CVI_U8)0);
  ASSERT_TRUE(ai_handle != NULL);
  EXPECT_EQ(CVI_AI_DestroyHandle(ai_handle), CVIAI_SUCCESS);

  // create handle with VPSS_MODE_DUAL
  ai_handle = NULL;
  CVI_SYS_SetVPSSMode(VPSS_MODE_DUAL);
  EXPECT_EQ(CVI_AI_CreateHandle2(&ai_handle, 0, 1), CVIAI_SUCCESS);
  ASSERT_TRUE(ai_handle != NULL);
  CVI_VPSS_GetGrpAttr(0, &grp_attr);
  EXPECT_EQ(grp_attr.u8VpssDev, (CVI_U8)1);
  EXPECT_EQ(CVI_AI_DestroyHandle(ai_handle), CVIAI_SUCCESS);

  // create multiple handles at the same time.
  {
    std::vector<cviai_handle_t> handles(VPSS_MAX_GRP_NUM - 1, NULL);
    for (size_t i = 0; i < handles.size(); i++) {
      EXPECT_EQ(CVI_AI_CreateHandle2(&handles[i], (uint32_t)i, 0), CVIAI_SUCCESS);
      ASSERT_TRUE(handles[i] != NULL);
    }

    for (size_t i = 0; i < handles.size(); i++) {
      EXPECT_EQ(CVI_AI_DestroyHandle(handles[i]), CVIAI_SUCCESS);
    }
  }
}

TEST_F(CoreTestSuite, set_modelpath) {
  cviai_handle_t ai_handle = NULL;
  EXPECT_EQ(CVI_AI_CreateHandle(&ai_handle), CVIAI_SUCCESS);
  ASSERT_TRUE(ai_handle != NULL);
  CVIAITestContext &context = CVIAITestContext::getInstance();

  fs::path model_base_path = context.getModelBaseDir();
  fs::path mobiledet_path = model_base_path / "mobiledetv2-person-vehicle-ls.cvimodel";

  EXPECT_EQ(CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE,
                             mobiledet_path.c_str()),
            CVIAI_SUCCESS);
  ASSERT_STREQ(CVI_AI_GetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE),
               mobiledet_path.c_str());

  // should be failed if set path again.
  EXPECT_EQ(CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE,
                             mobiledet_path.c_str()),
            CVIAI_ERR_MODEL_INITIALIZED);

  // set invalid model path
  EXPECT_EQ(CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_COCO80, "fake path"),
            CVIAI_ERR_INVALID_MODEL_PATH);

  EXPECT_EQ(CVI_AI_DestroyHandle(ai_handle), CVIAI_SUCCESS);
}

TEST_F(CoreTestSuite, set_vpss_thread) {
  cviai_handle_t ai_handle = NULL;
  EXPECT_EQ(CVI_AI_CreateHandle(&ai_handle), CVIAI_SUCCESS);
  ASSERT_TRUE(ai_handle != NULL);

  Image image(PIXEL_FORMAT_RGB_888, 1920, 1080);
  ASSERT_NO_FATAL_FAILURE(image.open());

  CVIAITestContext &context = CVIAITestContext::getInstance();

  fs::path model_base_path = context.getModelBaseDir();
  fs::path mobiledet_path = model_base_path / "mobiledetv2-person-vehicle-ls.cvimodel";

  EXPECT_EQ(CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE,
                             mobiledet_path.c_str()),
            CVIAI_SUCCESS);

  // test default vpss thread id
  uint32_t thread_id;
  VPSS_GRP *groups;
  uint32_t num;
  EXPECT_EQ(CVI_AI_GetVpssGrpIds(ai_handle, &groups, &num), CVIAI_SUCCESS);
  EXPECT_EQ(num, (uint32_t)1);
  free(groups);

  EXPECT_EQ(CVI_AI_GetVpssThread(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE,
                                 &thread_id),
            CVIAI_SUCCESS);
  EXPECT_EQ(thread_id, (uint32_t)0);

  // create second vpss group for model
  EXPECT_EQ(CVI_AI_SetVpssThread(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE, 1),
            CVIAI_SUCCESS);
  groups = NULL;
  num = 0;
  EXPECT_EQ(CVI_AI_GetVpssGrpIds(ai_handle, &groups, &num), CVIAI_SUCCESS);
  EXPECT_EQ(num, (uint32_t)2);
  free(groups);

  EXPECT_EQ(CVI_AI_GetVpssThread(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE,
                                 &thread_id),
            CVIAI_SUCCESS);
  EXPECT_EQ(thread_id, (uint32_t)1);

  {
    AIObject<cvai_object_t> obj_meta;
    EXPECT_EQ(CVI_AI_MobileDetV2_Person_Vehicle(ai_handle, image.getFrame(), obj_meta),
              CVIAI_SUCCESS);
  }

  // create third vpss group for model
  EXPECT_EQ(
      CVI_AI_SetVpssThread2(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE, 2, 2),
      CVIAI_SUCCESS);
  groups = NULL;
  num = 0;
  EXPECT_EQ(CVI_AI_GetVpssGrpIds(ai_handle, &groups, &num), CVIAI_SUCCESS);
  EXPECT_EQ(num, (uint32_t)3);
  free(groups);

  EXPECT_EQ(CVI_AI_GetVpssThread(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE,
                                 &thread_id),
            CVIAI_SUCCESS);
  EXPECT_EQ(thread_id, (uint32_t)2);

  {
    AIObject<cvai_object_t> obj_meta;
    EXPECT_EQ(CVI_AI_MobileDetV2_Person_Vehicle(ai_handle, image.getFrame(), obj_meta),
              CVIAI_SUCCESS);
  }

  EXPECT_EQ(CVI_AI_DestroyHandle(ai_handle), CVIAI_SUCCESS);
}

}  // namespace unitest
}  // namespace cviai