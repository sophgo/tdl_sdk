#include <gtest.h>
#include <vector>
#include "cvi_tdl.h"
#include "cvi_tdl_test.hpp"
#include "cvi_vpss.h"
#include "raii.hpp"

namespace fs = std::experimental::filesystem;

namespace cvitdl {
namespace unitest {

class CoreTestSuite : public CVI_TDLTestSuite {
 public:
  CoreTestSuite() {}

  virtual ~CoreTestSuite() = default;

 protected:
  virtual void SetUp() {}

  virtual void TearDown() {}
};

TEST_F(CoreTestSuite, create_handle) {
  // create handle with default vpss group id
  cvitdl_handle_t tdl_handle = NULL;
  EXPECT_EQ(CVI_TDL_CreateHandle(&tdl_handle), CVI_TDL_SUCCESS);
  ASSERT_TRUE(tdl_handle != NULL);
  VPSS_GRP *groups;
  uint32_t num_vpss_used;
  CVI_TDL_GetVpssGrpIds(tdl_handle, &groups, &num_vpss_used);
  EXPECT_EQ(num_vpss_used, (uint32_t)1);
  EXPECT_EQ(groups[0], (VPSS_GRP)0);
  free(groups);
  EXPECT_EQ(CVI_TDL_DestroyHandle(tdl_handle), CVI_TDL_SUCCESS);

  // create handle with specific vpss group id
  tdl_handle = NULL;
  groups = NULL;
  num_vpss_used = 0;
  EXPECT_EQ(CVI_TDL_CreateHandle2(&tdl_handle, 1, 0), CVI_TDL_SUCCESS);
  ASSERT_TRUE(tdl_handle != NULL);
  CVI_TDL_GetVpssGrpIds(tdl_handle, &groups, &num_vpss_used);
  EXPECT_EQ(num_vpss_used, (uint32_t)1);
  EXPECT_EQ(groups[0], (VPSS_GRP)1);
  free(groups);
  EXPECT_EQ(CVI_TDL_DestroyHandle(tdl_handle), CVI_TDL_SUCCESS);

  // create handle with invalid vpss group id
  tdl_handle = NULL;
  EXPECT_EQ(CVI_TDL_CreateHandle2(&tdl_handle, VPSS_MAX_GRP_NUM, 0), CVI_TDL_ERR_INIT_VPSS);
  EXPECT_TRUE(tdl_handle == NULL);

  // create handle with occurpied vpss group id
  {
    tdl_handle = NULL;
    VpssPreprocessor occurpied_vpss(0, 0, 100, 100, PIXEL_FORMAT_BGR_888);
    ASSERT_NO_FATAL_FAILURE(occurpied_vpss.open());
    EXPECT_EQ(CVI_TDL_CreateHandle2(&tdl_handle, 0, 0), CVI_TDL_SUCCESS);
    ASSERT_TRUE(tdl_handle != NULL);

    Image image(PIXEL_FORMAT_RGB_888, 1920, 1080);
    ASSERT_NO_FATAL_FAILURE(image.open());

    CVI_TDLTestContext &context = CVI_TDLTestContext::getInstance();

    fs::path model_base_path = context.getModelBaseDir();
    fs::path mobiledet_path = model_base_path / "mobiledetv2-person-vehicle-ls.cvimodel";

    EXPECT_EQ(CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE,
                                mobiledet_path.c_str()),
              CVI_TDL_SUCCESS);

    TDLObject<cvtdl_object_t> obj_meta;
    EXPECT_EQ(CVI_TDL_MobileDetV2_Person_Vehicle(tdl_handle, image.getFrame(), obj_meta),
              CVI_TDL_ERR_INIT_VPSS);
    EXPECT_EQ(CVI_TDL_DestroyHandle(tdl_handle), CVI_TDL_SUCCESS);
  }

  // create handle with VPSS_MODE_SINGLE
  {
    tdl_handle = NULL;
#ifndef ATHENA2
    CVI_SYS_SetVPSSMode(VPSS_MODE_SINGLE);
#endif
    EXPECT_EQ(CVI_TDL_CreateHandle2(&tdl_handle, 0, 1), CVI_TDL_SUCCESS);
    ASSERT_TRUE(tdl_handle != NULL);

    Image image(PIXEL_FORMAT_RGB_888, 1920, 1080);
    ASSERT_NO_FATAL_FAILURE(image.open());

    CVI_TDLTestContext &context = CVI_TDLTestContext::getInstance();

    fs::path model_base_path = context.getModelBaseDir();
    fs::path mobiledet_path = model_base_path / "mobiledetv2-person-vehicle-ls.cvimodel";

    EXPECT_EQ(CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE,
                                mobiledet_path.c_str()),
              CVI_TDL_SUCCESS);

    TDLObject<cvtdl_object_t> obj_meta;
    EXPECT_EQ(CVI_TDL_MobileDetV2_Person_Vehicle(tdl_handle, image.getFrame(), obj_meta),
              CVI_TDL_SUCCESS);
    EXPECT_EQ(CVI_TDL_DestroyHandle(tdl_handle), CVI_TDL_SUCCESS);
  }

  // create handle with VPSS_MODE_DUAL
  {
    tdl_handle = NULL;
#ifndef ATHENA2
    CVI_SYS_SetVPSSMode(VPSS_MODE_DUAL);
#endif
    EXPECT_EQ(CVI_TDL_CreateHandle2(&tdl_handle, 0, 1), CVI_SUCCESS);
    ASSERT_TRUE(tdl_handle != NULL);

    Image image(PIXEL_FORMAT_RGB_888, 1920, 1080);
    ASSERT_NO_FATAL_FAILURE(image.open());

    CVI_TDLTestContext &context = CVI_TDLTestContext::getInstance();

    fs::path model_base_path = context.getModelBaseDir();
    fs::path mobiledet_path = model_base_path / "mobiledetv2-person-vehicle-ls.cvimodel";

    EXPECT_EQ(CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE,
                                mobiledet_path.c_str()),
              CVI_TDL_SUCCESS);

    TDLObject<cvtdl_object_t> obj_meta;
    EXPECT_EQ(CVI_TDL_MobileDetV2_Person_Vehicle(tdl_handle, image.getFrame(), obj_meta),
              CVI_TDL_SUCCESS);

    ASSERT_TRUE(tdl_handle != NULL);
    EXPECT_EQ(CVI_TDL_DestroyHandle(tdl_handle), CVI_TDL_SUCCESS);
  }

  // create multiple handles at the same time.
  {
    std::vector<cvitdl_handle_t> handles(VPSS_MAX_GRP_NUM - 1, NULL);
    for (size_t i = 0; i < handles.size(); i++) {
      EXPECT_EQ(CVI_TDL_CreateHandle2(&handles[i], (uint32_t)i, 0), CVI_TDL_SUCCESS);
      ASSERT_TRUE(handles[i] != NULL);
    }

    for (size_t i = 0; i < handles.size(); i++) {
      EXPECT_EQ(CVI_TDL_DestroyHandle(handles[i]), CVI_TDL_SUCCESS);
    }
  }
}

TEST_F(CoreTestSuite, skip_vpsspreprocess) {
  const VPSS_GRP VPSS_GRP_ID = 0;
  const VPSS_CHN VPSS_CHN_ID = 0;
  VpssPreprocessor preprocessor(VPSS_GRP_ID, VPSS_CHN_ID, 100, 100, PIXEL_FORMAT_RGB_888);
  ASSERT_NO_FATAL_FAILURE(preprocessor.open());

  cvitdl_handle_t tdl_handle = NULL;
#ifndef ATHENA2
  CVI_SYS_SetVPSSMode(VPSS_MODE_DUAL);
#endif
  EXPECT_EQ(CVI_TDL_CreateHandle2(&tdl_handle, 1, 1), CVI_SUCCESS);
  ASSERT_TRUE(tdl_handle != NULL);

  Image image(PIXEL_FORMAT_RGB_888, 1920, 1080);
  ASSERT_NO_FATAL_FAILURE(image.open());
  VIDEO_FRAME_INFO_S *frame = image.getFrame();

  CVI_TDLTestContext &context = CVI_TDLTestContext::getInstance();

  fs::path model_base_path = context.getModelBaseDir();
  fs::path mobiledet_path = model_base_path / "mobiledetv2-person-vehicle-ls.cvimodel";

  EXPECT_EQ(CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE,
                              mobiledet_path.c_str()),
            CVI_TDL_SUCCESS);
  EXPECT_EQ(CVI_TDL_SetSkipVpssPreprocess(tdl_handle,
                                          CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE, true),
            CVI_TDL_SUCCESS);

  cvtdl_vpssconfig_t vpssconfig;
  ASSERT_EQ(
      CVI_TDL_GetVpssChnConfig(tdl_handle, CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE,
                               frame->stVFrame.u32Width, frame->stVFrame.u32Height, 0, &vpssconfig),
      CVI_TDL_SUCCESS);

  std::shared_ptr<VIDEO_FRAME_INFO_S> output_frame(
      {new VIDEO_FRAME_INFO_S, [this](VIDEO_FRAME_INFO_S *f) {
         CVI_VPSS_ReleaseChnFrame(VPSS_GRP_ID, VPSS_CHN_ID, f);
         delete f;
       }});
  preprocessor.resetVpss(image, vpssconfig);
  preprocessor.preprocess(frame, output_frame.get());

  TDLObject<cvtdl_object_t> obj_meta;
  EXPECT_EQ(CVI_TDL_MobileDetV2_Person_Vehicle(tdl_handle, output_frame.get(), obj_meta),
            CVI_TDL_SUCCESS);

  // TDL SDK won't create VPSS if we skip vpss preporcessing.
  EXPECT_EQ(CVI_VPSS_GetAvailableGrp(), (VPSS_GRP)2);

  EXPECT_EQ(CVI_TDL_DestroyHandle(tdl_handle), CVI_TDL_SUCCESS);
}

TEST_F(CoreTestSuite, set_modelpath) {
  cvitdl_handle_t tdl_handle = NULL;
  EXPECT_EQ(CVI_TDL_CreateHandle(&tdl_handle), CVI_TDL_SUCCESS);
  ASSERT_TRUE(tdl_handle != NULL);
  CVI_TDLTestContext &context = CVI_TDLTestContext::getInstance();

  fs::path model_base_path = context.getModelBaseDir();
  fs::path mobiledet_path = model_base_path / "mobiledetv2-person-vehicle-ls.cvimodel";

  EXPECT_EQ(CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE,
                              mobiledet_path.c_str()),
            CVI_TDL_SUCCESS);
  ASSERT_STREQ(CVI_TDL_GetModelPath(tdl_handle, CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE),
               mobiledet_path.c_str());

  // should be failed if set path again.
  EXPECT_EQ(CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE,
                              mobiledet_path.c_str()),
            CVI_TDL_ERR_MODEL_INITIALIZED);

  // set invalid model path
  EXPECT_EQ(CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_COCO80, "fake path"),
            CVI_TDL_ERR_INVALID_MODEL_PATH);

  EXPECT_EQ(CVI_TDL_DestroyHandle(tdl_handle), CVI_TDL_SUCCESS);
}

TEST_F(CoreTestSuite, set_vpss_thread) {
  cvitdl_handle_t tdl_handle = NULL;
  EXPECT_EQ(CVI_TDL_CreateHandle(&tdl_handle), CVI_TDL_SUCCESS);
  ASSERT_TRUE(tdl_handle != NULL);

  Image image(PIXEL_FORMAT_RGB_888, 1920, 1080);
  ASSERT_NO_FATAL_FAILURE(image.open());

  CVI_TDLTestContext &context = CVI_TDLTestContext::getInstance();

  fs::path model_base_path = context.getModelBaseDir();
  fs::path mobiledet_path = model_base_path / "mobiledetv2-person-vehicle-ls.cvimodel";

  EXPECT_EQ(CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE,
                              mobiledet_path.c_str()),
            CVI_TDL_SUCCESS);

  // test default vpss thread id
  uint32_t thread_id;
  VPSS_GRP *groups;
  uint32_t num;
  EXPECT_EQ(CVI_TDL_GetVpssGrpIds(tdl_handle, &groups, &num), CVI_TDL_SUCCESS);
  EXPECT_EQ(num, (uint32_t)1);
  free(groups);

  EXPECT_EQ(CVI_TDL_GetVpssThread(tdl_handle, CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE,
                                  &thread_id),
            CVI_TDL_SUCCESS);
  EXPECT_EQ(thread_id, (uint32_t)0);

  // create second vpss group for model
  EXPECT_EQ(
      CVI_TDL_SetVpssThread(tdl_handle, CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE, 1),
      CVI_TDL_SUCCESS);
  groups = NULL;
  num = 0;
  EXPECT_EQ(CVI_TDL_GetVpssGrpIds(tdl_handle, &groups, &num), CVI_TDL_SUCCESS);
  EXPECT_EQ(num, (uint32_t)2);
  free(groups);

  EXPECT_EQ(CVI_TDL_GetVpssThread(tdl_handle, CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE,
                                  &thread_id),
            CVI_TDL_SUCCESS);
  EXPECT_EQ(thread_id, (uint32_t)1);

  {
    TDLObject<cvtdl_object_t> obj_meta;
    EXPECT_EQ(CVI_TDL_MobileDetV2_Person_Vehicle(tdl_handle, image.getFrame(), obj_meta),
              CVI_TDL_SUCCESS);
  }

  // create third vpss group for model
  EXPECT_EQ(CVI_TDL_SetVpssThread2(tdl_handle, CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE,
                                   2, 2, 0),
            CVI_TDL_SUCCESS);
  groups = NULL;
  num = 0;
  EXPECT_EQ(CVI_TDL_GetVpssGrpIds(tdl_handle, &groups, &num), CVI_TDL_SUCCESS);
  EXPECT_EQ(num, (uint32_t)3);
  free(groups);

  EXPECT_EQ(CVI_TDL_GetVpssThread(tdl_handle, CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE,
                                  &thread_id),
            CVI_TDL_SUCCESS);
  EXPECT_EQ(thread_id, (uint32_t)2);

  {
    TDLObject<cvtdl_object_t> obj_meta;
    EXPECT_EQ(CVI_TDL_MobileDetV2_Person_Vehicle(tdl_handle, image.getFrame(), obj_meta),
              CVI_TDL_SUCCESS);
  }

  EXPECT_EQ(CVI_TDL_DestroyHandle(tdl_handle), CVI_TDL_SUCCESS);
}

}  // namespace unitest
}  // namespace cvitdl
