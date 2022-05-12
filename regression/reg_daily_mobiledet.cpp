#include <fstream>
#include <string>
#include <unordered_map>

#include <gtest.h>
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "cviai_test.hpp"
#include "evaluation/cviai_evaluation.h"
#include "evaluation/cviai_media.h"
#include "json.hpp"
#include "raii.hpp"
#include "regression_utils.hpp"

namespace cviai {
namespace unitest {

class MobileDetV2TestSuite : public CVIAIModelTestSuite {
 public:
  typedef int (*InferenceFunc)(cviai_handle_t, VIDEO_FRAME_INFO_S *, cvai_object_t *);
  struct ModelInfo {
    InferenceFunc inference;
    CVI_AI_SUPPORTED_MODEL_E index;
    std::string model_path;
  };

  MobileDetV2TestSuite() : CVIAIModelTestSuite("daily_reg_mobiledet.json", "reg_daily_mobildet") {}

  virtual ~MobileDetV2TestSuite() = default;

 protected:
  virtual void SetUp() {
    ASSERT_EQ(CVI_AI_CreateHandle2(&m_ai_handle, 1, 0), CVIAI_SUCCESS);
    ASSERT_EQ(CVI_AI_SetVpssTimeout(m_ai_handle, 1000), CVIAI_SUCCESS);
  }

  virtual void TearDown() {
    CVI_AI_DestroyHandle(m_ai_handle);
    m_ai_handle = NULL;
  }

  ModelInfo getModel(const std::string &model_name);

  static const std::unordered_map<std::string, std::pair<CVI_AI_SUPPORTED_MODEL_E, InferenceFunc>>
      MODEL_MAP;

  const float bbox_threhold = 0.90;
  const float score_threshold = 0.1;
};

const std::unordered_map<std::string,
                         std::pair<CVI_AI_SUPPORTED_MODEL_E, MobileDetV2TestSuite::InferenceFunc>>
    MobileDetV2TestSuite::MODEL_MAP = {
        {"mobiledetv2-d0-ls.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_COCO80, CVI_AI_MobileDetV2_COCO80}},
        {"mobiledetv2-d0.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_COCO80, CVI_AI_MobileDetV2_COCO80}},
        {"mobiledetv2-d1-ls.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_COCO80, CVI_AI_MobileDetV2_COCO80}},
        {"mobiledetv2-d1.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_COCO80, CVI_AI_MobileDetV2_COCO80}},
        {"mobiledetv2-d2-ls.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_COCO80, CVI_AI_MobileDetV2_COCO80}},
        {"mobiledetv2-d2.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_COCO80, CVI_AI_MobileDetV2_COCO80}},
        {"mobiledetv2-person-vehicle.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE, CVI_AI_MobileDetV2_Person_Vehicle}},
        {"mobiledetv2-lite-person-pets-ls.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PERSON_PETS, CVI_AI_MobileDetV2_Person_Pets}},
        {"mobiledetv2-lite-person-pets.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PERSON_PETS, CVI_AI_MobileDetV2_Person_Pets}},
        {"mobiledetv2-person-vehicle-ls.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE, CVI_AI_MobileDetV2_Person_Vehicle}},
        {"mobiledetv2-person-vehicle-ls-768.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE, CVI_AI_MobileDetV2_Person_Vehicle}},
        {"mobiledetv2-pedestrian-d0-ls-384.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN, CVI_AI_MobileDetV2_Pedestrian}},
        {"mobiledetv2-pedestrian-d0-ls-640.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN, CVI_AI_MobileDetV2_Pedestrian}},
        {"mobiledetv2-pedestrian-d0-ls-768.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN, CVI_AI_MobileDetV2_Pedestrian}},
        {"mobiledetv2-pedestrian-d0-ls.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN, CVI_AI_MobileDetV2_Pedestrian}},
        {"mobiledetv2-pedestrian-d0.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN, CVI_AI_MobileDetV2_Pedestrian}},
        {"mobiledetv2-pedestrian-d1-ls.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN, CVI_AI_MobileDetV2_Pedestrian}},
        {"mobiledetv2-pedestrian-d1-ls-1024.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN, CVI_AI_MobileDetV2_Pedestrian}},
        {"mobiledetv2-pedestrian-d1.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN, CVI_AI_MobileDetV2_Pedestrian}},
        {"mobiledetv2-vehicle-d0-ls.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE, CVI_AI_MobileDetV2_Vehicle}},
        {"mobiledetv2-vehicle-d0.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE, CVI_AI_MobileDetV2_Vehicle}},
};

MobileDetV2TestSuite::ModelInfo MobileDetV2TestSuite::getModel(const std::string &model_name) {
  ModelInfo model_info;
  model_info.index = CVI_AI_SUPPORTED_MODEL_END;

  std::string model_path = (m_model_dir / model_name).string();

  auto iter = MODEL_MAP.find(model_name);
  if (iter != MODEL_MAP.end()) {
    model_info.index = iter->second.first;
    model_info.inference = iter->second.second;
    model_info.model_path = model_path;
  }
  return model_info;
}

TEST_F(MobileDetV2TestSuite, open_close_model) {
  ModelInfo model_info = getModel("mobiledetv2-lite-person-pets.cvimodel");
  ASSERT_LT(model_info.index, CVI_AI_SUPPORTED_MODEL_END);

  ASSERT_EQ(CVI_AI_OpenModel(m_ai_handle, model_info.index, model_info.model_path.c_str()),
            CVIAI_SUCCESS)
      << "failed to set model path: " << model_info.model_path;

  const char *model_path_get = CVI_AI_GetModelPath(m_ai_handle, model_info.index);

  EXPECT_PRED2([](auto s1, auto s2) { return s1 == s2; }, model_info.model_path,
               std::string(model_path_get));

  ASSERT_EQ(CVI_AI_CloseModel(m_ai_handle, model_info.index), CVIAI_SUCCESS);
}

TEST_F(MobileDetV2TestSuite, get_vpss_config) {
  ModelInfo model_info = getModel("mobiledetv2-person-vehicle-ls.cvimodel");
  ASSERT_LT(model_info.index, CVI_AI_SUPPORTED_MODEL_END);

  cvai_vpssconfig_t vpssconfig;
  vpssconfig.chn_attr.u32Height = 200;
  vpssconfig.chn_attr.u32Width = 200;
  vpssconfig.chn_attr.enPixelFormat = PIXEL_FORMAT_ARGB_1555;
  vpssconfig.chn_attr.stNormalize.bEnable = false;
  EXPECT_EQ(CVI_AI_GetVpssChnConfig(m_ai_handle, model_info.index, 1920, 1080, 0, &vpssconfig),
            CVIAI_ERR_NOT_YET_INITIALIZED);
  EXPECT_EQ(vpssconfig.chn_attr.u32Height, (uint32_t)200);
  EXPECT_EQ(vpssconfig.chn_attr.u32Width, (uint32_t)200);
  EXPECT_EQ(vpssconfig.chn_attr.enPixelFormat, PIXEL_FORMAT_ARGB_1555);
  EXPECT_EQ(vpssconfig.chn_attr.stNormalize.bEnable, false);

  AIModelHandler aimodel(m_ai_handle, model_info.index, model_info.model_path, true);
  ASSERT_NO_FATAL_FAILURE(aimodel.open());

  uint32_t dstWidth = 640;
  uint32_t dstHeight = 384;
  uint32_t srcWidth = 1920;
  uint32_t srcHeight = 1080;

  EXPECT_EQ(CVI_AI_GetVpssChnConfig(m_ai_handle, model_info.index, 1920, 1080, 0, &vpssconfig),
            CVIAI_SUCCESS);

  float ratio_w = (float)dstWidth / srcWidth;
  float ratio_h = (float)dstHeight / srcHeight;
  float ratio = std::min(ratio_w, ratio_h);
  float factor[3] = {0.82988566, 0.84841, 0.844639};
  float mean[3] = {102.63611, 98.6531, 87.4455};

  EXPECT_EQ(vpssconfig.chn_attr.u32Width, dstWidth);
  EXPECT_EQ(vpssconfig.chn_attr.u32Width, dstWidth);
  EXPECT_EQ(vpssconfig.chn_attr.u32Height, dstHeight);
  EXPECT_EQ(vpssconfig.chn_attr.enVideoFormat, VIDEO_FORMAT_LINEAR);
  EXPECT_EQ(vpssconfig.chn_attr.enPixelFormat, PIXEL_FORMAT_RGB_888_PLANAR);
  EXPECT_EQ(vpssconfig.chn_attr.stFrameRate.s32SrcFrameRate, -1);
  EXPECT_EQ(vpssconfig.chn_attr.stFrameRate.s32DstFrameRate, -1);
  EXPECT_EQ(vpssconfig.chn_attr.u32Depth, (uint32_t)1);
  EXPECT_EQ(vpssconfig.chn_attr.bMirror, CVI_FALSE);
  EXPECT_EQ(vpssconfig.chn_attr.bFlip, CVI_FALSE);
  EXPECT_EQ(vpssconfig.chn_attr.stAspectRatio.enMode, ASPECT_RATIO_MANUAL);
  EXPECT_EQ(vpssconfig.chn_attr.stAspectRatio.stVideoRect.s32X, 0);
  EXPECT_EQ(vpssconfig.chn_attr.stAspectRatio.stVideoRect.s32Y, 0);
  EXPECT_EQ(vpssconfig.chn_attr.stAspectRatio.stVideoRect.u32Width,
            uint32_t((srcWidth * ratio) + 0.5));
  EXPECT_EQ(vpssconfig.chn_attr.stAspectRatio.stVideoRect.u32Height,
            uint32_t((srcHeight * ratio) + 0.5));
  EXPECT_EQ(vpssconfig.chn_attr.stAspectRatio.bEnableBgColor, CVI_TRUE);
  EXPECT_EQ(vpssconfig.chn_attr.stNormalize.bEnable, CVI_TRUE);
  for (uint32_t i = 0; i < 3; i++) {
    EXPECT_FLOAT_EQ(vpssconfig.chn_attr.stNormalize.factor[i], factor[i]);
  }
  for (uint32_t i = 0; i < 3; i++) {
    EXPECT_FLOAT_EQ(vpssconfig.chn_attr.stNormalize.mean[i], mean[i]);
  }
  EXPECT_EQ(vpssconfig.chn_attr.stNormalize.rounding, VPSS_ROUNDING_TO_EVEN);
}

TEST_F(MobileDetV2TestSuite, skip_vpsspreprocess) {
  const VPSS_GRP VPSS_GRP_ID = 0;
  const VPSS_CHN VPSS_CHN_ID = 0;
  VpssPreprocessor preprocessor(VPSS_GRP_ID, VPSS_CHN_ID, 100, 100, PIXEL_FORMAT_RGB_888);
  ASSERT_NO_FATAL_FAILURE(preprocessor.open());

  for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
    std::string model_name = std::string(m_json_object[test_index]["model_name"]);
    ModelInfo model_info = getModel(model_name);
    ASSERT_LT(model_info.index, CVI_AI_SUPPORTED_MODEL_END) << "model name: " << model_name;

    std::string img_name = m_json_object[test_index]["results"].begin().key();
    std::string image_path = (m_image_dir / img_name).string();
    Image image(image_path, PIXEL_FORMAT_RGB_888);
    ASSERT_TRUE(image.open());

    // test with skip vpss = false;
    AIObject<cvai_object_t> obj_meta_w_preproc;
    {
      AIModelHandler aimodel(m_ai_handle, model_info.index, model_info.model_path, false);
      ASSERT_NO_FATAL_FAILURE(aimodel.open());
      CVI_AI_SetModelThreshold(m_ai_handle, model_info.index, 0.6);

      ASSERT_EQ(model_info.inference(m_ai_handle, image.getFrame(), obj_meta_w_preproc),
                CVIAI_SUCCESS);
      ASSERT_GT(obj_meta_w_preproc->size, (uint32_t)0);
    }

    // test with skip vpss = true;
    AIObject<cvai_object_t> obj_meta_no_preproc;
    {
      VIDEO_FRAME_INFO_S *frame = image.getFrame();
      AIModelHandler aimodel(m_ai_handle, model_info.index, model_info.model_path, true);
      ASSERT_NO_FATAL_FAILURE(aimodel.open());
      CVI_AI_SetModelThreshold(m_ai_handle, model_info.index, 0.6);

      cvai_vpssconfig_t vpssconfig;
      ASSERT_EQ(CVI_AI_GetVpssChnConfig(m_ai_handle, model_info.index, frame->stVFrame.u32Width,
                                        frame->stVFrame.u32Height, 0, &vpssconfig),
                CVIAI_SUCCESS);

      std::shared_ptr<VIDEO_FRAME_INFO_S> output_frame(
          {new VIDEO_FRAME_INFO_S, [this](VIDEO_FRAME_INFO_S *f) {
             CVI_VPSS_ReleaseChnFrame(VPSS_GRP_ID, VPSS_CHN_ID, f);
             delete f;
           }});
      preprocessor.resetVpss(image, vpssconfig);
      preprocessor.preprocess(frame, output_frame.get());

      ASSERT_EQ(output_frame->stVFrame.u32Width, vpssconfig.chn_attr.u32Width);
      ASSERT_EQ(output_frame->stVFrame.u32Height, vpssconfig.chn_attr.u32Height);
      ASSERT_EQ(output_frame->stVFrame.enPixelFormat, vpssconfig.chn_attr.enPixelFormat);
      ASSERT_EQ(model_info.inference(m_ai_handle, output_frame.get(), obj_meta_no_preproc),
                CVIAI_SUCCESS);
      CVI_AI_RescaleMetaRB(frame, obj_meta_no_preproc);

      ASSERT_GT(obj_meta_no_preproc->size, (uint32_t)0);
      ASSERT_EQ(obj_meta_no_preproc->size, obj_meta_w_preproc->size);
      EXPECT_EQ(obj_meta_no_preproc->width, obj_meta_w_preproc->width);
      EXPECT_EQ(obj_meta_no_preproc->height, obj_meta_w_preproc->height);

      for (uint32_t oid = 0; oid < obj_meta_no_preproc->size; oid++) {
        EXPECT_EQ(obj_meta_no_preproc->info[oid].classes, obj_meta_w_preproc->info[oid].classes);
        EXPECT_FLOAT_EQ(obj_meta_no_preproc->info[oid].bbox.score,
                        obj_meta_w_preproc->info[oid].bbox.score);
        EXPECT_FLOAT_EQ(obj_meta_no_preproc->info[oid].bbox.x1,
                        obj_meta_w_preproc->info[oid].bbox.x1);
        EXPECT_FLOAT_EQ(obj_meta_no_preproc->info[oid].bbox.x2,
                        obj_meta_w_preproc->info[oid].bbox.x2);
        EXPECT_FLOAT_EQ(obj_meta_no_preproc->info[oid].bbox.y1,
                        obj_meta_w_preproc->info[oid].bbox.y1);
        EXPECT_FLOAT_EQ(obj_meta_no_preproc->info[oid].bbox.y2,
                        obj_meta_w_preproc->info[oid].bbox.y2);
      }
    }
  }
}

TEST_F(MobileDetV2TestSuite, set_threshold) {
  std::string model_name = std::string(m_json_object[0]["model_name"]);

  ModelInfo model_info = getModel(model_name);
  ASSERT_LT(model_info.index, CVI_AI_SUPPORTED_MODEL_END);
  const float threshold = 0.1;
  // set threshold before opening model
  {
    ASSERT_EQ(CVI_AI_SetModelThreshold(m_ai_handle, model_info.index, threshold), CVIAI_SUCCESS);
    AIModelHandler aimodel(m_ai_handle, model_info.index, model_info.model_path, false);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());
    float stored_threshold;
    CVI_AI_GetModelThreshold(m_ai_handle, model_info.index, &stored_threshold);
    EXPECT_FLOAT_EQ(stored_threshold, threshold);
  }

  // set threshold after opening model
  {
    AIModelHandler aimodel(m_ai_handle, model_info.index, model_info.model_path, false);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());
    ASSERT_EQ(CVI_AI_SetModelThreshold(m_ai_handle, model_info.index, threshold), CVIAI_SUCCESS);
    float stored_threshold;
    CVI_AI_GetModelThreshold(m_ai_handle, model_info.index, &stored_threshold);
    EXPECT_FLOAT_EQ(stored_threshold, threshold);
  }

  std::string img_name = m_json_object[0]["results"].begin().key();
  std::string image_path = (m_image_dir / img_name).string();
  Image image(image_path, PIXEL_FORMAT_RGB_888);
  ASSERT_TRUE(image.open());

  // check all scores greate than threshold
  {
    AIModelHandler aimodel(m_ai_handle, model_info.index, model_info.model_path, false);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());
    ASSERT_EQ(CVI_AI_SetModelThreshold(m_ai_handle, model_info.index, threshold), CVIAI_SUCCESS);

    AIObject<cvai_object_t> obj_meta;
    model_info.inference(m_ai_handle, image.getFrame(), obj_meta);

    for (uint32_t oid = 0; oid < obj_meta->size; oid++) {
      EXPECT_GE((float)std::ceil(obj_meta->info[oid].bbox.score * 100.0) / 100.0 + 0.001,
                threshold);
    }
  }
}

TEST_F(MobileDetV2TestSuite, select_classes) {
  ModelInfo model_info = getModel("mobiledetv2-d0-ls.cvimodel");
  ASSERT_LT(model_info.index, CVI_AI_SUPPORTED_MODEL_END);
  const float threshold = 0.1;

  std::string img_name = "000000311303.jpg";
  std::string image_path = (m_image_dir / img_name).string();
  Image image(image_path, PIXEL_FORMAT_RGB_888);
  ASSERT_TRUE(image.open());

  // select Dining Table class
  {
    AIModelHandler aimodel(m_ai_handle, model_info.index, model_info.model_path, false);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());
    ASSERT_EQ(CVI_AI_SetModelThreshold(m_ai_handle, model_info.index, threshold), CVIAI_SUCCESS);
    ASSERT_EQ(
        CVI_AI_SelectDetectClass(m_ai_handle, model_info.index, 1, CVI_AI_DET_TYPE_DINING_TABLE),
        CVIAI_SUCCESS);

    AIObject<cvai_object_t> obj_meta;
    model_info.inference(m_ai_handle, image.getFrame(), obj_meta);

    for (uint32_t oid = 0; oid < obj_meta->size; oid++) {
      EXPECT_EQ(obj_meta->info[oid].classes, CVI_AI_DET_TYPE_DINING_TABLE);
    }
  }

  // select Knife and Bus Dining Table
  {
    AIModelHandler aimodel(m_ai_handle, model_info.index, model_info.model_path, false);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());
    ASSERT_EQ(CVI_AI_SetModelThreshold(m_ai_handle, model_info.index, threshold), CVIAI_SUCCESS);
    ASSERT_EQ(CVI_AI_SelectDetectClass(m_ai_handle, model_info.index, 2, CVI_AI_DET_TYPE_KNIFE,
                                       CVI_AI_DET_TYPE_DINING_TABLE),
              CVIAI_SUCCESS);

    AIObject<cvai_object_t> obj_meta;
    model_info.inference(m_ai_handle, image.getFrame(), obj_meta);

    for (uint32_t oid = 0; oid < obj_meta->size; oid++) {
      EXPECT_TRUE(obj_meta->info[oid].classes == CVI_AI_DET_TYPE_KNIFE ||
                  obj_meta->info[oid].classes == CVI_AI_DET_TYPE_DINING_TABLE)
          << "unselected classes: " << obj_meta->info[oid].classes;
    }
  }
}

bool match_detections(cvai_object_t *obj_meta, cvai_bbox_t &expected_bbox, int catId,
                      float bbox_threhold, float score_threshold) {
  bool found = false;
  for (uint32_t actual_det_index = 0; actual_det_index < obj_meta->size; actual_det_index++) {
    if (obj_meta->info[actual_det_index].classes == catId) {
      found =
          iou(obj_meta->info[actual_det_index].bbox, expected_bbox) >= bbox_threhold &&
          abs(obj_meta->info[actual_det_index].bbox.score - expected_bbox.score) < score_threshold;
      if (found) {
        return found;
      }
    }
  }

  return false;
}

TEST_F(MobileDetV2TestSuite, accuracy) {
  for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
    ModelInfo model_info = getModel(std::string(m_json_object[test_index]["model_name"]));
    ASSERT_LT(model_info.index, CVI_AI_SUPPORTED_MODEL_END);

    AIModelHandler aimodel(m_ai_handle, model_info.index, model_info.model_path, false);

    CVI_AI_SetModelThreshold(m_ai_handle, model_info.index, 0.6);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());

    auto results = m_json_object[test_index]["results"];

    for (nlohmann::json::iterator iter = results.begin(); iter != results.end(); iter++) {
      std::string image_path = (m_image_dir / iter.key()).string();
      Image image(image_path, PIXEL_FORMAT_RGB_888);
      ASSERT_TRUE(image.open());

      AIObject<cvai_object_t> obj_meta;

      ASSERT_EQ(model_info.inference(m_ai_handle, image.getFrame(), obj_meta), CVIAI_SUCCESS);

      auto expected_dets = iter.value();

      EXPECT_EQ(obj_meta->size, expected_dets.size()) << "model: " << model_info.model_path << "\n"
                                                      << "image path: " << image_path;

      if (obj_meta->size != expected_dets.size()) {
        continue;
      }

      for (uint32_t det_index = 0; det_index < expected_dets.size(); det_index++) {
        auto bbox = expected_dets[det_index]["bbox"];
        int catId = int(expected_dets[det_index]["category_id"]) - 1;

        cvai_bbox_t expected_bbox = {
            .x1 = float(bbox[0]),
            .y1 = float(bbox[1]),
            .x2 = float(bbox[2]) + float(bbox[0]),
            .y2 = float(bbox[3]) + float(bbox[1]),
            .score = float(expected_dets[det_index]["score"]),
        };

        EXPECT_TRUE(
            match_detections(obj_meta, expected_bbox, catId, bbox_threhold, score_threshold))
            << "image path: " << image_path << "\n"
            << "model path: " << model_info.model_path << "\n"
            << "expected bbox: (" << expected_bbox.x1 << ", " << expected_bbox.y1 << ", "
            << expected_bbox.x2 << ", " << expected_bbox.y2 << ")\n"
            << "score: " << expected_bbox.score << "\n";
      }
    }
  }
}

}  // namespace unitest
}  // namespace cviai
