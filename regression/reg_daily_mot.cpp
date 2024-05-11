#include <experimental/filesystem>
#include <fstream>
#include <string>
#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_test.hpp"
#include "gtest.h"
#include "json.hpp"
#include "raii.hpp"
#include "regression_utils.hpp"

#define BIAS 1.0

namespace fs = std::experimental::filesystem;
namespace cvitdl {
namespace unitest {

class MultiObjectTrackingTestSuite : public CVI_TDLModelTestSuite {
 public:
  MultiObjectTrackingTestSuite() : CVI_TDLModelTestSuite("daily_reg_MOT.json", "") {}

  virtual ~MultiObjectTrackingTestSuite() = default;

  struct MOTResult {
    uint64_t unique_id;
    int state;
    cvtdl_bbox_t bbox;
  };

  std::vector<std::string> TARGETS = {"pedestrian", "vehicle", "pet", "face"};
  std::map<std::string, cvtdl_deepsort_config_t> m_config;

  bool SetTestConfig(cvtdl_deepsort_config_t &conf, std::string target);

 protected:
  virtual void SetUp() {
    m_tdl_handle = NULL;
    ASSERT_EQ(CVI_TDL_CreateHandle2(&m_tdl_handle, 1, 0), CVI_TDL_SUCCESS);
    for (size_t i = 0; i < TARGETS.size(); i++) {
      cvtdl_deepsort_config_t conf;
      ASSERT_EQ(CVI_TDL_DeepSORT_GetDefaultConfig(&conf), CVI_TDL_SUCCESS);
      ASSERT_EQ(SetTestConfig(conf, TARGETS[i]), true);
      m_config[TARGETS[i]] = conf;
    }
  }

  virtual void TearDown() {
    CVI_TDL_DestroyHandle(m_tdl_handle);
    m_tdl_handle = NULL;
  }
};

bool MultiObjectTrackingTestSuite::SetTestConfig(cvtdl_deepsort_config_t &conf,
                                                 std::string target) {
  if (target == "pedestrian") {
    return true;
  } else if (target == "vehicle") {
    conf.ktracker_conf.max_unmatched_num = 20;
    return true;
  } else if (target == "pet") {
    conf.ktracker_conf.max_unmatched_num = 30;
    conf.ktracker_conf.accreditation_threshold = 5;
    conf.ktracker_conf.P_beta[2] = 0.1;
    conf.ktracker_conf.P_beta[6] = 2.5 * 1e-2;
    conf.kfilter_conf.Q_beta[2] = 0.1;
    conf.kfilter_conf.Q_beta[6] = 2.5 * 1e-2;
    return true;
  } else if (target == "face") {
    conf.ktracker_conf.max_unmatched_num = 10;
    conf.ktracker_conf.accreditation_threshold = 10;
    conf.ktracker_conf.P_beta[2] = 0.1;
    conf.ktracker_conf.P_beta[6] = 2.5e-2;
    conf.kfilter_conf.Q_beta[2] = 0.1;
    conf.kfilter_conf.Q_beta[6] = 2.5e-2;
    return true;
  } else {
    return false;
  }
}

TEST_F(MultiObjectTrackingTestSuite, init) {
  ASSERT_EQ(CVI_TDL_DeepSORT_Init(m_tdl_handle, false), CVI_TDL_SUCCESS);
  cvtdl_deepsort_config_t conf;
  ASSERT_EQ(CVI_TDL_DeepSORT_GetDefaultConfig(&conf), CVI_TDL_SUCCESS);
  ASSERT_EQ(CVI_TDL_DeepSORT_SetConfig(m_tdl_handle, &conf, -1, false), CVI_TDL_SUCCESS);
}

TEST_F(MultiObjectTrackingTestSuite, accruacy) {
  int reg_num = int(m_json_object["reg_num"]);

  for (int reg_idx = 0; reg_idx < reg_num; reg_idx++) {
    int img_num = int(m_json_object["reg_config"][reg_idx]["image_num"]);

    std::string target = std::string(m_json_object["reg_config"][reg_idx]["target"]);
    bool use_ReID = 1 == int(m_json_object["reg_config"][reg_idx]["use_ReID"]);
    std::string type = std::string(m_json_object["reg_config"][reg_idx]["type"]);
    int feature_size = int(m_json_object["reg_config"][reg_idx]["feature_size"]);

    // Init DeepSORT
    ASSERT_EQ(CVI_TDL_DeepSORT_Init(m_tdl_handle, false), CVI_TDL_SUCCESS);
    ASSERT_EQ(CVI_TDL_DeepSORT_SetConfig(m_tdl_handle, &m_config[target], -1, false),
              CVI_TDL_SUCCESS);

    for (int img_idx = 0; img_idx < img_num; img_idx++) {
      cvtdl_object_t obj_meta;
      cvtdl_face_t face_meta;
      cvtdl_tracker_t tracker_meta;
      memset(&obj_meta, 0, sizeof(cvtdl_object_t));
      memset(&face_meta, 0, sizeof(cvtdl_face_t));
      memset(&tracker_meta, 0, sizeof(cvtdl_tracker_t));

      int output_size = int(m_json_object["reg_config"][reg_idx]["output_size"][img_idx]);

      MOTResult *expected_res = new MOTResult[output_size];
      for (int out_i = 0; out_i < output_size; out_i++) {
        int unique_id = int(
            m_json_object["reg_config"][reg_idx]["expected_results"][img_idx][out_i]["unique_id"]);
        expected_res[out_i].unique_id = static_cast<uint64_t>(unique_id);
        expected_res[out_i].state =
            int(m_json_object["reg_config"][reg_idx]["expected_results"][img_idx][out_i]["state"]);
        expected_res[out_i].bbox.x1 =
            float(m_json_object["reg_config"][reg_idx]["expected_results"][img_idx][out_i]["x1"]);
        expected_res[out_i].bbox.y1 =
            float(m_json_object["reg_config"][reg_idx]["expected_results"][img_idx][out_i]["y1"]);
        expected_res[out_i].bbox.x2 =
            float(m_json_object["reg_config"][reg_idx]["expected_results"][img_idx][out_i]["x2"]);
        expected_res[out_i].bbox.y2 =
            float(m_json_object["reg_config"][reg_idx]["expected_results"][img_idx][out_i]["y2"]);
      }
      if (type == "obj") {
        // *******************************************
        // Detection & ReID
        obj_meta.size = static_cast<uint32_t>(output_size);
        obj_meta.info = (cvtdl_object_info_t *)malloc(obj_meta.size * sizeof(cvtdl_object_info_t));
        memset(obj_meta.info, 0, obj_meta.size * sizeof(cvtdl_object_info_t));
        for (int out_i = 0; out_i < output_size; out_i++) {
          obj_meta.info[out_i].classes =
              int(m_json_object["reg_config"][reg_idx]["bbox_info"][img_idx][out_i]["classes"]);
          obj_meta.info[out_i].bbox.x1 =
              float(m_json_object["reg_config"][reg_idx]["bbox_info"][img_idx][out_i]["x1"]);
          obj_meta.info[out_i].bbox.y1 =
              float(m_json_object["reg_config"][reg_idx]["bbox_info"][img_idx][out_i]["y1"]);
          obj_meta.info[out_i].bbox.x2 =
              float(m_json_object["reg_config"][reg_idx]["bbox_info"][img_idx][out_i]["x2"]);
          obj_meta.info[out_i].bbox.y2 =
              float(m_json_object["reg_config"][reg_idx]["bbox_info"][img_idx][out_i]["y2"]);
          if (use_ReID) {
            obj_meta.info[out_i].feature.size = static_cast<uint32_t>(feature_size);
            obj_meta.info[out_i].feature.type = TYPE_INT8;
            obj_meta.info[out_i].feature.ptr = (int8_t *)malloc(feature_size * sizeof(int8_t));
            for (int feat_i = 0; feat_i < feature_size; feat_i++) {
              int value =
                  int(m_json_object["reg_config"][reg_idx]["feature_info"][img_idx][out_i][feat_i]);
              obj_meta.info[out_i].feature.ptr[feat_i] = static_cast<int8_t>(value);
            }
          }
        }
        ASSERT_EQ(CVI_TDL_DeepSORT_Obj(m_tdl_handle, &obj_meta, &tracker_meta, use_ReID),
                  CVI_TDL_SUCCESS);
        // *******************************************

      } else { /* face */
        // *******************************************
        // Detection & ReID
        face_meta.size = static_cast<uint32_t>(output_size);
        face_meta.info = (cvtdl_face_info_t *)malloc(face_meta.size * sizeof(cvtdl_face_info_t));
        memset(face_meta.info, 0, face_meta.size * sizeof(cvtdl_face_info_t));
        for (int out_i = 0; out_i < output_size; out_i++) {
          face_meta.info[out_i].bbox.x1 =
              float(m_json_object["reg_config"][reg_idx]["bbox_info"][img_idx][out_i]["x1"]);
          face_meta.info[out_i].bbox.y1 =
              float(m_json_object["reg_config"][reg_idx]["bbox_info"][img_idx][out_i]["y1"]);
          face_meta.info[out_i].bbox.x2 =
              float(m_json_object["reg_config"][reg_idx]["bbox_info"][img_idx][out_i]["x2"]);
          face_meta.info[out_i].bbox.y2 =
              float(m_json_object["reg_config"][reg_idx]["bbox_info"][img_idx][out_i]["y2"]);
        }
        ASSERT_EQ(CVI_TDL_DeepSORT_Face(m_tdl_handle, &face_meta, &tracker_meta), CVI_TDL_SUCCESS);
        // *******************************************
      }

      for (uint32_t i = 0; i < tracker_meta.size; i++) {
        uint64_t unique_id =
            (type == "obj") ? obj_meta.info[i].unique_id : face_meta.info[i].unique_id;
        ASSERT_EQ(unique_id, expected_res[i].unique_id);
        ASSERT_EQ(tracker_meta.info[i].state, expected_res[i].state);
        ASSERT_LT(ABS(tracker_meta.info[i].bbox.x1 - expected_res[i].bbox.x1), BIAS);
        ASSERT_LT(ABS(tracker_meta.info[i].bbox.y1 - expected_res[i].bbox.y1), BIAS);
        ASSERT_LT(ABS(tracker_meta.info[i].bbox.x2 - expected_res[i].bbox.x2), BIAS);
        ASSERT_LT(ABS(tracker_meta.info[i].bbox.y2 - expected_res[i].bbox.y2), BIAS);
#if 0
        printf("[%u] (%llu,%d,%f,%f,%f,%f) expected: (%llu,%d,%f,%f,%f,%f)\n", i, unique_id,
               tracker_meta.info[i].state, tracker_meta.info[i].bbox.x1,
               tracker_meta.info[i].bbox.y1, tracker_meta.info[i].bbox.x2,
               tracker_meta.info[i].bbox.y2, expected_res[i].unique_id, expected_res[i].state,
               expected_res[i].bbox.x1, expected_res[i].bbox.y1, expected_res[i].bbox.x2,
               expected_res[i].bbox.y2);
#endif
      }
      delete[] expected_res;

      CVI_TDL_Free(&obj_meta);
      CVI_TDL_Free(&face_meta);
      CVI_TDL_Free(&tracker_meta);
    }
  }
}

}  // namespace unitest
}  // namespace cvitdl