#include <fstream>
#include <string>
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "json.hpp"

#define WRITE_RESULT_TO_FILE 0

#define BIAS 1.0

typedef struct {
  uint64_t unique_id;
  int state;
  cvai_bbox_t bbox;
} mot_result_t;

void setConfig(cvai_deepsort_config_t &ds_conf, std::string target) {
  if (target == "pedestrian") {
    /* Default Setting:
     *   ds_conf.ktracker_conf.P_std_beta[2] = 0.01;
     *   ds_conf.ktracker_conf.P_std_beta[6] = 1e-5;
     *   ds_conf.kfilter_conf.Q_std_beta[2] = 0.01;
     *   ds_conf.kfilter_conf.Q_std_beta[6] = 1e-5;
     *   ds_conf.kfilter_conf.R_std_beta[2] = 0.1;
     */
    return;
  } else if (target == "vehicle") {
    ds_conf.ktracker_conf.max_unmatched_num = 20;
    return;
  } else if (target == "pet") {
    ds_conf.ktracker_conf.max_unmatched_num = 30;
    ds_conf.ktracker_conf.accreditation_threshold = 5;
    ds_conf.ktracker_conf.P_std_beta[2] = 0.1;
    ds_conf.ktracker_conf.P_std_beta[6] = 2.5 * 1e-2;
    ds_conf.kfilter_conf.Q_std_beta[2] = 0.1;
    ds_conf.kfilter_conf.Q_std_beta[6] = 2.5 * 1e-2;
    return;
  } else if (target == "face") {
    ds_conf.ktracker_conf.max_unmatched_num = 10;
    ds_conf.ktracker_conf.accreditation_threshold = 10;
    ds_conf.ktracker_conf.P_std_beta[2] = 0.1;
    ds_conf.ktracker_conf.P_std_beta[6] = 2.5e-2;
    ds_conf.kfilter_conf.Q_std_beta[2] = 0.1;
    ds_conf.kfilter_conf.Q_std_beta[6] = 2.5e-2;
    return;
  } else {
    printf("Target error.\n");
    exit(-1);
  }
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf(
        "Usage: %s <model_dir>\n"
        "          <image_dir>\n"
        "          <regression_json>\n",
        argv[0]);
    return CVI_FAILURE;
  }
  nlohmann::json m_json_read;
  std::ofstream m_ofs_results;

  std::ifstream filestr(argv[3]);
  filestr >> m_json_read;
  filestr.close();

  CVI_S32 ret = CVI_SUCCESS;
  CVI_S32 vpssgrp_width = 1920;
  CVI_S32 vpssgrp_height = 1080;

  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 5, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888, 5);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  cviai_handle_t ai_handle = NULL;
  ret = CVI_AI_CreateHandle(&ai_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create handle failed with %#x!\n", ret);
    return ret;
  }

  int reg_num = int(m_json_read["reg_num"]);
  printf("regression num = %d\n", reg_num);

  bool pass = true;
  for (int reg_idx = 0; reg_idx < reg_num; reg_idx++) {
    int img_num = int(m_json_read["reg_config"][reg_idx]["image_num"]);

    std::string target = std::string(m_json_read["reg_config"][reg_idx]["target"]);
    bool use_ReID = 1 == int(m_json_read["reg_config"][reg_idx]["use_ReID"]);
    std::string type = std::string(m_json_read["reg_config"][reg_idx]["type"]);
    int feature_size = int(m_json_read["reg_config"][reg_idx]["feature_size"]);

    // Init DeepSORT
    CVI_AI_DeepSORT_Init(ai_handle, false);
    cvai_deepsort_config_t ds_conf;
    CVI_AI_DeepSORT_GetDefaultConfig(&ds_conf);
    setConfig(ds_conf, target);
    CVI_AI_DeepSORT_SetConfig(ai_handle, &ds_conf, -1, true);

    // CVI_AI_DeepSORT_GetDefaultConfig(&ds_conf);
    // setConfig(ds_conf, "pedestrian");
    // CVI_AI_DeepSORT_SetConfig(ai_handle, &ds_conf, CVI_AI_DET_TYPE_PERSON, false);
    // CVI_AI_DeepSORT_GetDefaultConfig(&ds_conf);
    // setConfig(ds_conf, "vehicle");
    // CVI_AI_DeepSORT_SetConfig(ai_handle, &ds_conf, CVI_AI_DET_TYPE_CAR, false);
    // CVI_AI_DeepSORT_SetConfig(ai_handle, &ds_conf, CVI_AI_DET_TYPE_MOTORBIKE, false);
    // CVI_AI_DeepSORT_GetDefaultConfig(&ds_conf);
    // setConfig(ds_conf, "face");
    // CVI_AI_DeepSORT_SetConfig(ai_handle, &ds_conf, -1, false);

    for (int img_idx = 0; img_idx < img_num; img_idx++) {
      // printf("\n[%d/%d]\n", img_idx, img_num);

      cvai_object_t obj_meta;
      cvai_face_t face_meta;
      cvai_tracker_t tracker_meta;
      memset(&obj_meta, 0, sizeof(cvai_object_t));
      memset(&face_meta, 0, sizeof(cvai_face_t));
      memset(&tracker_meta, 0, sizeof(cvai_tracker_t));

      int output_size = int(m_json_read["reg_config"][reg_idx]["output_size"][img_idx]);

      mot_result_t *expected_res = new mot_result_t[output_size];
      for (int out_i = 0; out_i < output_size; out_i++) {
        int unique_id = int(
            m_json_read["reg_config"][reg_idx]["expected_results"][img_idx][out_i]["unique_id"]);
        expected_res[out_i].unique_id = static_cast<uint64_t>(unique_id);
        expected_res[out_i].state =
            int(m_json_read["reg_config"][reg_idx]["expected_results"][img_idx][out_i]["state"]);
        expected_res[out_i].bbox.x1 =
            float(m_json_read["reg_config"][reg_idx]["expected_results"][img_idx][out_i]["x1"]);
        expected_res[out_i].bbox.y1 =
            float(m_json_read["reg_config"][reg_idx]["expected_results"][img_idx][out_i]["y1"]);
        expected_res[out_i].bbox.x2 =
            float(m_json_read["reg_config"][reg_idx]["expected_results"][img_idx][out_i]["x2"]);
        expected_res[out_i].bbox.y2 =
            float(m_json_read["reg_config"][reg_idx]["expected_results"][img_idx][out_i]["y2"]);
      }
      if (type == "obj") {
        // *******************************************
        // Detection & ReID
        obj_meta.size = static_cast<uint32_t>(output_size);
        obj_meta.info = (cvai_object_info_t *)malloc(obj_meta.size * sizeof(cvai_object_info_t));
        memset(obj_meta.info, 0, obj_meta.size * sizeof(cvai_object_info_t));
        for (int out_i = 0; out_i < output_size; out_i++) {
          obj_meta.info[out_i].classes =
              int(m_json_read["reg_config"][reg_idx]["bbox_info"][img_idx][out_i]["classes"]);
          obj_meta.info[out_i].bbox.x1 =
              float(m_json_read["reg_config"][reg_idx]["bbox_info"][img_idx][out_i]["x1"]);
          obj_meta.info[out_i].bbox.y1 =
              float(m_json_read["reg_config"][reg_idx]["bbox_info"][img_idx][out_i]["y1"]);
          obj_meta.info[out_i].bbox.x2 =
              float(m_json_read["reg_config"][reg_idx]["bbox_info"][img_idx][out_i]["x2"]);
          obj_meta.info[out_i].bbox.y2 =
              float(m_json_read["reg_config"][reg_idx]["bbox_info"][img_idx][out_i]["y2"]);
          if (use_ReID) {
            obj_meta.info[out_i].feature.size = static_cast<uint32_t>(feature_size);
            obj_meta.info[out_i].feature.type = TYPE_INT8;
            obj_meta.info[out_i].feature.ptr = (int8_t *)malloc(feature_size * sizeof(int8_t));
            for (int feat_i = 0; feat_i < feature_size; feat_i++) {
              int value =
                  int(m_json_read["reg_config"][reg_idx]["feature_info"][img_idx][out_i][feat_i]);
              obj_meta.info[out_i].feature.ptr[feat_i] = static_cast<int8_t>(value);
            }
          }
        }
        // Tracking
        CVI_AI_DeepSORT_Obj(ai_handle, &obj_meta, &tracker_meta, use_ReID);
        // *******************************************

      } else { /* face */
        // *******************************************
        // Detection & ReID
        face_meta.size = static_cast<uint32_t>(output_size);
        face_meta.info = (cvai_face_info_t *)malloc(face_meta.size * sizeof(cvai_face_info_t));
        memset(face_meta.info, 0, face_meta.size * sizeof(cvai_face_info_t));
        for (int out_i = 0; out_i < output_size; out_i++) {
          face_meta.info[out_i].bbox.x1 =
              float(m_json_read["reg_config"][reg_idx]["bbox_info"][img_idx][out_i]["x1"]);
          face_meta.info[out_i].bbox.y1 =
              float(m_json_read["reg_config"][reg_idx]["bbox_info"][img_idx][out_i]["y1"]);
          face_meta.info[out_i].bbox.x2 =
              float(m_json_read["reg_config"][reg_idx]["bbox_info"][img_idx][out_i]["x2"]);
          face_meta.info[out_i].bbox.y2 =
              float(m_json_read["reg_config"][reg_idx]["bbox_info"][img_idx][out_i]["y2"]);
        }
        // Tracking
        CVI_AI_DeepSORT_Face(ai_handle, &face_meta, &tracker_meta, false);
        // *******************************************
      }

      bool pass_tmp = true;
      for (uint32_t i = 0; i < tracker_meta.size; i++) {
        uint64_t unique_id =
            (type == "obj") ? obj_meta.info[i].unique_id : face_meta.info[i].unique_id;
        pass_tmp &= unique_id == expected_res[i].unique_id;
        pass_tmp &= tracker_meta.info[i].state == expected_res[i].state;
        pass_tmp &= ABS(tracker_meta.info[i].bbox.x1 - expected_res[i].bbox.x1) < BIAS;
        pass_tmp &= ABS(tracker_meta.info[i].bbox.y1 - expected_res[i].bbox.y1) < BIAS;
        pass_tmp &= ABS(tracker_meta.info[i].bbox.x2 - expected_res[i].bbox.x2) < BIAS;
        pass_tmp &= ABS(tracker_meta.info[i].bbox.y2 - expected_res[i].bbox.y2) < BIAS;
#if 0
        printf("[%u] (%lu,%d,%f,%f,%f,%f) expected: (%lu,%d,%f,%f,%f,%f)\n", i, unique_id,
               tracker_meta.info[i].state, tracker_meta.info[i].bbox.x1,
               tracker_meta.info[i].bbox.y1, tracker_meta.info[i].bbox.x2,
               tracker_meta.info[i].bbox.y2, expected_res[i].unique_id, expected_res[i].state,
               expected_res[i].bbox.x1, expected_res[i].bbox.y1, expected_res[i].bbox.x2,
               expected_res[i].bbox.y2);
#endif
      }
      pass &= pass_tmp;
      // printf(" > %s\n", (pass ? "PASS" : "FAILURE"));
      delete[] expected_res;

      CVI_AI_Free(&obj_meta);
      CVI_AI_Free(&face_meta);
      CVI_AI_Free(&tracker_meta);
    }
  }
  printf("Regression Result: %s\n", (pass ? "PASS" : "FAILURE"));

  CVI_AI_DestroyHandle(ai_handle);
  CVI_SYS_Exit();

  return pass ? CVI_SUCCESS : CVI_FAILURE;
}
