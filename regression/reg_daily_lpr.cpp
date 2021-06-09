#include <fstream>
#include <string>
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "json.hpp"

enum LicenseFormat { taiwan, china };

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf(
        "Usage: %s <model_dir>\n"
        "          <image_dir>\n"
        "          <regression_json>\n",
        argv[0]);
    return CVI_FAILURE;
  }

  char const *LP_FORMAT[2] = {"TW", "CN"};

  std::string model_dir = std::string(argv[1]);
  std::string image_dir = std::string(argv[2]);

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
  std::string model_name, model_path;
  model_name = std::string(m_json_read["TW"]["reg_config"][0]["model_name"]);
  model_path = model_dir + "/" + model_name;
  ret |= CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_LPRNET_TW, model_path.c_str());

  model_name = std::string(m_json_read["CN"]["reg_config"][0]["model_name"]);
  model_path = model_dir + "/" + model_name;
  ret |= CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_LPRNET_CN, model_path.c_str());
  if (ret != CVI_SUCCESS) {
    printf("Set license plate detection model failed with %#x!\n", ret);
    return ret;
  }

  cvai_object_t vehicle_obj;
  bool pass = true;
  for (int t = 0; t < 2; t++) {
    int img_num = int(m_json_read[LP_FORMAT[t]]["reg_config"][0]["image_num"]);
    enum LicenseFormat license_format;
    if (strcmp(LP_FORMAT[t], "TW") == 0) {
      license_format = taiwan;
    } else if (strcmp(LP_FORMAT[t], "CN") == 0) {
      license_format = china;
    } else {
      printf("Unknown license type %s\n", LP_FORMAT[t]);
      return CVI_FAILURE;
    }
    for (int img_idx = 0; img_idx < img_num; img_idx++) {
      std::string image_path =
          image_dir + "/" +
          std::string(m_json_read[LP_FORMAT[t]]["reg_config"][0]["test_images"][img_idx]);
      // printf("[%d] %s\n", img_idx, image_path.c_str());

      VB_BLK blk_fr;
      VIDEO_FRAME_INFO_S frame;
      CVI_S32 ret = CVI_AI_ReadImage(image_path.c_str(), &blk_fr, &frame, PIXEL_FORMAT_RGB_888);
      if (ret != CVI_SUCCESS) {
        printf("Read image failed with %#x!\n", ret);
        return ret;
      }

      memset(&vehicle_obj, 0, sizeof(cvai_object_t));

      vehicle_obj.size = 1;
      vehicle_obj.height = frame.stVFrame.u32Height;
      vehicle_obj.width = frame.stVFrame.u32Width;
      vehicle_obj.info =
          (cvai_object_info_t *)malloc(vehicle_obj.size * sizeof(cvai_object_info_t));
      vehicle_obj.info[0].bbox.x1 = 0;
      vehicle_obj.info[0].bbox.y1 = 0;
      vehicle_obj.info[0].bbox.x2 = vehicle_obj.width - 1;
      vehicle_obj.info[0].bbox.y2 = vehicle_obj.height - 1;
      vehicle_obj.info[0].feature.size = 0;
      vehicle_obj.info[0].feature.ptr = NULL;
      vehicle_obj.info[0].pedestrian_properity = NULL;
      vehicle_obj.info[0].vehicle_properity =
          (cvai_vehicle_meta *)malloc(sizeof(cvai_vehicle_meta));
      vehicle_obj.info[0].vehicle_properity->license_pts.x[0] = 0.0;
      vehicle_obj.info[0].vehicle_properity->license_pts.x[1] = (float)(vehicle_obj.width);
      vehicle_obj.info[0].vehicle_properity->license_pts.x[2] = (float)(vehicle_obj.width);
      vehicle_obj.info[0].vehicle_properity->license_pts.x[3] = 0.0;
      vehicle_obj.info[0].vehicle_properity->license_pts.y[0] = 0.0;
      vehicle_obj.info[0].vehicle_properity->license_pts.y[1] = 0.0;
      vehicle_obj.info[0].vehicle_properity->license_pts.y[2] = (float)(vehicle_obj.height);
      vehicle_obj.info[0].vehicle_properity->license_pts.y[3] = (float)(vehicle_obj.height);

      switch (license_format) {
        case taiwan:
          CVI_AI_LicensePlateRecognition_TW(ai_handle, &frame, &vehicle_obj);
          break;
        case china:
          CVI_AI_LicensePlateRecognition_CN(ai_handle, &frame, &vehicle_obj);
          break;
        default:
          return CVI_FAILURE;
      }

      std::string expected_res =
          std::string(m_json_read[LP_FORMAT[t]]["reg_config"][0]["expected_results"][img_idx]);
#if 0
      printf("ID number: %s (expected: %s)\n", vehicle_obj.info[0].vehicle_properity->license_char,
             expected_res.c_str());
#endif

      pass = strcmp(vehicle_obj.info[0].vehicle_properity->license_char, expected_res.c_str()) == 0;
      // printf(" > %s\n", (pass ? "PASS" : "FAILURE"));

      CVI_AI_Free(&vehicle_obj);
      CVI_VB_ReleaseBlock(blk_fr);
    }
  }
  printf("Regression Result: %s\n", (pass ? "PASS" : "FAILURE"));

  CVI_AI_DestroyHandle(ai_handle);
  CVI_SYS_Exit();
}
