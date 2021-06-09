#include <fstream>
#include <string>
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "json.hpp"

#define DISTANCE_BIAS 2.0

bool is_close(cvai_4_pts_t *pts_1, cvai_4_pts_t *pts_2) {
  for (int i = 0; i < 4; i++) {
    if (ABS(pts_1->x[i] - pts_2->x[i]) > DISTANCE_BIAS) return false;
    if (ABS(pts_1->y[i] - pts_2->y[i]) > DISTANCE_BIAS) return false;
  }
  return true;
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
  std::string model_dir = std::string(argv[1]);
  std::string image_dir = std::string(argv[2]);

  nlohmann::json m_json_read;
  std::ofstream m_ofs_results;

  std::ifstream filestr(argv[3]);
  filestr >> m_json_read;
  filestr.close();

  std::string model_name = std::string(m_json_read["reg_config"][0]["model_name"]);
  std::string model_path = model_dir + "/" + model_name;
  int img_num = int(m_json_read["reg_config"][0]["image_num"]);

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
  ret = CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_WPODNET, model_path.c_str());
  if (ret != CVI_SUCCESS) {
    printf("Set license plate detection model failed with %#x!\n", ret);
    return ret;
  }

  bool pass = true;
  for (int img_idx = 0; img_idx < img_num; img_idx++) {
    std::string image_path =
        image_dir + "/" + std::string(m_json_read["reg_config"][0]["test_images"][img_idx]);
    printf("[%d] %s\n", img_idx, image_path.c_str());

    VB_BLK blk_fr;
    VIDEO_FRAME_INFO_S frame;
    CVI_S32 ret = CVI_AI_ReadImage(image_path.c_str(), &blk_fr, &frame, PIXEL_FORMAT_RGB_888);
    if (ret != CVI_SUCCESS) {
      printf("Read image failed with %#x!\n", ret);
      return ret;
    }

    cvai_object_t vehicle_obj;
    memset(&vehicle_obj, 0, sizeof(cvai_object_t));
    vehicle_obj.size = 1;
    vehicle_obj.height = frame.stVFrame.u32Height;
    vehicle_obj.width = frame.stVFrame.u32Width;
    vehicle_obj.info = (cvai_object_info_t *)malloc(vehicle_obj.size * sizeof(cvai_object_info_t));
    vehicle_obj.info[0].bbox.x1 = 0;
    vehicle_obj.info[0].bbox.y1 = 0;
    vehicle_obj.info[0].bbox.x2 = vehicle_obj.width - 1;
    vehicle_obj.info[0].bbox.y2 = vehicle_obj.height - 1;
    vehicle_obj.info[0].feature.size = 0;
    vehicle_obj.info[0].feature.ptr = NULL;
    vehicle_obj.info[0].pedestrian_properity = NULL;

    CVI_AI_LicensePlateDetection(ai_handle, &frame, &vehicle_obj);

    if (!vehicle_obj.info[0].vehicle_properity) {
      pass = false;
      // printf("WARNING: license plate not found.\n");
    } else {
      cvai_4_pts_t *pred = &vehicle_obj.info[0].vehicle_properity->license_pts;
      cvai_4_pts_t *expected_res = new cvai_4_pts_t;
      for (int i = 0; i < 4; i++) {
        expected_res->x[i] =
            float(m_json_read["reg_config"][0]["expected_results"][img_idx][2 * i]);
        expected_res->y[i] =
            float(m_json_read["reg_config"][0]["expected_results"][img_idx][2 * i + 1]);
      }
      pass &= is_close(pred, expected_res);
#if 0
      printf("license plate: (%f,%f,%f,%f,%f,%f,%f,%f)\n", pred->x[0], pred->y[0], pred->x[1],
             pred->y[1], pred->x[2], pred->y[2], pred->x[3], pred->y[3]);
      printf("expected: (%f,%f,%f,%f,%f,%f,%f,%f)\n", expected_res->x[0], expected_res->y[0],
             expected_res->x[1], expected_res->y[1], expected_res->x[2], expected_res->y[2],
             expected_res->x[3], expected_res->y[3]);
#endif
      delete expected_res;
    }
    // printf(" > %s\n", (pass ? "PASS" : "FAILURE"));

    CVI_AI_Free(&vehicle_obj);
    CVI_VB_ReleaseBlock(blk_fr);
  }
  printf("Regression Result: %s\n", (pass ? "PASS" : "FAILURE"));

  CVI_AI_DestroyHandle(ai_handle);
  CVI_SYS_Exit();
}
