#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "sample_comm.h"

#include <cvi_sys.h>
#include <cvi_vb.h>
#include <cvi_vi.h>
#include "cviai_perfetto.h"

#include <dirent.h>
#include <inttypes.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ive/ive.h"

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#define USE_MOBILEDETV2_VEHICLE 0

int main(int argc, char *argv[]) {
  if (argc != 7) {
    printf(
        "Usage: %s <vehicle_detection_model_path>\n"
        "          <license_plate_detection_model_path>\n"
        "          <license_plate_recognition_model_path>\n"
        "          <images_path>\n"
        "          <evaluate_json>\n"
        "          <result_path>\n",
        argv[0]);
    return CVI_FAILURE;
  }
  if (access(argv[1], F_OK) != 0) {
    printf("check model fail: %s\n", argv[1]);
    return CVI_FAILURE;
  }
  if (access(argv[2], F_OK) != 0) {
    printf("check model fail: %s\n", argv[2]);
    return CVI_FAILURE;
  }
  if (access(argv[3], F_OK) != 0) {
    printf("check model fail: %s\n", argv[3]);
    return CVI_FAILURE;
  }
  if (access(argv[5], F_OK) != 0) {
    printf("check json fail: %s\n", argv[5]);
    return CVI_FAILURE;
  }
  DIR *dir = opendir(argv[4]);
  if (dir) {
    closedir(dir);
  } else {
    printf("check images folder fail: %s\n", argv[4]);
    return CVI_FAILURE;
  }

  CVI_AI_PerfettoInit();
  CVI_S32 ret = CVI_SUCCESS;

  // Init VB pool size.
  const CVI_S32 vpssgrp_width = 1920;
  const CVI_S32 vpssgrp_height = 1080;
  ret = MMF_INIT_HELPER(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, vpssgrp_width,
                        vpssgrp_height, PIXEL_FORMAT_RGB_888_PLANAR);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  cviai_handle_t ai_handle = NULL;
  ret = CVI_AI_CreateHandle2(&ai_handle, 1);
#if USE_MOBILEDETV2_VEHICLE
  ret |= CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE_D0, argv[1]);
#else
  ret |= CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D0, argv[1]);
#endif
  ret |= CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_WPODNET, argv[2]);
  ret |= CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_LPRNET, argv[3]);
  if (ret != CVI_SUCCESS) {
    printf("open failed with %#x!\n", ret);
    return ret;
  }

  cviai_eval_handle_t eval_handle;
  ret = CVI_AI_Eval_CreateHandle(&eval_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create Eval handle failed with %#x!\n", ret);
    return ret;
  }

  uint32_t image_num;
  CVI_AI_Eval_LPDRInit(eval_handle, argv[4], argv[5], &image_num);

  FILE *outFile;
  outFile = fopen(argv[6], "w");
  if (outFile == NULL) {
    printf("There is a problem opening the output file: %s\n", argv[6]);
    exit(EXIT_FAILURE);
  }
  fprintf(outFile, "%u\n", image_num);

  for (uint32_t n = 0; n < image_num; n++) {
    char *filename = NULL;
    int id = 0;
    CVI_AI_Eval_LPDRGetImageIdPair(eval_handle, n, &filename, &id);

    printf("[%u] image path = %s\n", n, filename);

    VB_BLK blk;
    VIDEO_FRAME_INFO_S frame;
    CVI_S32 ret = CVI_AI_ReadImage(filename, &blk, &frame, PIXEL_FORMAT_RGB_888);
    if (ret != CVI_SUCCESS) {
      printf("Read image failed with %#x!\n", ret);
      return ret;
    }

    cvai_object_t vehicle_obj;
    printf("CVI_AI_MobileDetV2_D0 ... start\n");
#if USE_MOBILEDETV2_VEHICLE
    CVI_AI_MobileDetV2_Vehicle_D0(ai_handle, &frame, &vehicle_obj);
#else
    cvai_obj_det_type_e det_type = CVI_DET_TYPE_VEHICLE;
    CVI_AI_MobileDetV2_D0(ai_handle, &frame, &vehicle_obj, det_type);
#endif
    printf("Find %u vehicles.\n", vehicle_obj.size);

    /* LP Detection */
    cvai_object_t license_plate_obj;
    license_plate_obj.size = vehicle_obj.size;
    license_plate_obj.info =
        (cvai_object_info_t *)malloc(sizeof(cvai_object_info_t) * vehicle_obj.size);
    memset(license_plate_obj.info, 0, sizeof(cvai_object_info_t) * vehicle_obj.size);

    printf("CVI_AI_LicensePlateDetection ... start\n");
    CVI_AI_LicensePlateDetection(ai_handle, &frame, &vehicle_obj, &license_plate_obj);

    /* LP Recognition */
    printf("CVI_AI_LicensePlateRecognition ... start\n");
    CVI_AI_LicensePlateRecognition(ai_handle, &frame, &license_plate_obj);

    int counter = 0;
    for (size_t i = 0; i < license_plate_obj.size; i++) {
      if (license_plate_obj.info[i].bpts.size > 0) {
        printf("Vec[%zu] ID number: %s\n", i, license_plate_obj.info[i].name);
        counter += 1;
      } else {
        printf("Vec[%zu] license plate not found.\n", i);
      }
    }

    printf("counter = %d\n", counter);
    fprintf(outFile, "%u,%u\n", n, license_plate_obj.size);
    for (size_t i = 0; i < license_plate_obj.size; i++) {
      if (license_plate_obj.info[i].bpts.size > 0) {
        fprintf(outFile, "%s,%f,%f,%f,%f,%s,%f,%f,%f,%f,%f,%f,%f,%f\n", vehicle_obj.info[i].name,
                vehicle_obj.info[i].bbox.x1, vehicle_obj.info[i].bbox.y1,
                vehicle_obj.info[i].bbox.x2, vehicle_obj.info[i].bbox.y2,
                license_plate_obj.info[i].name, license_plate_obj.info[i].bpts.x[0],
                license_plate_obj.info[i].bpts.y[0], license_plate_obj.info[i].bpts.x[1],
                license_plate_obj.info[i].bpts.y[1], license_plate_obj.info[i].bpts.x[2],
                license_plate_obj.info[i].bpts.y[2], license_plate_obj.info[i].bpts.x[3],
                license_plate_obj.info[i].bpts.y[3]);
      } else {
        fprintf(outFile, "%s,%f,%f,%f,%f,NULL,,,,,,,,\n", vehicle_obj.info[i].name,
                vehicle_obj.info[i].bbox.x1, vehicle_obj.info[i].bbox.y1,
                vehicle_obj.info[i].bbox.x2, vehicle_obj.info[i].bbox.y2);
      }
    }

    CVI_AI_Free(&vehicle_obj);
    CVI_AI_Free(&license_plate_obj);
    CVI_VB_ReleaseBlock(blk);
  }

  fclose(outFile);

  CVI_AI_DestroyHandle(ai_handle);
  CVI_AI_Eval_DestroyHandle(eval_handle);
  CVI_SYS_Exit();
}