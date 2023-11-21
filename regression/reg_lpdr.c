#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_evaluation.h"
#include "cvi_tdl_media.h"
#include "cvi_tdl_perfetto.h"

#include <dirent.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
  CVI_TDL_PerfettoInit();
  if (argc != 8) {
    printf(
        "Usage: %s <vehicle_detection_model_path>\n"
        "          <use_mobiledet_vehicle (0/1)>\n"
        "          <license_plate_detection_model_path>\n"
        "          <license_plate_recognition_model_path>\n"
        "          <images_path>\n"
        "          <evaluate_json>\n"
        "          <result_path>\n",
        argv[0]);
    return CVI_TDL_FAILURE;
  }
  if (access(argv[1], F_OK) != 0) {
    printf("check model fail: %s\n", argv[1]);
    return CVI_TDL_FAILURE;
  }
  if (access(argv[3], F_OK) != 0) {
    printf("check model fail: %s\n", argv[2]);
    return CVI_TDL_FAILURE;
  }
  if (access(argv[4], F_OK) != 0) {
    printf("check model fail: %s\n", argv[3]);
    return CVI_TDL_FAILURE;
  }
  if (access(argv[6], F_OK) != 0) {
    printf("check json fail: %s\n", argv[5]);
    return CVI_TDL_FAILURE;
  }
  DIR *dir = opendir(argv[5]);
  if (dir) {
    closedir(dir);
  } else {
    printf("check images folder fail: %s\n", argv[4]);
    return CVI_TDL_FAILURE;
  }

  CVI_S32 ret = CVI_TDL_SUCCESS;

  // Init VB pool size.
  const CVI_S32 vpssgrp_width = 1920;
  const CVI_S32 vpssgrp_height = 1080;
  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 5, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888_PLANAR, 5);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  cvitdl_handle_t tdl_handle = NULL;
  ret = CVI_TDL_CreateHandle2(&tdl_handle, 1, 0);
  int use_vehicle = atoi(argv[2]);
  if (use_vehicle == 1) {
    ret |= CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE, argv[1]);
  } else if (use_vehicle == 0) {
    ret |= CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_COCO80, argv[1]);
    CVI_TDL_SelectDetectClass(tdl_handle, CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_COCO80, 1,
                              CVI_TDL_DET_GROUP_VEHICLE);
  } else {
    printf("Unknow det model type.\n");
    return CVI_TDL_FAILURE;
  }
  ret |= CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_WPODNET, argv[3]);
  ret |= CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_LPRNET_TW, argv[4]);
  if (ret != CVI_TDL_SUCCESS) {
    printf("open failed with %#x!\n", ret);
    return ret;
  }

  cvitdl_eval_handle_t eval_handle;
  ret = CVI_TDL_Eval_CreateHandle(&eval_handle);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Create Eval handle failed with %#x!\n", ret);
    return ret;
  }

  uint32_t image_num;
  CVI_TDL_Eval_LPDRInit(eval_handle, argv[5], argv[6], &image_num);

  FILE *outFile;
  outFile = fopen(argv[7], "w");
  if (outFile == NULL) {
    printf("There is a problem opening the output file: %s\n", argv[6]);
    exit(EXIT_FAILURE);
  }
  fprintf(outFile, "%u\n", image_num);

  cvtdl_object_t vehicle_obj, license_plate_obj;
  memset(&vehicle_obj, 0, sizeof(cvtdl_object_t));
  memset(&license_plate_obj, 0, sizeof(cvtdl_object_t));
  for (uint32_t n = 0; n < image_num; n++) {
    char *filename = NULL;
    int id = 0;
    CVI_TDL_Eval_LPDRGetImageIdPair(eval_handle, n, &filename, &id);

    printf("[%u] image path = %s\n", n, filename);

    VIDEO_FRAME_INFO_S frame;
    CVI_S32 ret = CVI_TDL_ReadImage(filename, &frame, PIXEL_FORMAT_RGB_888);
    if (ret != CVI_TDL_SUCCESS) {
      printf("Read image failed with %#x!\n", ret);
      return ret;
    }

    if (use_vehicle == 1) {
      printf("CVI_TDL_MobileDetV2_Vehicle ... start\n");
      CVI_TDL_MobileDetV2_Vehicle(tdl_handle, &frame, &vehicle_obj);
    } else {
      printf("CVI_TDL_MobileDetV2_COCO80 ... start\n");
      CVI_TDL_MobileDetV2_COCO80(tdl_handle, &frame, &vehicle_obj);
    }
    printf("Find %u vehicles.\n", vehicle_obj.size);

    /* LP Detection */
    printf("CVI_TDL_LicensePlateDetection ... start\n");
    CVI_TDL_LicensePlateDetection(tdl_handle, &frame, &vehicle_obj);

    /* LP Recognition */
    printf("CVI_TDL_LicensePlateRecognition ... start\n");
    CVI_TDL_LicensePlateRecognition_TW(tdl_handle, &frame, &vehicle_obj);

    int counter = 0;
    for (size_t i = 0; i < vehicle_obj.size; i++) {
      if (vehicle_obj.info[i].vehicle_properity) {
        printf("Vec[%zu] ID number: %s\n", i, vehicle_obj.info[i].vehicle_properity->license_char);
        counter += 1;
      } else {
        printf("Vec[%zu] license plate not found.\n", i);
      }
    }

    printf("counter = %d\n", counter);
    fprintf(outFile, "%u,%u\n", n, vehicle_obj.size);
    for (size_t i = 0; i < vehicle_obj.size; i++) {
      if (vehicle_obj.info[i].vehicle_properity) {
        cvtdl_4_pts_t *license_pts = &vehicle_obj.info[i].vehicle_properity->license_pts;
        const char *license_char = vehicle_obj.info[i].vehicle_properity->license_char;
        fprintf(outFile, "%s,%f,%f,%f,%f,%s,%f,%f,%f,%f,%f,%f,%f,%f\n", vehicle_obj.info[i].name,
                vehicle_obj.info[i].bbox.x1, vehicle_obj.info[i].bbox.y1,
                vehicle_obj.info[i].bbox.x2, vehicle_obj.info[i].bbox.y2, license_char,
                license_pts->x[0], license_pts->y[0], license_pts->x[1], license_pts->y[1],
                license_pts->x[2], license_pts->y[2], license_pts->x[3], license_pts->y[3]);
      } else {
        fprintf(outFile, "%s,%f,%f,%f,%f,NULL,,,,,,,,\n", vehicle_obj.info[i].name,
                vehicle_obj.info[i].bbox.x1, vehicle_obj.info[i].bbox.y1,
                vehicle_obj.info[i].bbox.x2, vehicle_obj.info[i].bbox.y2);
      }
    }

    free(filename);
    CVI_TDL_Free(&vehicle_obj);
    CVI_TDL_ReleaseImage(&frame);
  }

  fclose(outFile);

  CVI_TDL_DestroyHandle(tdl_handle);
  CVI_TDL_Eval_DestroyHandle(eval_handle);
  CVI_SYS_Exit();
}