#define _GNU_SOURCE
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "sample_comm.h"
#include "vi_vo_utils.h"

#include <cvi_sys.h>
#include <cvi_vb.h>
#include <cvi_vi.h>
#include "cviai_perfetto.h"

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
  if (argc != 6) {
    printf(
        "Usage: %s <vehicle_detection_model_path>\n"
        "          <license_plate_detection_model_path>\n"
        "          <license_plate_recognition_model_path>\n"
        "          <sample_imagelist_path>\n"
        "          <inference_count>\n",
        argv[0]);
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

  char *imagelist_path = argv[4];
  FILE *inFile;
  char *line = NULL;
  size_t len = 0;
  ssize_t read;
  inFile = fopen(imagelist_path, "r");
  if (inFile == NULL) {
    printf("There is a problem opening the rcfile: %s\n", imagelist_path);
    exit(EXIT_FAILURE);
  }
  if ((read = getline(&line, &len, inFile)) == -1) {
    printf("get line error\n");
    exit(EXIT_FAILURE);
  }
  *strchrnul(line, '\n') = '\0';
  int imageNum = atoi(line);

  int inference_count = atoi(argv[5]);

  for (int counter = 0; counter < imageNum; counter++) {
    if (counter == inference_count) {
      break;
    }
    if ((read = getline(&line, &len, inFile)) == -1) {
      printf("get line error\n");
      exit(EXIT_FAILURE);
    }
    *strchrnul(line, '\n') = '\0';
    char *image_path = line;
    printf("[%i] image path = %s\n", counter, image_path);

    VB_BLK blk_fr;
    VIDEO_FRAME_INFO_S frame;
    CVI_S32 ret = CVI_AI_ReadImage(image_path, &blk_fr, &frame, PIXEL_FORMAT_RGB_888);
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

    for (size_t i = 0; i < license_plate_obj.size; i++) {
      if (license_plate_obj.info[i].bpts.size > 0) {
        printf("Vec[%lu] ID number: %s\n", i, license_plate_obj.info[i].name);
      } else {
        printf("Vec[%lu] license plate not found.\n", i);
      }
    }
  }

  CVI_AI_DestroyHandle(ai_handle);
}