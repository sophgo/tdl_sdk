#define _GNU_SOURCE
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "ive/ive.h"

#define WRITE_RESULT_TO_FILE 0

enum LicenseFormat { taiwan, china };

int main(int argc, char *argv[]) {
  if (argc != 8) {
    printf(
        "Usage: %s <vehicle_detection_model_path>\n"
        "          <use_mobiledet_vehicle (0/1)>\n"
        "          <license_plate_detection_model_path>\n"
        "          <license_plate_recognition_model_path>\n"
        "          <license_format (tw/cn)>\n"
        "          <sample_imagelist_path>\n"
        "          <inference_count>\n",
        argv[0]);
    return CVIAI_FAILURE;
  }
  CVI_S32 ret = CVI_SUCCESS;

  // Init VB pool size.
  const CVI_S32 vpssgrp_width = 1920;
  const CVI_S32 vpssgrp_height = 1080;
  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 5, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888_PLANAR, 5);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }
  cviai_handle_t ai_handle = NULL;
  ret = CVI_AI_CreateHandle2(&ai_handle, 1, 0);
  int use_vehicle = atoi(argv[2]);
  if (use_vehicle == 1) {
    printf("set:CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE_D0\n");
    ret |= CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE_D0, argv[1]);
    CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE_D0, false);
  } else if (use_vehicle == 0) {
    printf("set:CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D0\n");
    ret |= CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D0, argv[1]);
    ret |= CVI_AI_SelectDetectClass(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D0, 1,
                                    CVI_AI_DET_GROUP_VEHICLE);
    CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D0, false);
  } else {
    printf("Unknown det model type.\n");
    return CVIAI_FAILURE;
  }
  enum LicenseFormat license_format;
  if (strcmp(argv[5], "tw") == 0) {
    license_format = taiwan;
  } else if (strcmp(argv[5], "cn") == 0) {
    license_format = china;
  } else {
    printf("Unknown license type %s\n", argv[5]);
    return CVIAI_FAILURE;
  }

  ret |= CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_WPODNET, argv[3]);
  CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_WPODNET, false);
  switch (license_format) {
    case taiwan:
      ret |= CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_LPRNET_TW, argv[4]);
      CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_LPRNET_TW, false);
      break;
    case china:
      ret |= CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_LPRNET_CN, argv[4]);
      CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_LPRNET_CN, false);
      break;
    default:
      return CVIAI_FAILURE;
  }

  if (ret != CVIAI_SUCCESS) {
    printf("open failed with %#x!\n", ret);
    return ret;
  }

#if WRITE_RESULT_TO_FILE
  FILE *outFile;
  char outFile_name[128];
  outFile_name[0] = '\0';
  strcat(outFile_name, "result_sample_lpr_");
  strcat(outFile_name, argv[2]);
  strcat(outFile_name, ".txt");
  outFile = fopen(outFile_name, "w");
  if (outFile == NULL) {
    printf("There is a problem opening the output file.\n");
    exit(EXIT_FAILURE);
  }
#endif

  char *imagelist_path = argv[6];
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

#if WRITE_RESULT_TO_FILE
  fprintf(outFile, "%u\n", imageNum);
#endif

  int inference_count = atoi(argv[7]);

  cvai_object_t vehicle_obj, license_plate_obj;
  memset(&vehicle_obj, 0, sizeof(cvai_object_t));
  memset(&license_plate_obj, 0, sizeof(cvai_object_t));
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

    IVE_HANDLE ive_handle = CVI_IVE_CreateHandle();
    // Read image using IVE.
    IVE_IMAGE_S ive_frame = CVI_IVE_ReadImage(ive_handle, image_path, IVE_IMAGE_TYPE_U8C3_PACKAGE);
    // CVI_IVE_ReadImage(ive_handle, image_path, IVE_IMAGE_TYPE_U8C3_PLANAR);
    if (ive_frame.u16Width == 0) {
      printf("Read image failed with %x!\n", ret);
      return ret;
    }
    // Convert to VIDEO_FRAME_INFO_S. IVE_IMAGE_S must be kept to release when not used.
    VIDEO_FRAME_INFO_S frame;
    ret = CVI_IVE_Image2VideoFrameInfo(&ive_frame, &frame, false);
    if (ret != CVI_SUCCESS) {
      printf("Convert to video frame failed with %#x!\n", ret);
      return ret;
    }

    printf("Vehicle Detection ... start\n");
    if (use_vehicle == 1) {
      CVI_AI_MobileDetV2_Vehicle_D0(ai_handle, &frame, &vehicle_obj);
    } else {
      CVI_AI_MobileDetV2_D0(ai_handle, &frame, &vehicle_obj);
    }
    printf("Find %u vehicles.\n", vehicle_obj.size);

    /* LP Detection */
    printf("CVI_AI_LicensePlateDetection ... start\n");
    CVI_AI_LicensePlateDetection(ai_handle, &frame, &vehicle_obj);

    /* LP Recognition */
    printf("CVI_AI_LicensePlateRecognition ... start\n");
    switch (license_format) {
      case taiwan:
        CVI_AI_LicensePlateRecognition_TW(ai_handle, &frame, &vehicle_obj);
        break;
      case china:
        CVI_AI_LicensePlateRecognition_CN(ai_handle, &frame, &vehicle_obj);
        break;
      default:
        return CVIAI_FAILURE;
    }

#if WRITE_RESULT_TO_FILE
    int counter = 0;
#endif
    for (size_t i = 0; i < vehicle_obj.size; i++) {
      if (vehicle_obj.info[i].vehicle_properity) {
        printf("Vec[%zu] ID number: %s\n", i, vehicle_obj.info[i].vehicle_properity->license_char);
#if WRITE_RESULT_TO_FILE
        counter += 1;
#endif
      } else {
        printf("Vec[%zu] license plate not found.\n", i);
      }
    }

#if WRITE_RESULT_TO_FILE
    fprintf(outFile, "%s\n", image_path);
    fprintf(outFile, "%u,%d\n", vehicle_obj.size, counter);
    for (size_t i = 0; i < vehicle_obj.size; i++) {
      if (vehicle_obj.info[i].vehicle_properity) {
        cvai_4_pts_t *license_pts = &vehicle_obj.info[i].vehicle_properity->license_pts;
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
#endif

    CVI_SYS_FreeI(ive_handle, &ive_frame);
    CVI_IVE_DestroyHandle(ive_handle);
    CVI_AI_Free(&vehicle_obj);
  }
#if WRITE_RESULT_TO_FILE
  fclose(outFile);
#endif

  CVI_AI_DestroyHandle(ai_handle);
  CVI_SYS_Exit();
}