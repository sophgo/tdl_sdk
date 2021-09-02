#define _GNU_SOURCE
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "ive/ive.h"

#define WRITE_RESULT_TO_FILE 0

int main(int argc, char **argv) {
  if (argc != 3) {
    printf(
        "Usage: %s <sample_imagelist_path>\n"
        "          <inference_count>\n",
        argv[0]);
    return CVIAI_FAILURE;
  }
  CVI_S32 ret = CVIAI_SUCCESS;

  // Init VB pool size.
  const CVI_S32 vpssgrp_width = 1920;
  const CVI_S32 vpssgrp_height = 1080;
  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 5, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888_PLANAR, 5);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }
  IVE_HANDLE handle = CVI_IVE_CreateHandle();

  cviai_handle_t ai_handle = NULL;
  ret = CVI_AI_CreateHandle2(&ai_handle, 1, 0);
  if (ret != CVIAI_SUCCESS) {
    printf("failed with %#x!\n", ret);
    return ret;
  }

#if WRITE_RESULT_TO_FILE
  FILE *outFile;
  outFile = fopen("result_sample_td.txt", "w");
  if (outFile == NULL) {
    printf("There is a problem opening the output file.\n");
    exit(EXIT_FAILURE);
  }
#endif

  char *imagelist_path = argv[1];
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

  int inference_count = atoi(argv[2]);

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
    // printf("\n[%i] image path = %s\n", counter, image_path);

    // Read image using IVE.
    IVE_IMAGE_S ive_frame = CVI_IVE_ReadImage(handle, image_path, IVE_IMAGE_TYPE_U8C3_PLANAR);
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

    float moving_score;
    CVI_AI_TamperDetection(ai_handle, &frame, &moving_score);
    printf("[%d] %f\n", counter, moving_score);

#if WRITE_RESULT_TO_FILE
    fprintf(outFile, "%f\n", moving_score);
#endif

    CVI_SYS_FreeI(handle, &ive_frame);
  }

#if WRITE_RESULT_TO_FILE
  fclose(outFile);
#endif

  CVI_IVE_DestroyHandle(handle);
  CVI_AI_DestroyHandle(ai_handle);
  CVI_SYS_Exit();
}