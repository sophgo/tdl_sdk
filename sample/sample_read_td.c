#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cviai.h"
#include "ive/ive.h"

#define UPDATE_INTERVAL 10

char *cstrconcat(const char *s1, const char *s2);

int main(int argc, char **argv) {
  CVI_S32 ret = CVI_SUCCESS;
  char *imagefile_list_name = "/imagefile_list.txt";
  char *imagefile_path;
  if (argc == 1) {
    imagefile_path = "./images";
  } else if (argc == 2) {
    imagefile_path = argv[1];
  } else {
    printf("Usage: %s <images_path>\n", argv[0]);
    return -1;
  }
  printf("start loading images\n");
  char *imagefile_list_path = cstrconcat(imagefile_path, imagefile_list_name);
  printf("imagefile_list_path: %s\n", imagefile_list_path);

  FILE *inFile;
  char *line = NULL;
  size_t len = 0;
  ssize_t read;

  inFile = fopen(imagefile_list_path, "r");
  if (inFile == NULL) {
    printf("There is a problem opening the image file list:\n");
    printf("    %s/imagefile_list.txt\n", imagefile_list_path);
    exit(EXIT_FAILURE);
  }
  if ((read = getline(&line, &len, inFile)) == -1) {
    printf("get line error\n");
    exit(EXIT_FAILURE);
  }
  int image_num = atoi(line);
  //   printf("image_num: %d\n", image_num);

  char *image_name = NULL;
  char *image_path;
  float moving_score;
  printf("start simulation...\n");

  if ((read = getline(&image_name, &len, inFile)) == -1) {
    printf("get line error\n");
    exit(EXIT_FAILURE);
  }
  *strchrnul(image_name, '\n') = '\0';
  image_path = cstrconcat(imagefile_path, image_name);

  IVE_HANDLE handle = CVI_IVE_CreateHandle();
  // Read image using IVE.
  IVE_IMAGE_S frame = CVI_IVE_ReadImage(handle, image_path, IVE_IMAGE_TYPE_U8C3_PLANAR);
  if (frame.u16Width == 0) {
    printf("Read image failed with %x!\n", ret);
    return ret;  // ?
  }
  // Convert to VIDEO_FRAME_INFO_S. IVE_IMAGE_S must be kept to release when not used.
  VIDEO_FRAME_INFO_S cameraFrame;
  ret = CVI_IVE_Image2VideoFrameInfo(&frame, &cameraFrame, false);
  if (ret != CVI_SUCCESS) {
    printf("Convert to video frame failed with %#x!\n", ret);
    return ret;
  }

  // Init cviai handle.
  cviai_handle_t ai_handle = NULL;
  ret = CVI_AI_CreateHandle(&ai_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create ai handle failed with %#x!\n", ret);
    return ret;
  }

  CVI_AI_TamperDetection(ai_handle, &cameraFrame, &moving_score);

  CVI_SYS_FreeI(handle, &frame);

  for (int img_counter = 1; img_counter < image_num; img_counter++) {
    image_name = NULL;
    if ((read = getline(&image_name, &len, inFile)) == -1) {
      printf("get line error\n");
      exit(EXIT_FAILURE);
    }
    *strchrnul(image_name, '\n') = '\0';
    image_path = cstrconcat(imagefile_path, image_name);

    frame = CVI_IVE_ReadImage(handle, image_path, IVE_IMAGE_TYPE_U8C3_PLANAR);
    ret = CVI_IVE_Image2VideoFrameInfo(&frame, &cameraFrame, false);
    if (ret != CVI_SUCCESS) {
      printf("Convert to video frame failed with %#x!\n", ret);
      return ret;
    }

    CVI_AI_TamperDetection(ai_handle, &cameraFrame, &moving_score);

    printf("[%d] moving: %f\n", img_counter, moving_score);

    CVI_SYS_FreeI(handle, &frame);
  }

  CVI_IVE_DestroyHandle(handle);
  CVI_AI_DestroyHandle(ai_handle);

  fclose(inFile);

  printf("done\n");

  return 0;
}

char *cstrconcat(const char *s1, const char *s2) {
  char *result = malloc(strlen(s1) + strlen(s2) + 1);  // +1 for the null-terminator
  // in real code you would check for errors in malloc here
  strcpy(result, s1);
  strcat(result, s2);
  return result;
}