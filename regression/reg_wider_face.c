#include <dirent.h>
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <sys/stat.h> 

#include <cvimath/cvimath.h>

#include "cviai.h"
#include "core/utils/vpss_helper.h"

cviai_handle_t facelib_handle = NULL;

static CVI_S32 vpssgrp_width = 2048;
static CVI_S32 vpssgrp_height = 1536;

static void genPath(const char *base, const char *tail, char *result)
{
  strcpy(result, base);
  strcat(result, "/");
  strcat(result, tail);
}

// TODO:: Copy to here first
static cvai_face_info_t bbox_rescale(float width, float height, cvai_face_t *face_meta, int face_idx) {
  cvai_bbox_t bbox = face_meta->info[face_idx].bbox;
  cvai_face_info_t face_info;
  float x1, x2, y1, y2;

  memset(&face_info, 0, sizeof(cvai_face_info_t));
  face_info.face_pts.size = face_meta->info[face_idx].face_pts.size;
  face_info.face_pts.x = (float *)malloc(sizeof(float) * face_meta->info[face_idx].face_pts.size);
  face_info.face_pts.y = (float *)malloc(sizeof(float) * face_meta->info[face_idx].face_pts.size);

  if (width >= height) {
    float ratio_x, ratio_y, bbox_y_height, bbox_padding_top;
    ratio_x = width / face_meta->width;
    bbox_y_height = face_meta->height * height / width;
    ratio_y = height / bbox_y_height;
    bbox_padding_top = (face_meta->height - bbox_y_height) / 2;
    x1 = bbox.x1 * ratio_x;
    x2 = bbox.x2 * ratio_x;
    y1 = (bbox.y1 - bbox_padding_top) * ratio_y;
    y2 = (bbox.y2 - bbox_padding_top) * ratio_y;

    for (int j = 0; j < 5; ++j) {
      face_info.face_pts.x[j] = face_meta->info[face_idx].face_pts.x[j] * ratio_x;
      face_info.face_pts.y[j] =
          (face_meta->info[face_idx].face_pts.y[j] - bbox_padding_top) * ratio_y;
    }
  } else {
    float ratio_x, ratio_y, bbox_x_height, bbox_padding_left;
    ratio_y = height / face_meta->height;
    bbox_x_height = face_meta->width * width / height;
    ratio_x = width / bbox_x_height;
    bbox_padding_left = (face_meta->width - bbox_x_height) / 2;
    x1 = (bbox.x1 - bbox_padding_left) * ratio_x;
    x2 = (bbox.x2 - bbox_padding_left) * ratio_x;
    y1 = bbox.y1 * ratio_y;
    y2 = bbox.y2 * ratio_y;

    for (int j = 0; j < 5; ++j) {
      face_info.face_pts.x[j] =
          (face_meta->info[face_idx].face_pts.x[j] - bbox_padding_left) * ratio_x;
      face_info.face_pts.y[j] = face_meta->info[face_idx].face_pts.y[j] * ratio_y;
    }
  }

  face_info.bbox.x1 = fmax(fmin(x1, width - 1), (float)0);
  face_info.bbox.x2 = fmax(fmin(x2, width - 1), (float)0);
  face_info.bbox.y1 = fmax(fmin(y1, height - 1), (float)0);
  face_info.bbox.y2 = fmax(fmin(y2, height - 1), (float)0);

  return face_info;
}

static int genResult(char *dataset_dir, char *result_dir)
{
  DIR* dirp;
  struct dirent* entry;

  char dataset_path[1000];
  genPath(dataset_dir, "images", dataset_path);

  dirp = opendir(dataset_path);
  while ((entry = readdir(dirp)) != NULL) {
    if (entry->d_type != 4 || 0 == strcmp(entry->d_name, ".") || 0 == strcmp(entry->d_name, "..")) continue;

    DIR* group_dir;
    struct dirent* group_entry;
    char group_path[1000];
    char result_group_path[1000];

    genPath(dataset_path, entry->d_name, group_path);
    genPath(result_dir, entry->d_name, result_group_path);

    group_dir = opendir(group_path);
    if (0 != mkdir(result_group_path, S_IRWXO)) {
      if (errno != EEXIST) {
        printf("Create folder failed. %s\n", result_group_path);
        return -1;
      }
    }

    printf("%s, %s\n", result_group_path, result_dir);

    while ((group_entry = readdir(group_dir)) != NULL) {
      if (group_entry->d_type != 8) continue;

      char image_path[1000];
      genPath(group_path, group_entry->d_name, image_path);

      char result_path[1000];
      strcpy(result_path, result_group_path);
      strcat(result_path, "/");
      strncat(result_path, group_entry->d_name, strlen(group_entry->d_name)-4);
      strcat(result_path, ".txt");
      remove(result_path);

      VB_BLK blk;
      VIDEO_FRAME_INFO_S frame;
      int face_count = 0;
      cvai_face_t face;
      memset(&face, 0, sizeof(cvai_face_t));

      CVI_S32 ret = CVI_AI_ReadImage(image_path, &blk, &frame, PIXEL_FORMAT_RGB_888);
      if (ret != CVI_SUCCESS) {
        printf("Read image failed. %s!\n", image_path);
      } else {
        CVI_AI_RetinaFace(facelib_handle, &frame, &face, &face_count);
        CVI_VB_ReleaseBlock(blk);
      }

      printf("%s, %d\n", image_path, face_count);

      FILE *fp;
      if((fp = fopen(result_path, "w+")) == NULL) {
        printf("Write file open error. %s!\n", result_path);
        return CVI_FAILURE;
      }

      fprintf(fp, "%s\n", group_entry->d_name);
      fprintf(fp, "%d\n", face.size);
      for (int i = 0; i < face.size; i++) {
        cvai_face_info_t face_info =
          bbox_rescale(frame.stVFrame.u32Width, frame.stVFrame.u32Height, &face, i);
        fprintf(fp, "%f %f %f %f %f\n", face_info.bbox.x1, face_info.bbox.y1,
                face_info.bbox.x2 - face_info.bbox.x1,
                face_info.bbox.y2 - face_info.bbox.y1, face.info[i].bbox.score);
      }
      fclose(fp);

      CVI_AI_Free(&face);
    }
    closedir(group_dir);
  }
  closedir(dirp);

  return CVI_SUCCESS;
}

int main(int argc, char *argv[])
{
  if (argc != 3) {
    printf("Usage: reg_wider_face <dataset dir path> <result dir path>.\n");
    printf("dataset dir path: Wider face validation folder. eg. /mnt/data/WIDER_val\n");
    printf("result dir path: Result directory path. eg. /mnt/data/wider_result\n");
    printf("Using wider face matlab code to evaluate AUC!!\n");
    return CVI_FAILURE;
  }

  CVI_S32 ret = CVI_SUCCESS;

  ret = MMF_INIT_HELPER(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, vpssgrp_width,
                        vpssgrp_height, PIXEL_FORMAT_RGB_888);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_CreateHandle(&facelib_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE,
                            "/mnt/data/retina_face.cvimodel");
  if (ret != CVI_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }

  CVI_AI_SetSkipVpssPreprocess(facelib_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, false);

  genResult(argv[1], argv[2]);

  CVI_AI_DestroyHandle(facelib_handle);
  CVI_SYS_Exit();
}
