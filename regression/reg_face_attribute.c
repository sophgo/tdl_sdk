#include <assert.h>
#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cvimath/cvimath.h>

#include "cviai.h"
#include "sample_comm.h"
#include "core/utils/vpss_helper.h"

#define FEATURE_LENGTH  512
#define NAME_LENGTH     1024
#define DB_FEATURE_DIR  "/mnt/data/db_feature/"
#define IN_FEATURE_DIR  "/mnt/data/in_db_feature/"
#define NOT_FEATURE_DIR  "/mnt/data/not_db_feature/"

cviai_handle_t facelib_handle = NULL;

static VPSS_GRP VpssGrp = 0;
static VPSS_CHN VpssChn = VPSS_CHN0;
static CVI_S32 vpssgrp_width = 1920;
static CVI_S32 vpssgrp_height = 1080;

int genFeatureFile(const char *img_name_list, const char *feature_dir, bool face_quality) {
  FILE *fp;
  if((fp = fopen(img_name_list, "r")) == NULL) {
    printf("file open error: %s!\n", img_name_list);
    return CVI_FAILURE;
  }

  char line[1024];
  while(fscanf(fp, "%[^\n]", line)!=EOF) {
    fgetc(fp);

    printf("%s\n", line);
    VB_BLK blk_fr;
    VIDEO_FRAME_INFO_S bgr_frame;
    CVI_S32 ret = CVI_AI_ReadImage(line, &blk_fr, &bgr_frame, PIXEL_FORMAT_BGR_888);
    if (ret != CVI_SUCCESS) {
      printf("Read image failed with %#x!\n", ret);
      return ret;
    }

    int face_count = 0;
    cvai_face_t face;
    memset(&face, 0, sizeof(cvai_face_t));
    CVI_AI_RetinaFace(facelib_handle, &bgr_frame, &face, &face_count);
    if (face_count > 0 && face_quality == true) {
      VIDEO_FRAME_INFO_S rgb_frame;
      VPSS_GRP_ATTR_S vpss_grp_attr;
      VPSS_GRP_DEFAULT_HELPER(&vpss_grp_attr, bgr_frame.stVFrame.u32Width, bgr_frame.stVFrame.u32Height,
                              bgr_frame.stVFrame.enPixelFormat);
      CVI_VPSS_SetGrpAttr(VpssGrp, &vpss_grp_attr);
      CVI_VPSS_SendFrame(VpssGrp, &bgr_frame, -1);
      CVI_VPSS_GetChnFrame(VpssGrp, VpssChn, &rgb_frame, 1000);

      CVI_AI_FaceQuality(facelib_handle, &rgb_frame, &face);

      CVI_VPSS_ReleaseChnFrame(VpssGrp, VpssChn, &rgb_frame);
    }

    int face_idx = 0;
    float max_area = 0;
    for (int i = 0; i < face.size; i++) {
      cvai_bbox_t bbox = face.info[i].bbox;
      float curr_area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1);
      if (curr_area > max_area) {
        max_area = curr_area;
        face_idx = i;
      }
    }

    if (face_count > 0 && (face_quality == false || face.info[face_idx].face_quality.quality > 0.05)) {
      CVI_AI_FaceAttribute(facelib_handle, &bgr_frame, &face);

      char *file_name;
      file_name = strrchr(line, '/');
      file_name++;

      char base_name[500] = "\0";
      strcat(base_name, feature_dir);
      strcat(base_name, file_name);
      strcat(base_name, ".txt");

      FILE *fp_feature;
      if((fp_feature = fopen(base_name, "w+")) == NULL) {
        printf("Write file open error!");
        return CVI_FAILURE;
      }
      for (int i = 0; i < face.info[face_idx].face_feature.size; i++) {
        fprintf(fp_feature, "%d\n", (int)face.info[face_idx].face_feature.ptr[i]);
      }
      fclose(fp_feature);
    }

    CVI_AI_Free(&face);
    CVI_VB_ReleaseBlock(blk_fr);
  }
  fclose(fp);

  return CVI_SUCCESS;
}

static int loadCount(const char *dir_path)
{
  DIR * dirp;
  struct dirent * entry;
  dirp = opendir(dir_path);

  int count = 0;
  while ((entry = readdir(dirp)) != NULL) {
    if (entry->d_type != 8 && entry->d_type != 0) continue;
    count++;
  }
  closedir(dirp);

  return count;
}

static char** loadName(const char *dir_path, int count)
{
  DIR * dirp;
  struct dirent * entry;
  dirp = opendir(dir_path);

  char **name = calloc(count, sizeof(char *));
  for (int i = 0; i < count; i++) {
    name[i] = (char *)calloc(NAME_LENGTH, sizeof(char));
  }

  int i = 0;
  while ((entry = readdir(dirp)) != NULL) {
    if (entry->d_type != 8 && entry->d_type != 0) continue;

    strncpy(name[i], entry->d_name, strlen(entry->d_name));
    i++;
  }
  closedir(dirp);

  return name;
}

static int8_t* loadFeature(const char *dir_path, int count)
{
  DIR * dirp;
  struct dirent * entry;
  dirp = opendir(dir_path);

  int8_t *feature = calloc(count * FEATURE_LENGTH, sizeof(int8_t));
  int i = 0;
  while ((entry = readdir(dirp)) != NULL) {
    if (entry->d_type != 8 && entry->d_type != 0) continue;

    char base_name[500] = "\0";
    strcat(base_name, dir_path);
    strcat(base_name, entry->d_name);

    FILE *fp_db;
    if((fp_db = fopen(base_name, "r")) == NULL) {
      printf("file open error %s!\n", base_name);
      continue;
    }

    int line = 0;
    int idx = 0;
    while(fscanf(fp_db, "%d\n", &line) != EOF) {
      feature[i*FEATURE_LENGTH + idx] = line;
      idx++;
    }

    fclose(fp_db);
    i++;
  }

  closedir(dirp);

  return feature;
}

static int evaluateResult(int8_t *db_feature, int8_t *in_db_feature, int8_t *not_db_feature,
                          char **db_name, char **in_name,
                          int db_count, int in_count, int not_count)
{
  float *db_f = calloc(db_count * FEATURE_LENGTH, sizeof(float));
  cvm_gen_db_i8_unit_length(db_feature, db_f, FEATURE_LENGTH, db_count);

  float threshold = 0.41;
  int frr = 0;
  int far = 0;
  for (int i = 0; i < in_count; i++) {
    unsigned int *k_index = calloc(db_count, sizeof(unsigned int));
    float *k_value = calloc(db_count, sizeof(float));
    float *buffer = calloc(db_count * FEATURE_LENGTH, sizeof(float));
    cvm_cpu_i8data_ip_match(&in_db_feature[i * FEATURE_LENGTH], db_feature, db_f, k_index, k_value, buffer,
                            FEATURE_LENGTH, db_count, 1);
    if (k_value[0] < threshold ||
        strcmp(in_name[i], db_name[k_index[0]]) != 0) frr++;

    free(k_index);
    free(k_value);
    free(buffer);
  }
  for (int i = 0; i < not_count; i++) {
    unsigned int *k_index = calloc(db_count, sizeof(unsigned int));
    float *k_value = calloc(db_count, sizeof(float));
    float *buffer = calloc(db_count * FEATURE_LENGTH, sizeof(float));
    cvm_cpu_i8data_ip_match(&not_db_feature[i * FEATURE_LENGTH], db_feature, db_f, k_index, k_value, buffer,
                            FEATURE_LENGTH, db_count, 1);
    if (k_value[0] > threshold) far++;
    free(k_index);
    free(k_value);
    free(buffer);
  }

  free(db_f);

  printf("frr: %d\n", frr);
  printf("far: %d\n", far);

  return 0;
}

int main(void) {
  CVI_S32 ret = CVI_SUCCESS;

  ret = MMF_INIT_HELPER(vpssgrp_width, vpssgrp_height, SAMPLE_PIXEL_FORMAT, vpssgrp_width,
                        vpssgrp_height, SAMPLE_PIXEL_FORMAT);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  VPSS_INIT_HELPER(VpssGrp, vpssgrp_width, vpssgrp_height, 0, PIXEL_FORMAT_BGR_888, 608, 608,
                   PIXEL_FORMAT_RGB_888, VPSS_MODE_SINGLE, true, false);

  ret = CVI_AI_CreateHandle(&facelib_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE,
                            "/mnt/data/retina_face.cvimodel");
  ret = CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE,
                            "/mnt/data/bmface.cvimodel");
  ret = CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_FACEQUALITY,
                            "/mnt/data/fqnet-v5_shufflenetv2-softmax.cvimodel");
  if (ret != CVI_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }

  CVI_AI_SetSkipVpssPreprocess(facelib_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, false);

  genFeatureFile("/mnt/data/db_name.txt", DB_FEATURE_DIR, true);
  genFeatureFile("/mnt/data/in_db_name.txt", IN_FEATURE_DIR, false);
  genFeatureFile("/mnt/data/not_db_name.txt", NOT_FEATURE_DIR, false);

  CVI_AI_DestroyHandle(facelib_handle);
  CVI_VPSS_StopGrp(VpssGrp);
  CVI_VPSS_DestroyGrp(VpssGrp);
  CVI_SYS_Exit();

  int db_count = loadCount(DB_FEATURE_DIR);
  int in_count = loadCount(IN_FEATURE_DIR);
  int not_count = loadCount(NOT_FEATURE_DIR);
  printf("db count: %d, in count: %d, not count: %d\n", db_count, in_count, not_count);

  char **db_name = loadName(DB_FEATURE_DIR, db_count);
  char **in_name = loadName(IN_FEATURE_DIR, in_count);
  char **not_name = loadName(NOT_FEATURE_DIR, not_count);
  int8_t *db_feature = loadFeature(DB_FEATURE_DIR, db_count);
  int8_t *in_feature = loadFeature(IN_FEATURE_DIR, in_count);
  int8_t *not_feature = loadFeature(NOT_FEATURE_DIR, not_count);

  evaluateResult(db_feature, in_feature, not_feature, db_name, in_name, db_count, in_count, not_count);

  free(db_feature);
  free(in_feature);
  free(not_feature);
  for (int i = 0; i < db_count; i++) {
    free(db_name[i]);
  }
  free(db_name);
  for (int i = 0; i < in_count; i++) {
    free(in_name[i]);
  }
  free(in_name);
  for (int i = 0; i < not_count; i++) {
    free(not_name[i]);
  }
  free(not_name);
}
