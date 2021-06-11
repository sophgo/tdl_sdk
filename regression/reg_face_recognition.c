#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cvimath/cvimath.h>
#include "core/core/cvai_core_types.h"
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "cviai_perfetto.h"
#include "evaluation/cviai_evaluation.h"
#include "evaluation/cviai_media.h"

#define FEATURE_LENGTH 512
#define NAME_LENGTH 1024
#define DB_IMAGE_DIR "/db/"
#define IN_IMAGE_DIR "/in_db/"
#define NOT_IMAGE_DIR "/not_db/"
#define DB_FEATURE_DIR "/db_feature/"
#define IN_FEATURE_DIR "/in_db_feature/"
#define NOT_FEATURE_DIR "/not_db_feature/"

cviai_handle_t facelib_handle = NULL;
cviai_service_handle_t service_handle = NULL;

static VPSS_GRP VpssGrp = 0;
static CVI_S32 vpssgrp_width = 1920;
static CVI_S32 vpssgrp_height = 1080;

typedef FILE *(*create_logger_file_fn)(const char *, const char *);
typedef void (*print_pass_fn)(FILE *, const char *, int, float, float);
typedef void (*print_fail_fn)(FILE *, const char *, int, float);
typedef void (*close_logger_file_fn)(FILE *);

static void log_reg_pass(FILE *fp, const char *id, int face_count, float face_quality) {
  if (fp) {
    fprintf(fp, "baseid=%-100s facecount=%d quality_score=%f\n", id, face_count, face_quality);
  }
}

static void log_reg_fail(FILE *fp, const char *id, int face_count, float face_quality,
                         float quality_threshold) {
  if (fp) {
    fprintf(fp, "baseid=%-100s facecount=%d no_face=%-4s quality_score=%f low_quality=%-4s \n", id,
            face_count, face_count <= 0 ? "YES" : "NO", face_quality,
            face_quality < quality_threshold ? "YES" : "NO");
  }
}

static void log_comp_result(FILE *fp, const char *id, const char *comp_id, float score) {
  if (fp) {
    fprintf(fp, "baseid=%-50s verity_file=%-50s score=%f\n", id, comp_id, score);
  }
}

static void removePreviousFile(const char *dir_path) {
  DIR *dirp;
  struct dirent *entry;
  dirp = opendir(dir_path);

  while ((entry = readdir(dirp)) != NULL) {
    if (entry->d_type != 8 && entry->d_type != 0) continue;

    char base_name[500] = "\0";
    strcat(base_name, dir_path);
    strcat(base_name, entry->d_name);
    remove(base_name);
  }
  closedir(dirp);
}

int genFeatureFile(const char *img_dir, const char *feature_dir, bool do_face_quality,
                   float quality_thresh, FILE *fp_pass, FILE *fp_fail) {
  DIR *dirp;
  struct dirent *entry;
  dirp = opendir(img_dir);

  if (0 != mkdir(feature_dir, S_IRWXO) && EEXIST != errno) {
    printf("Create %s failed.\n", feature_dir);
    return CVI_FAILURE;
  }
  removePreviousFile(feature_dir);

  while ((entry = readdir(dirp)) != NULL) {
    if (entry->d_type != 8 && entry->d_type != 0) continue;
    char line[500] = "\0";
    strcat(line, img_dir);
    strcat(line, entry->d_name);

    printf("%s\n", line);
    VB_BLK blk_fr;
    VIDEO_FRAME_INFO_S rgb_frame;
    CVI_S32 ret = CVI_AI_ReadImage(line, &blk_fr, &rgb_frame, PIXEL_FORMAT_RGB_888);
    if (ret != CVI_SUCCESS) {
      printf("Read image failed with %#x!\n", ret);
      return ret;
    }

    cvai_face_t face;
    memset(&face, 0, sizeof(cvai_face_t));
    CVI_AI_RetinaFace(facelib_handle, &rgb_frame, &face);
    if (face.size > 0 && do_face_quality == true) {
      CVI_AI_FaceQuality(facelib_handle, &rgb_frame, &face);
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

    char *file_name;
    file_name = strrchr(line, '/');
    file_name++;

    if (face.size > 0 &&
        (do_face_quality == false || face.info[face_idx].face_quality > quality_thresh)) {
      CVI_AI_FaceRecognitionOne(facelib_handle, &rgb_frame, &face, face_idx);

      char base_name[500] = "\0";
      strcat(base_name, feature_dir);
      strcat(base_name, file_name);
      strcat(base_name, ".txt");

      FILE *fp_feature;
      if ((fp_feature = fopen(base_name, "w+")) == NULL) {
        printf("Write file open error!");
        return CVI_FAILURE;
      }
      for (int i = 0; i < face.info[face_idx].feature.size; i++) {
        fprintf(fp_feature, "%d\n", (int)face.info[face_idx].feature.ptr[i]);
      }

      log_reg_pass(fp_pass, file_name, face.size, face.info[face_idx].face_quality);
      fclose(fp_feature);
    } else {
      log_reg_fail(fp_fail, file_name, face.size,
                   face.size > 0 ? face.info[face_idx].face_quality : 0, quality_thresh);
    }

    CVI_AI_Free(&face);
    CVI_VB_ReleaseBlock(blk_fr);
  }
  closedir(dirp);

  return CVI_SUCCESS;
}

static int loadCount(const char *dir_path) {
  DIR *dirp;
  struct dirent *entry;
  dirp = opendir(dir_path);

  int count = 0;
  while ((entry = readdir(dirp)) != NULL) {
    if (entry->d_type != 8 && entry->d_type != 0) continue;
    count++;
  }
  closedir(dirp);

  return count;
}

static char **loadName(const char *dir_path, int count) {
  DIR *dirp;
  struct dirent *entry;
  dirp = opendir(dir_path);

  char **name = calloc(count, sizeof(char *));
  for (int i = 0; i < count; i++) {
    name[i] = (char *)calloc(NAME_LENGTH, sizeof(char));
  }

  int i = 0;
  while ((entry = readdir(dirp)) != NULL) {
    if (entry->d_type != 8 && entry->d_type != 0) continue;

    strncpy(name[i], entry->d_name, strlen(entry->d_name) - 4);
    i++;
  }
  closedir(dirp);

  return name;
}

static int8_t *loadFeature(const char *dir_path, int count) {
  DIR *dirp;
  struct dirent *entry;
  dirp = opendir(dir_path);

  int8_t *feature = calloc(count * FEATURE_LENGTH, sizeof(int8_t));
  int i = 0;
  while ((entry = readdir(dirp)) != NULL) {
    if (entry->d_type != 8 && entry->d_type != 0) continue;

    char base_name[500] = "\0";
    strcat(base_name, dir_path);
    strcat(base_name, entry->d_name);

    FILE *fp_db;
    if ((fp_db = fopen(base_name, "r")) == NULL) {
      printf("file open error %s!\n", base_name);
      continue;
    }

    int line = 0;
    int idx = 0;
    while (fscanf(fp_db, "%d\n", &line) != EOF) {
      feature[i * FEATURE_LENGTH + idx] = line;
      idx++;
    }

    fclose(fp_db);
    i++;
  }

  closedir(dirp);

  return feature;
}

static int evaluateResult(int8_t *db_feature, int8_t *in_db_feature, int8_t *not_db_feature,
                          char **db_name, char **in_name, char **not_in_name, int db_count,
                          int in_count, int not_count, float threshold, FILE *fp_far,
                          FILE *fp_frr) {
  cvai_service_feature_array_t feature_array;
  feature_array.data_num = db_count;
  feature_array.feature_length = FEATURE_LENGTH;
  feature_array.ptr = db_feature;
  feature_array.type = TYPE_INT8;

  CVI_AI_Service_RegisterFeatureArray(service_handle, feature_array, COS_SIMILARITY);

  int frr = 0;
  int far = 0;
  int topk = 1;
  for (int i = 0; i < in_count; i++) {
    uint32_t indices[topk];
    float scores[topk];
    uint32_t score_size;
    CVI_AI_Service_RawMatching(service_handle, in_db_feature + (i * FEATURE_LENGTH), TYPE_INT8,
                               topk, threshold, indices, scores, &score_size);
    if (score_size == 0 || strcmp(in_name[i], db_name[indices[0]]) != 0) frr++;

    log_comp_result(fp_frr, db_name[indices[0]], in_name[i], scores[0]);
  }
  for (int i = 0; i < not_count; i++) {
    uint32_t indices[topk];
    float scores[topk];
    uint32_t score_size;
    CVI_AI_Service_RawMatching(service_handle, not_db_feature + (i * FEATURE_LENGTH), TYPE_INT8,
                               topk, threshold, indices, scores, &score_size);

    if (score_size > 0) far++;
    log_comp_result(fp_far, db_name[indices[0]], not_in_name[i], scores[0]);
  }

  if (threshold > 0.0) {
    printf("frr: %d\n", frr);
    printf("far: %d\n", far);
  }

  return 0;
}

int main(int argc, char *argv[]) {
  if (argc < 6 || argc > 8) {
    printf(
        "Usage: %s <face detect model path> <face recognition model path> "
        "<face quality model path> <image_root_dir> <feature_root_dir> [quality_thresh] "
        "[simularity_thresh].\n",
        argv[0]);
    printf("Face detect model path: Path to face detect cvimodel.\n");
    printf("Face attribute model path: Path to face attribute cvimodel.\n");
    printf("Face quality model path: Path to face quaity cvimodel.\n");
    printf("Image root dir: Image root directory.\n");
    printf("Feature root dir: Root directory to temporarily save feature file.\n");
    printf("quality_thresh (Optional): Threshold of face quality (Default: 0.05).\n");
    printf(
        "simularity_thresh (Optional): Threshold of face matching simularity (Default: 0.41).\n");
    return CVI_FAILURE;
  }

  CVI_AI_PerfettoInit();
  CVI_S32 ret = CVI_SUCCESS;

  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 5, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888, 5);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_CreateHandle(&facelib_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create ai handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_Service_CreateHandle(&service_handle, &facelib_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create service handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, argv[1]);
  ret |= CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_FACERECOGNITION, argv[2]);
  ret |= CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_FACEQUALITY, argv[3]);
  if (ret != CVI_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }

  CVI_AI_SetSkipVpssPreprocess(facelib_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, false);

  char db_dir_full[500] = "\0";
  char in_dir_full[500] = "\0";
  char not_dir_full[500] = "\0";
  strcat(db_dir_full, argv[4]);
  strcat(db_dir_full, DB_IMAGE_DIR);
  strcat(in_dir_full, argv[4]);
  strcat(in_dir_full, IN_IMAGE_DIR);
  strcat(not_dir_full, argv[4]);
  strcat(not_dir_full, NOT_IMAGE_DIR);

  char db_feature_full[500] = "\0";
  char in_feature_full[500] = "\0";
  char not_feature_full[500] = "\0";
  strcat(db_feature_full, argv[5]);
  strcat(db_feature_full, DB_FEATURE_DIR);
  strcat(in_feature_full, argv[5]);
  strcat(in_feature_full, IN_FEATURE_DIR);
  strcat(not_feature_full, argv[5]);
  strcat(not_feature_full, NOT_FEATURE_DIR);

  if (0 != mkdir(argv[5], S_IRWXO) && EEXIST != errno) {
    printf("Create %s failed.\n", argv[5]);
    return CVI_FAILURE;
  }

  float quality_thresh = argc >= 7 ? atof(argv[6]) : 0.05;
  float sim_thresh = argc >= 8 ? atof(argv[7]) : 0.41;
  printf("face quality threshold: %f, face match threshold: %f\n", quality_thresh, sim_thresh);

  FILE *fp_reg_pass = fopen("reg_pass.txt", "w+");
  FILE *fp_reg_fail = fopen("reg_fail.txt", "w+");
  genFeatureFile(db_dir_full, db_feature_full, true, quality_thresh, fp_reg_pass, fp_reg_fail);
  fclose(fp_reg_pass);
  fclose(fp_reg_fail);

  genFeatureFile(in_dir_full, in_feature_full, false, quality_thresh, NULL, NULL);
  genFeatureFile(not_dir_full, not_feature_full, false, quality_thresh, NULL, NULL);

  int db_count = loadCount(db_feature_full);
  int in_count = loadCount(in_feature_full);
  int not_count = loadCount(not_feature_full);
  printf("db count: %d, in count: %d, not count: %d\n", db_count, in_count, not_count);

  char **db_name = loadName(db_feature_full, db_count);
  char **in_name = loadName(in_feature_full, in_count);
  char **not_name = loadName(not_feature_full, not_count);
  int8_t *db_feature = loadFeature(db_feature_full, db_count);
  int8_t *in_feature = loadFeature(in_feature_full, in_count);
  int8_t *not_feature = loadFeature(not_feature_full, not_count);

  FILE *fp_far = fopen("far_results.txt", "w+");
  FILE *fp_frr = fopen("frr_results.txt", "w+");

  evaluateResult(db_feature, in_feature, not_feature, db_name, in_name, not_name, db_count,
                 in_count, not_count, sim_thresh, fp_far, fp_frr);
  fclose(fp_far);
  fclose(fp_frr);

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

  CVI_AI_Service_DestroyHandle(service_handle);
  CVI_AI_DestroyHandle(facelib_handle);
  CVI_VPSS_StopGrp(VpssGrp);
  CVI_VPSS_DestroyGrp(VpssGrp);
  CVI_SYS_Exit();
  return ret;
}
