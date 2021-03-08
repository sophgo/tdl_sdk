#define _GNU_SOURCE
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "cviai_perfetto.h"

#define WRITE_RESULT_TO_FILE 0

#define SAVE_TRACKER_NUM 64
#define QUALITY_THRESHOLD 0.95
#define COVER_RATE_THRESHOLD 0.9
#define MISS_TIME_LIMIT 100

typedef int (*InferenceFunc)(cviai_handle_t, VIDEO_FRAME_INFO_S *, cvai_object_t *,
                             cvai_obj_det_type_e);
typedef struct _ModelConfig {
  CVI_AI_SUPPORTED_MODEL_E model_id;
  int input_size;
  InferenceFunc inference;
} ModelConfig;

CVI_S32 createModelConfig(const char *model_name, ModelConfig *config) {
  CVI_S32 ret = CVI_SUCCESS;

  if (strcmp(model_name, "mobiledetv2-d0") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D0;
    config->inference = CVI_AI_MobileDetV2_D0;
  } else if (strcmp(model_name, "mobiledetv2-d1") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D1;
    config->inference = CVI_AI_MobileDetV2_D1;
  } else if (strcmp(model_name, "mobiledetv2-d2") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D2;
    config->inference = CVI_AI_MobileDetV2_D2;
  } else if (strcmp(model_name, "yolov3") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_YOLOV3;
    config->inference = CVI_AI_Yolov3;
  } else {
    ret = CVI_FAILURE;
  }
  return ret;
}

typedef enum { MISS = 0, ALIVE } tracker_state_e;

typedef struct {
  uint64_t id;
  tracker_state_e state;
  cvai_feature_t feature;
  float quality;
  VIDEO_FRAME_INFO_S face;
} face_quality_tracker_t;

typedef struct {
  bool match;
  uint32_t idx;
} match_index_t;

void feature_copy(cvai_feature_t *src_feature, cvai_feature_t *dst_feature) {
  dst_feature->size = src_feature->size;
  dst_feature->type = src_feature->type;
  size_t type_size = getFeatureTypeSize(dst_feature->type);
  dst_feature->ptr = (int8_t *)malloc(dst_feature->size * type_size);
  memcpy(dst_feature->ptr, src_feature->ptr, dst_feature->size * type_size);
}

bool update_tracker(cviai_handle_t ai_handle, VIDEO_FRAME_INFO_S *frame,
                    face_quality_tracker_t *fq_trackers, cvai_face_t *face_meta,
                    cvai_tracker_t *tracker_meta, int *miss_time) {
  for (int j = 0; j < SAVE_TRACKER_NUM; j++) {
    if (fq_trackers[j].state == ALIVE) {
      miss_time[j] += 1;
    }
  }
  for (uint32_t i = 0; i < tracker_meta->size; i++) {
    /* we only consider the stable tracker in this sample code. */
    if (tracker_meta->info[i].state != CVI_TRACKER_STABLE) {
      continue;
    }
    uint64_t trk_id = face_meta->info[i].unique_id;
    /* check whether the tracker id exist or not. */
    int match_idx = -1;
    for (int j = 0; j < SAVE_TRACKER_NUM; j++) {
      if (fq_trackers[j].state == ALIVE && fq_trackers[j].id == trk_id) {
        match_idx = j;
        break;
      }
    }
    if (match_idx == -1) {
      /* if not found, create new one. */
      bool is_created = false;
      /* search available index for new tracker. */
      for (int j = 0; j < SAVE_TRACKER_NUM; j++) {
        if (fq_trackers[j].state == MISS) {
          miss_time[j] = 0;
          fq_trackers[j].state = ALIVE;
          fq_trackers[j].id = trk_id;
          fq_trackers[j].face.stVFrame.u32Height = 112;
          fq_trackers[j].face.stVFrame.u32Width = 112;
          for (int chn = 0; chn < 3; chn++) {
            fq_trackers[j].face.stVFrame.pu8VirAddr[chn] =
                (CVI_U8 *)malloc(112 * 112 * sizeof(CVI_U8));
            memset(fq_trackers[j].face.stVFrame.pu8VirAddr[chn], 0, 112 * 112 * sizeof(CVI_U8));
          }
          if (face_meta->info[i].face_quality.quality >= QUALITY_THRESHOLD) {
            fq_trackers[j].quality = face_meta->info[i].face_quality.quality;
            feature_copy(&fq_trackers[j].feature, &face_meta->info[i].feature);
            if (!CVI_AI_GetAlignedFace(ai_handle, frame, &fq_trackers[j].face,
                                       &face_meta->info[i])) {
              printf("AI get aligned face failed(1).\n");
              return false;
            }
            // TODO: add pitch, roll, yaw
          }
          is_created = true;
          break;
        }
      }
      /* if fail to create, return false. */
      if (!is_created) {
        printf("buffer overflow.\n");
        return false;
      }
    } else {
      /* if found, check whether the quality(or feature) need to be update. */
      miss_time[match_idx] = 0;
      if (face_meta->info[i].face_quality.quality >= QUALITY_THRESHOLD &&
          face_meta->info[i].face_quality.quality > fq_trackers[match_idx].quality) {
        fq_trackers[match_idx].quality = face_meta->info[i].face_quality.quality;
        feature_copy(&fq_trackers[match_idx].feature, &face_meta->info[i].feature);
        CVI_S32 ret = CVI_AI_GetAlignedFace(ai_handle, frame, &fq_trackers[match_idx].face,
                                            &face_meta->info[i]);
        if (ret != CVI_SUCCESS) {
          printf("AI get aligned face failed(2).\n");
          return false;
        }
        // TODO: add pitch, roll, yaw
      }
    }
  }
  return true;
}

void clean_tracker(face_quality_tracker_t *fq_trackers, int *miss_time) {
  for (int j = 0; j < SAVE_TRACKER_NUM; j++) {
    if (fq_trackers[j].state == ALIVE && miss_time[j] > MISS_TIME_LIMIT) {
      free(fq_trackers[j].face.stVFrame.pu8VirAddr[0]);
      free(fq_trackers[j].face.stVFrame.pu8VirAddr[1]);
      free(fq_trackers[j].face.stVFrame.pu8VirAddr[2]);
      memset(&fq_trackers[j], 0, sizeof(face_quality_tracker_t));
      miss_time[j] = -1;
    }
  }
}

int get_alive_num(face_quality_tracker_t *fq_trackers) {
  int counter = 0;
  for (int j = 0; j < SAVE_TRACKER_NUM; j++) {
    if (fq_trackers[j].state == ALIVE) {
      counter += 1;
    }
  }
  return counter;
}

float cover_rate_face2people(cvai_bbox_t face_bbox, cvai_bbox_t people_bbox) {
  float inter_x1 = MAX2(face_bbox.x1, people_bbox.x1);
  float inter_y1 = MAX2(face_bbox.y1, people_bbox.y1);
  float inter_x2 = MIN2(face_bbox.x2, people_bbox.x2);
  float inter_y2 = MIN2(face_bbox.y2, people_bbox.y2);
  float inter_w = MAX2(0.0f, inter_x2 - inter_x1);
  float inter_h = MAX2(0.0f, inter_y2 - inter_y1);
  float inter_area = inter_w * inter_h;
  float face_w = MAX2(0.0f, face_bbox.x2 - face_bbox.x1);
  float face_h = MAX2(0.0f, face_bbox.y2 - face_bbox.y1);
  float face_area = face_w * face_h;
  return inter_area / face_area;
}

int main(int argc, char *argv[]) {
  CVI_AI_PerfettoInit();
  if (argc != 8) {
    printf(
        "Usage: %s <obj_detection_model_name>\n"
        "          <obj_detection_model_path>\n"
        "          <face_detection_model_path>\n"
        "          <face_attribute_model_path>\n"
        "          <face_quality_model_path>\n"
        "          <sample_imagelist_path>\n"
        "          <inference_count>\n",
        argv[0]);
    return CVI_FAILURE;
  }
  CVI_S32 ret = CVI_SUCCESS;

  face_quality_tracker_t fq_trackers[SAVE_TRACKER_NUM];
  memset(fq_trackers, 0, sizeof(face_quality_tracker_t) * SAVE_TRACKER_NUM);
  int miss_time[SAVE_TRACKER_NUM];
  memset(miss_time, -1, sizeof(int) * SAVE_TRACKER_NUM);

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

  ModelConfig model_config;
  if (createModelConfig(argv[1], &model_config) == CVI_FAILURE) {
    printf("unsupported model: %s\n", argv[1]);
    return CVI_FAILURE;
  }

  ret = CVI_AI_CreateHandle2(&ai_handle, 1);
  ret |= CVI_AI_SetModelPath(ai_handle, model_config.model_id, argv[2]);
  ret |= CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, argv[3]);
  ret |= CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE, argv[4]);
  ret |= CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_FACEQUALITY, argv[5]);
  if (ret != CVI_SUCCESS) {
    printf("failed with %#x!\n", ret);
    return ret;
  }
  CVI_AI_SetSkipVpssPreprocess(ai_handle, model_config.model_id, false);
  CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, false);
  CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE, false);

  // Init DeepSORT
  CVI_AI_DeepSORT_Init(ai_handle);
#if 1
  cvai_deepsort_config_t ds_conf;
  CVI_AI_DeepSORT_GetDefaultConfig(&ds_conf);
  ds_conf.ktracker_conf.max_unmatched_num = 10;
  ds_conf.ktracker_conf.accreditation_threshold = 10;
  ds_conf.ktracker_conf.P_std_beta[2] = 0.1;
  ds_conf.ktracker_conf.P_std_beta[6] = 2.5e-2;
  ds_conf.kfilter_conf.Q_std_beta[2] = 0.1;
  ds_conf.kfilter_conf.Q_std_beta[6] = 2.5e-2;
  ds_conf.kfilter_conf.R_std_beta[2] = 0.1;
  CVI_AI_DeepSORT_SetConfig(ai_handle, &ds_conf);
#endif

#if WRITE_RESULT_TO_FILE
  FILE *outFile;
  outFile = fopen("sample_fq_tracking_result.txt", "w");
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
    printf("\n[%i] image path = %s\n", counter, image_path);

    int trk_num = get_alive_num(fq_trackers);
    printf("FQ Tracker Num = %d\n", trk_num);

    VB_BLK blk_fr;
    VIDEO_FRAME_INFO_S frame;
    CVI_S32 ret = CVI_AI_ReadImage(image_path, &blk_fr, &frame, PIXEL_FORMAT_RGB_888);
    if (ret != CVI_SUCCESS) {
      printf("Read image failed with %#x!\n", ret);
      return ret;
    }

    cvai_face_t face_meta;
    memset(&face_meta, 0, sizeof(cvai_face_t));
    cvai_tracker_t tracker_meta;
    memset(&tracker_meta, 0, sizeof(cvai_tracker_t));
    cvai_object_t obj_meta;
    memset(&obj_meta, 0, sizeof(cvai_object_t));

    CVI_AI_RetinaFace(ai_handle, &frame, &face_meta);
    printf("Found %x faces.\n", face_meta.size);
    CVI_AI_FaceQuality(ai_handle, &frame, &face_meta);

    CVI_AI_DeepSORT_Face(ai_handle, &face_meta, &tracker_meta, false);

    for (uint32_t i = 0; i < tracker_meta.size; i++) {
      printf("[%u][%lu] [%d] (%d,%d,%d,%d) <%f>\n", i, face_meta.info[i].unique_id,
             tracker_meta.info[i].state, (int)tracker_meta.info[i].bbox.x1,
             (int)tracker_meta.info[i].bbox.y1, (int)tracker_meta.info[i].bbox.x2,
             (int)tracker_meta.info[i].bbox.y2, face_meta.info[i].face_quality.quality);
    }

    if (!update_tracker(ai_handle, &frame, fq_trackers, &face_meta, &tracker_meta, miss_time)) {
      printf("update tracker failed.\n");
      CVI_AI_Free(&face_meta);
      CVI_AI_Free(&tracker_meta);
      CVI_AI_Free(&obj_meta);
      CVI_VB_ReleaseBlock(blk_fr);
      break;
    }
    clean_tracker(fq_trackers, miss_time);

    model_config.inference(ai_handle, &frame, &obj_meta, CVI_DET_TYPE_PEOPLE);
    match_index_t *p2f = (match_index_t *)malloc(obj_meta.size * sizeof(match_index_t));
    memset(p2f, 0, obj_meta.size * sizeof(match_index_t));
    for (uint32_t i = 0; i < obj_meta.size; i++) {
      for (uint32_t j = 0; j < face_meta.size; j++) {
        if (cover_rate_face2people(face_meta.info[j].bbox, obj_meta.info[i].bbox) >=
            COVER_RATE_THRESHOLD) {
          p2f[i].match = true;
          p2f[i].idx = j;
          break;
        }
      }
    }

#if WRITE_RESULT_TO_FILE
    fprintf(outFile, "%u\n", tracker_meta.size);
    for (uint32_t i = 0; i < tracker_meta.size; i++) {
      fprintf(outFile, "%lu,%d,%d,%d,%d,%d,%d,%d,%d,%d,%f,%f,%f,%f\n", face_meta.info[i].unique_id,
              (int)face_meta.info[i].bbox.x1, (int)face_meta.info[i].bbox.y1,
              (int)face_meta.info[i].bbox.x2, (int)face_meta.info[i].bbox.y2,
              tracker_meta.info[i].state, (int)tracker_meta.info[i].bbox.x1,
              (int)tracker_meta.info[i].bbox.y1, (int)tracker_meta.info[i].bbox.x2,
              (int)tracker_meta.info[i].bbox.y2, face_meta.info[i].face_quality.quality,
              face_meta.info[i].face_quality.pitch, face_meta.info[i].face_quality.roll,
              face_meta.info[i].face_quality.yaw);
    }

    // fprintf(outFile, "%u\n", 0);
    char debug_info[8192];
    CVI_AI_DeepSORT_DebugInfo_1(ai_handle, debug_info);
    fprintf(outFile, debug_info);

    fprintf(outFile, "%u\n", obj_meta.size);
    for (uint32_t i = 0; i < obj_meta.size; i++) {
      fprintf(outFile, "%d,%lu,%d,%d,%d,%d\n", (p2f[i].match) ? 1 : 0,
              (p2f[i].match) ? face_meta.info[p2f[i].idx].unique_id : -1,
              (int)obj_meta.info[i].bbox.x1, (int)obj_meta.info[i].bbox.y1,
              (int)obj_meta.info[i].bbox.x2, (int)obj_meta.info[i].bbox.y2);
    }
#endif

    free(p2f);
    CVI_AI_Free(&face_meta);
    CVI_AI_Free(&tracker_meta);
    CVI_AI_Free(&obj_meta);
    CVI_VB_ReleaseBlock(blk_fr);
  }

#if WRITE_RESULT_TO_FILE
  fclose(outFile);
#endif
  for (int i = 0; i < SAVE_TRACKER_NUM; i++) {
    if (fq_trackers[i].state == ALIVE) {
      free(fq_trackers[i].face.stVFrame.pu8VirAddr[0]);
      free(fq_trackers[i].face.stVFrame.pu8VirAddr[1]);
      free(fq_trackers[i].face.stVFrame.pu8VirAddr[2]);
    }
  }
  CVI_AI_DestroyHandle(ai_handle);
  CVI_SYS_Exit();
}