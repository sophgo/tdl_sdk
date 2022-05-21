#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "sample_comm.h"
#include "sample_utils.h"
#include "vi_vo_utils.h"

#include <cvi_sys.h>
#include <cvi_vb.h>
#include <cvi_vi.h>

#include <inttypes.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static volatile bool bExit = false;

#define SAVE_TRACKER_NUM 32
#define QUALITY_THRESHOLD 0.95
#define COVER_RATE_THRESHOLD 0.9
#define MISS_TIME_LIMIT 100

/* cv182x can't detect object and face at the same time. */
#define DETECT_PEOPLE 0

typedef enum { MISS = 0, ALIVE } tracker_state_e;

typedef struct {
  uint64_t id;
  tracker_state_e state;
  cvai_feature_t feature;
  float quality;
  VIDEO_FRAME_INFO_S face; /* TODO: remove this */
  float pitch;
  float roll;
  float yaw;
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
          fq_trackers[j].pitch = face_meta->info[i].head_pose.pitch;
          fq_trackers[j].roll = face_meta->info[i].head_pose.roll;
          fq_trackers[j].yaw = face_meta->info[i].head_pose.yaw;
          if (face_meta->info[i].face_quality >= QUALITY_THRESHOLD) {
            fq_trackers[j].quality = face_meta->info[i].face_quality;
            feature_copy(&fq_trackers[j].feature, &face_meta->info[i].feature);
#if 0
            CVI_S32 ret =
                CVI_AI_GetAlignedFace(ai_handle, frame, &fq_trackers[j].face, &face_meta->info[i]);
            if (ret != CVIAI_SUCCESS) {
              printf("AI get aligned face failed(1).\n");
              return false;
            }
#endif
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
      fq_trackers[match_idx].pitch = face_meta->info[i].head_pose.pitch;
      fq_trackers[match_idx].roll = face_meta->info[i].head_pose.roll;
      fq_trackers[match_idx].yaw = face_meta->info[i].head_pose.yaw;
      if (face_meta->info[i].face_quality >= QUALITY_THRESHOLD &&
          face_meta->info[i].face_quality > fq_trackers[match_idx].quality) {
        fq_trackers[match_idx].quality = face_meta->info[i].face_quality;
        feature_copy(&fq_trackers[match_idx].feature, &face_meta->info[i].feature);
#if 0
        CVI_S32 ret = CVI_AI_GetAlignedFace(ai_handle, frame, &fq_trackers[match_idx].face,
                                            &face_meta->info[i]);
        if (ret != CVIAI_SUCCESS) {
          printf("AI get aligned face failed(2).\n");
          return false;
        }
#endif
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
  if (argc != 7) {
    printf(
        "Usage: %s <obj_detection_model_name>\n"
        "          <obj_detection_model_path>\n"
        "          <face_detection_model_path>\n"
        "          <face_attribute_model_path>\n"
        "          <face_quality_model_path>\n"
        "          video output, 0: disable, 1: output to panel, 2: output through rtsp\n",
        argv[0]);
    return CVIAI_FAILURE;
  }

  CVI_S32 voType = atoi(argv[6]);

  CVI_S32 s32Ret = CVIAI_SUCCESS;
  VideoSystemContext vs_ctx = {0};
  SIZE_S aiInputSize = {.u32Width = 1280, .u32Height = 720};

  if (InitVideoSystem(&vs_ctx, &aiInputSize, PIXEL_FORMAT_RGB_888, voType) != CVI_SUCCESS) {
    printf("failed to init video system\n");
    return CVIAI_FAILURE;
  }

  face_quality_tracker_t fq_trackers[SAVE_TRACKER_NUM];
  memset(fq_trackers, 0, sizeof(face_quality_tracker_t) * SAVE_TRACKER_NUM);
  int miss_time[SAVE_TRACKER_NUM];
  memset(miss_time, -1, sizeof(int) * SAVE_TRACKER_NUM);

  cviai_handle_t ai_handle = NULL;
  cviai_service_handle_t service_handle = NULL;
  ODInferenceFunc inference;
  CVI_AI_SUPPORTED_MODEL_E od_model_id;
  if (get_pd_model_info(argv[1], &od_model_id, &inference) == CVIAI_FAILURE) {
    printf("unsupported model: %s\n", argv[1]);
    return CVIAI_FAILURE;
  }

  GOTO_IF_FAILED(CVI_AI_CreateHandle2(&ai_handle, 1, 0), s32Ret, create_ai_fail);
  GOTO_IF_FAILED(CVI_AI_Service_CreateHandle(&service_handle, ai_handle), s32Ret,
                 create_service_fail);

  GOTO_IF_FAILED(CVI_AI_OpenModel(ai_handle, od_model_id, argv[2]), s32Ret, setup_ai_fail);
  GOTO_IF_FAILED(CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, argv[3]), s32Ret,
                 setup_ai_fail);
  GOTO_IF_FAILED(CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_FACERECOGNITION, argv[4]),
                 s32Ret, setup_ai_fail);
  GOTO_IF_FAILED(CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_FACEQUALITY, argv[5]), s32Ret,
                 setup_ai_fail);
  GOTO_IF_FAILED(CVI_AI_SelectDetectClass(ai_handle, od_model_id, 1, CVI_AI_DET_TYPE_PERSON),
                 s32Ret, setup_ai_fail);
  CVI_AI_SetVpssTimeout(ai_handle, 1000);

  // Init DeepSORT
  CVI_AI_DeepSORT_Init(ai_handle, false);
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
  CVI_AI_DeepSORT_SetConfig(ai_handle, &ds_conf, -1, false);
#endif

  VIDEO_FRAME_INFO_S stfdFrame, stVOFrame;
  size_t counter = 0;
  while (bExit == false) {
    counter += 1;
    printf("\nGet Frame %zu\n", counter);

    s32Ret = CVI_VPSS_GetChnFrame(vs_ctx.vpssConfigs.vpssGrp, vs_ctx.vpssConfigs.vpssChnAI,
                                  &stfdFrame, 2000);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
      break;
    }

    int trk_num = get_alive_num(fq_trackers);
    printf("FQ Tracker Num = %d\n", trk_num);

    cvai_face_t face_meta;
    memset(&face_meta, 0, sizeof(cvai_face_t));
    cvai_tracker_t tracker_meta;
    memset(&tracker_meta, 0, sizeof(cvai_tracker_t));
#if DETECT_PEOPLE
    cvai_object_t obj_meta;
    memset(&obj_meta, 0, sizeof(cvai_object_t));
#endif

    CVI_AI_RetinaFace(ai_handle, &stfdFrame, &face_meta);
    printf("Found %x faces.\n", face_meta.size);
    CVI_AI_FaceRecognition(ai_handle, &stfdFrame, &face_meta);
    CVI_AI_Service_FaceAngleForAll(&face_meta);
    CVI_AI_FaceQuality(ai_handle, &stfdFrame, &face_meta, NULL);
    for (uint32_t j = 0; j < face_meta.size; j++) {
      printf("face[%u] quality: %f\n", j, face_meta.info[j].face_quality);
    }

    // CVI_AI_DeepSORT_Face(ai_handle, &face_meta, &tracker_meta, false);
    CVI_AI_DeepSORT_Face(ai_handle, &face_meta, &tracker_meta, true);

    if (!update_tracker(ai_handle, &stfdFrame, fq_trackers, &face_meta, &tracker_meta, miss_time)) {
      printf("update tracker failed.\n");
      CVI_AI_Free(&face_meta);
      CVI_AI_Free(&tracker_meta);
#if DETECT_PEOPLE
      CVI_AI_Free(&obj_meta);
#endif
      break;
    }
    clean_tracker(fq_trackers, miss_time);

#if DETECT_PEOPLE
    inference(ai_handle, &stfdFrame, &obj_meta);
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
#endif

    s32Ret = CVI_VPSS_ReleaseChnFrame(vs_ctx.vpssConfigs.vpssGrp, vs_ctx.vpssConfigs.vpssChnAI,
                                      &stfdFrame);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
      break;
    }

    // Send frame to VO if opened.
    if (voType) {
      s32Ret = CVI_VPSS_GetChnFrame(vs_ctx.vpssConfigs.vpssGrp,
                                    vs_ctx.vpssConfigs.vpssChnVideoOutput, &stVOFrame, 1000);
      if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
        break;
      }
      CVI_AI_Service_FaceDrawRect(service_handle, &face_meta, &stVOFrame, false,
                                  CVI_AI_Service_GetDefaultBrush());
      for (uint32_t j = 0; j < face_meta.size; j++) {
        char *id_num = calloc(64, sizeof(char));
        sprintf(id_num, "%" PRIu64 "", face_meta.info[j].unique_id);
        CVI_AI_Service_ObjectWriteText(id_num, face_meta.info[j].bbox.x1, face_meta.info[j].bbox.y1,
                                       &stVOFrame, -1, -1, -1);
        free(id_num);
        char *fq_score = calloc(64, sizeof(char));
        sprintf(fq_score, "%.2f", face_meta.info[j].face_quality);
        CVI_AI_Service_ObjectWriteText(fq_score, face_meta.info[j].bbox.x1,
                                       face_meta.info[j].bbox.y1 + 45, &stVOFrame, -1, -1, -1);
        free(fq_score);
      }

#if DETECT_PEOPLE
      for (uint32_t i = 0; i < obj_meta.size; i++) {
        if (p2f[i].match) {
          char *id_num = calloc(64, sizeof(char));
          sprintf(id_num, "%" PRIu64 "", face_meta.info[p2f[i].idx].unique_id);
          CVI_AI_Service_ObjectWriteText(id_num, obj_meta.info[i].bbox.x1, obj_meta.info[i].bbox.y1,
                                         &stVOFrame, -1, -1, -1);
          free(id_num);
        }
      }
      CVI_AI_Service_ObjectDrawRect(service_handle, &obj_meta, &stVOFrame, false,
                                    CVI_AI_Service_GetDefaultBrush());
#endif

      s32Ret = SendOutputFrame(&stVOFrame, &vs_ctx.outputContext);
      if (s32Ret != CVI_SUCCESS) {
        printf("Send Output Frame NG\n");
      }

      s32Ret = CVI_VPSS_ReleaseChnFrame(vs_ctx.vpssConfigs.vpssGrp,
                                        vs_ctx.vpssConfigs.vpssChnVideoOutput, &stVOFrame);
      if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
        break;
      }
    }

    CVI_AI_Free(&face_meta);
    CVI_AI_Free(&tracker_meta);
#if DETECT_PEOPLE
    free(p2f);
    CVI_AI_Free(&obj_meta);
#endif
  }

  for (int i = 0; i < SAVE_TRACKER_NUM; i++) {
    if (fq_trackers[i].state == ALIVE) {
      free(fq_trackers[i].face.stVFrame.pu8VirAddr[0]);
      free(fq_trackers[i].face.stVFrame.pu8VirAddr[1]);
      free(fq_trackers[i].face.stVFrame.pu8VirAddr[2]);
    }
  }

setup_ai_fail:
  CVI_AI_Service_DestroyHandle(service_handle);
create_service_fail:
  CVI_AI_DestroyHandle(ai_handle);
create_ai_fail:
  DestroyVideoSystem(&vs_ctx);
  CVI_SYS_Exit();
  CVI_VB_Exit();
}
