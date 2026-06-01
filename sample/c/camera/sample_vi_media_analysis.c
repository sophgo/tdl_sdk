#include <dirent.h>
#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#include <ctype.h>
#include <pthread.h>
#include "cvi_comm_video.h"
#include "cvi_vi.h"
#include "pthread_utils.h"
#include "sample_utils.h"
#include "tdl_ex.h"
#include "tdl_sdk.h"

#define VI_WIDTH 1920
#define VI_HEIGHT 1080
#define FEATURE_SIZE 256
#define SIMILARITY_THRESHOLD 0.45f
#define MAX_REGISTERED_FACES 256

// H.264 behavior video recording
#define BEHAVIOR_VIDEO_MAX_SEC 3
#define BEHAVIOR_VIDEO_FPS 5
#define BEHAVIOR_VIDEO_MAX_FRAMES (BEHAVIOR_VIDEO_FPS * BEHAVIOR_VIDEO_MAX_SEC)
#define BEHAVIOR_VENC_CHN 1
#define BEHAVIOR_FRAME_INTERVAL 5  // encode every 5th frame
#define BEHAVIOR_TIMEOUT_FRAMES \
  25  // ~1 second at 25fps, person disappeared timeout

static const char *emotion_str[] = {"Anger",   "Disgust", "Fear",    "Happy",
                                    "Neutral", "Sad",     "Surprise"};

static volatile bool to_exit = false;
static ImageQueue image_queue;
static int8_t g_registered_faces[MAX_REGISTERED_FACES][FEATURE_SIZE];
static int g_registered_face_ids[MAX_REGISTERED_FACES];
static int g_registered_face_count = 0;
static pthread_mutex_t g_face_mutex = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
  TDLHandle tdl_handle;
  int vi_chn;
  uint8_t channel_size;
  char **channel_names;
} SEND_FRAME_THREAD_ARG_S;

typedef struct {
  TDLHandle tdl_handle;
  int vi_chn;
  uint8_t channel_size;
  char **channel_names;
  char data_dir[512];
  TDLFeatureInfo *gallery_feature;
} RUN_TDL_THREAD_ARG_S;

typedef struct {
  bool is_recording;
  int person_id;
  uint64_t track_id;
  int encoded_frames;
  int skip_counter;
  int person_lost_counter;
  uint64_t appearance_id;
  char file_path[512];
  FILE *h264_file;
} BehaviorRecorder;

static BehaviorRecorder g_behavior_recorder = {.is_recording = false,
                                               .encoded_frames = 0,
                                               .skip_counter = 0,
                                               .person_lost_counter = 0,
                                               .appearance_id = 0,
                                               .h264_file = NULL};
static uint64_t g_appearance_id_counter = 0;

static uint32_t get_time_in_ms() {
  struct timeval tv;
  if (gettimeofday(&tv, NULL) < 0) {
    return 0;
  }
  return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

static void handle_signal(int signo) {
  if (signo == SIGINT || signo == SIGTERM) {
    to_exit = true;
    ExitQueue(&image_queue);
  }
}

static int ensure_dir(const char *path) {
  if (path == NULL || path[0] == '\0') {
    return -1;
  }
  if (mkdir(path, 0755) == 0) {
    return 0;
  }
  if (errno == EEXIST) {
    return 0;
  }
  return -1;
}

typedef struct {
  int id;
  char name[128];
} RegisteredNameEntry;

static void trim_trailing_spaces(char *s) {
  if (s == NULL) return;
  size_t len = strlen(s);
  while (len > 0 && isspace((unsigned char)s[len - 1])) {
    s[len - 1] = '\0';
    len--;
  }
}

static int load_registered_name_map(const char *registered_info_path,
                                    RegisteredNameEntry *entries,
                                    int max_entries) {
  if (registered_info_path == NULL || entries == NULL || max_entries <= 0) {
    return 0;
  }

  FILE *fp = fopen(registered_info_path, "r");
  if (fp == NULL) {
    return 0;
  }

  int count = 0;
  char line[256];
  while (fgets(line, sizeof(line), fp) != NULL && count < max_entries) {
    size_t len = strlen(line);
    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
      line[len - 1] = '\0';
      len--;
    }
    if (len == 0) continue;

    char *tail = line + len - 1;
    while (tail >= line && isspace((unsigned char)(*tail))) {
      *tail = '\0';
      tail--;
    }
    if (tail < line) continue;

    while (tail >= line && !isspace((unsigned char)(*tail))) {
      tail--;
    }
    if (tail < line) continue;

    char *id_str = tail + 1;
    *tail = '\0';
    trim_trailing_spaces(line);
    if (line[0] == '\0') continue;

    char *end_ptr = NULL;
    long id_val = strtol(id_str, &end_ptr, 10);
    if (end_ptr == id_str || *end_ptr != '\0') {
      continue;
    }

    entries[count].id = (int)id_val;
    strncpy(entries[count].name, line, sizeof(entries[count].name) - 1);
    entries[count].name[sizeof(entries[count].name) - 1] = '\0';
    count++;
  }

  fclose(fp);
  return count;
}

static const char *find_registered_name(const RegisteredNameEntry *entries,
                                        int count, int id) {
  if (entries == NULL || count <= 0) {
    return NULL;
  }
  for (int i = 0; i < count; i++) {
    if (entries[i].id == id) {
      return entries[i].name;
    }
  }
  return NULL;
}

static int json_escape_copy(const char *src, char *dst, size_t dst_size) {
  if (src == NULL || dst == NULL || dst_size == 0) {
    return 0;
  }

  size_t pos = 0;
  for (size_t i = 0; src[i] != '\0' && pos + 1 < dst_size; i++) {
    if ((src[i] == '"' || src[i] == '\\') && pos + 2 < dst_size) {
      dst[pos++] = '\\';
      dst[pos++] = src[i];
    } else if (src[i] != '"' && src[i] != '\\') {
      dst[pos++] = src[i];
    } else {
      break;
    }
  }
  dst[pos] = '\0';
  return (int)pos;
}

static void move_unknown_person_images(const char *identity_dir,
                                       int registered_id, uint64_t track_id) {
  if (identity_dir == NULL || registered_id == -1) {
    return;
  }

  char unknown_dir[512];
  snprintf(unknown_dir, sizeof(unknown_dir), "%s/-1", identity_dir);
  DIR *dir = opendir(unknown_dir);
  if (dir == NULL) {
    return;
  }

  char target_dir[512];
  snprintf(target_dir, sizeof(target_dir), "%s/%d", identity_dir,
           registered_id);
  if (ensure_dir(target_dir) != 0) {
    closedir(dir);
    return;
  }

  char track_token[64];
  snprintf(track_token, sizeof(track_token), "_personID_%" PRIu64 "_",
           track_id);

  struct dirent *entry = NULL;
  while ((entry = readdir(dir)) != NULL) {
    if (entry->d_type != DT_REG) continue;
    const char *filename = entry->d_name;
    size_t file_len = strlen(filename);
    if (file_len < 4 || strcmp(filename + file_len - 4, ".jpg") != 0) {
      continue;
    }
    if (strstr(filename, track_token) == NULL) {
      continue;
    }

    char new_filename[768];
    strncpy(new_filename, filename, sizeof(new_filename) - 1);
    new_filename[sizeof(new_filename) - 1] = '\0';

    char *reg_pos = strstr(new_filename, "_registeredID_-1");
    if (reg_pos != NULL) {
      char tail[512];
      strncpy(tail, reg_pos + strlen("_registeredID_-1"), sizeof(tail) - 1);
      tail[sizeof(tail) - 1] = '\0';
      snprintf(reg_pos, (size_t)(new_filename + sizeof(new_filename) - reg_pos),
               "_registeredID_%d%s", registered_id, tail);
    }

    char src_path[1024];
    char dst_path[1024];
    snprintf(src_path, sizeof(src_path), "%s/%s", unknown_dir, filename);
    snprintf(dst_path, sizeof(dst_path), "%s/%s", target_dir, new_filename);
    if (rename(src_path, dst_path) != 0) {
      printf("rename failed: %s -> %s, err=%s\n", src_path, dst_path,
             strerror(errno));
    }
  }

  closedir(dir);
}

static void build_attr_str(const TDLSnapshotInfo *snapshot, char *attr,
                           size_t attr_size) {
  attr[0] = '\0';
  if (snapshot->object_type != TDL_OBJECT_TYPE_FACE) {
    return;
  }
  uint8_t emotion = snapshot->emotion;
  if (emotion > 6) {
    emotion = 4;
  }
  snprintf(attr, attr_size, "_male[%d]_glass[%d]_age[%d]_emotion[%s]",
           snapshot->male, snapshot->glass, snapshot->age,
           emotion_str[emotion]);
}

static int feature_to_int8(const TDLFeature *feature, int8_t *dst,
                           uint32_t dst_size) {
  if (feature == NULL || feature->ptr == NULL || dst == NULL) {
    return -1;
  }
  if (feature->size < dst_size) {
    return -1;
  }

  if (feature->type == TDL_TYPE_INT8) {
    memcpy(dst, feature->ptr, dst_size);
    return 0;
  }

  if (feature->type == TDL_TYPE_FP32) {
    const float *src = (const float *)(feature->ptr);
    for (uint32_t i = 0; i < dst_size; i++) {
      float v = src[i];
      if (v > 127.0f) v = 127.0f;
      if (v < -128.0f) v = -128.0f;
      dst[i] = (int8_t)v;
    }
    return 0;
  }

  for (uint32_t i = 0; i < dst_size; i++) {
    dst[i] = feature->ptr[i];
  }
  return 0;
}

static int save_feature_bin(const char *path, const TDLFeature *feature) {
  if (path == NULL || feature == NULL || feature->ptr == NULL) {
    return -1;
  }
  int8_t feat_int8[FEATURE_SIZE] = {0};
  if (feature_to_int8(feature, feat_int8, FEATURE_SIZE) != 0) {
    return -1;
  }
  FILE *fp = fopen(path, "wb");
  if (fp == NULL) {
    return -1;
  }
  fwrite(feat_int8, 1, FEATURE_SIZE, fp);
  fclose(fp);
  return 0;
}

// ---- Behavior Video Recording Helpers ----

static void behavior_recorder_stop_and_submit(const char *data_dir);

static void behavior_recorder_start(int person_id, uint64_t track_id,
                                    const char *data_dir) {
  BehaviorRecorder *rec = &g_behavior_recorder;
  if (rec->is_recording) {
    behavior_recorder_stop_and_submit(data_dir);
  }

  char behavior_video_dir[512];
  snprintf(behavior_video_dir, sizeof(behavior_video_dir), "%s/behavior_video",
           data_dir);
  ensure_dir(behavior_video_dir);

  uint64_t app_id = g_appearance_id_counter++;
  time_t now = time(NULL);
  struct tm *tm_info = localtime(&now);
  char ts[32];
  strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", tm_info);

  snprintf(rec->file_path, sizeof(rec->file_path),
           "%s/p_%d_%s_app_%03" PRIu64 ".h264", behavior_video_dir, person_id,
           ts, app_id);

  rec->h264_file = fopen(rec->file_path, "wb");
  if (rec->h264_file == NULL) {
    printf("BehaviorRecorder: failed to create file %s\n", rec->file_path);
    return;
  }

  rec->is_recording = true;
  rec->person_id = person_id;
  rec->track_id = track_id;
  rec->encoded_frames = 0;
  rec->skip_counter = 0;
  rec->person_lost_counter = 0;
  rec->appearance_id = app_id;

  printf("BehaviorRecorder: started recording person_id=%d track_id=%" PRIu64
         " → %s\n",
         person_id, track_id, rec->file_path);
}

static void behavior_recorder_stop_and_submit(const char *data_dir) {
  BehaviorRecorder *rec = &g_behavior_recorder;
  if (!rec->is_recording) {
    return;
  }

  if (rec->h264_file) {
    fclose(rec->h264_file);
    rec->h264_file = NULL;
  }

  printf("BehaviorRecorder: stopped, encoded_frames=%d, file=%s\n",
         rec->encoded_frames, rec->file_path);

  // 如果录制帧数太少（< 3帧），丢弃（路过式出现，无分析价值）
  if (rec->encoded_frames < 3) {
    printf("BehaviorRecorder: discarding (too few frames)\n");
    remove(rec->file_path);
  } else {
#if !defined(__CMODEL_CV181X__) && !defined(__CMODEL_CV184X__)
    char person_name[128];
    snprintf(person_name, sizeof(person_name), "人员%d", rec->person_id + 1);
    uint32_t duration_sec =
        (rec->encoded_frames + BEHAVIOR_VIDEO_FPS - 1) / BEHAVIOR_VIDEO_FPS;
    TDL_MediaAnalysisServer_SubmitBehaviorVideo(
        rec->file_path, person_name, rec->person_id, rec->appearance_id,
        duration_sec);
#endif
  }

  rec->is_recording = false;
  rec->encoded_frames = 0;
  rec->skip_counter = 0;
  rec->person_lost_counter = 0;
}

static void behavior_recorder_encode_frame(TDLHandle tdl_handle,
                                           TDLImage image) {
  BehaviorRecorder *rec = &g_behavior_recorder;
  if (!rec->is_recording) {
    return;
  }

  rec->skip_counter++;
  if (rec->skip_counter % BEHAVIOR_FRAME_INTERVAL != 0) {
    return;
  }

  if (rec->encoded_frames >= BEHAVIOR_VIDEO_MAX_FRAMES) {
    // Max duration reached, stop and optionally restart
    // We'll stop here; the next person snapshot will trigger a new recording
    rec->person_lost_counter = BEHAVIOR_TIMEOUT_FRAMES + 1;
    return;
  }

  uint8_t *encoded_data = NULL;
  uint32_t encoded_size = 0;

  int ret = TDL_EncodeH264FrameRaw(tdl_handle, image, BEHAVIOR_VENC_CHN, 960,
                                   540, BEHAVIOR_VIDEO_FPS,
                                   512,  // 512 kbps, low bitrate for analysis
                                   BEHAVIOR_VIDEO_FPS,  // GOP = FPS
                                   &encoded_data, &encoded_size);

  if (ret == 0 && encoded_data != NULL && encoded_size > 0) {
    fwrite(encoded_data, 1, encoded_size, rec->h264_file);
    rec->encoded_frames++;
    free(encoded_data);
  } else {
    printf("BehaviorRecorder: H264 encode failed, ret=%d\n", ret);
  }
}

static int create_id_dir(const char *base_dir, int id, char *out_path,
                         size_t out_size) {
  if (ensure_dir(base_dir) != 0) {
    return -1;
  }
  snprintf(out_path, out_size, "%s/%d", base_dir, id);
  return ensure_dir(out_path);
}

static int register_new_face(const TDLFeature *query_feature) {
  int8_t feat_int8[FEATURE_SIZE] = {0};
  if (feature_to_int8(query_feature, feat_int8, FEATURE_SIZE) != 0) {
    return -1;
  }

  pthread_mutex_lock(&g_face_mutex);
  if (g_registered_face_count >= MAX_REGISTERED_FACES) {
    pthread_mutex_unlock(&g_face_mutex);
    return -1;
  }
  int new_id = g_registered_face_count;
  memcpy(g_registered_faces[g_registered_face_count], feat_int8, FEATURE_SIZE);
  g_registered_face_ids[g_registered_face_count] = new_id;
  g_registered_face_count++;
  pthread_mutex_unlock(&g_face_mutex);
  return new_id;
}

static int match_face_online(const TDLFeature *query_feature, float *best_sim) {
  int8_t query_int8[FEATURE_SIZE] = {0};
  if (feature_to_int8(query_feature, query_int8, FEATURE_SIZE) != 0) {
    return -1;
  }

  *best_sim = 0.0f;
  int match_id = -1;
  pthread_mutex_lock(&g_face_mutex);
  for (int i = 0; i < g_registered_face_count; i++) {
    TDLFeature feat_db = {.ptr = g_registered_faces[i],
                          .size = FEATURE_SIZE,
                          .type = TDL_TYPE_INT8};
    TDLFeature feat_query = {
        .ptr = query_int8, .size = FEATURE_SIZE, .type = TDL_TYPE_INT8};
    float sim = 0.0f;
    if (TDL_CaculateSimilarity(feat_db, feat_query, &sim) == 0 &&
        sim > *best_sim) {
      *best_sim = sim;
      match_id = g_registered_face_ids[i];
    }
  }
  pthread_mutex_unlock(&g_face_mutex);
  return (*best_sim >= SIMILARITY_THRESHOLD) ? match_id : -1;
}

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf("  %s -c <config_file> -d <data_dir> [-v <vi_chn>]\n", prog_name);
  printf("Options:\n");
  printf("  -c, --config_file : json config file (required)\n");
  printf("  -d, --data_dir    : data root dir (required)\n");
  printf("  -v, --vi_chn      : optional, default 0\n");
  printf("  -h, --help        : print help\n");
}

void *send_frame_thread(void *args) {
  SEND_FRAME_THREAD_ARG_S *pstArgs = (SEND_FRAME_THREAD_ARG_S *)args;
  uint64_t *channel_frame_id =
      (uint64_t *)malloc(pstArgs->channel_size * sizeof(uint64_t));
  if (channel_frame_id == NULL) {
    return NULL;
  }
  memset(channel_frame_id, 0, pstArgs->channel_size * sizeof(uint64_t));

  while (!to_exit) {
    fd_set rfds;
    struct timeval tv = {0, 0};
    FD_ZERO(&rfds);
    FD_SET(STDIN_FILENO, &rfds);
    int key_pressed = select(STDIN_FILENO + 1, &rfds, NULL, NULL, &tv);
    if (key_pressed > 0 && FD_ISSET(STDIN_FILENO, &rfds)) {
      to_exit = true;
      ExitQueue(&image_queue);
      break;
    }

    for (size_t i = 0; i < pstArgs->channel_size && !to_exit; i++) {
      TDLImage image = GetCameraFrame(pstArgs->tdl_handle, pstArgs->vi_chn);
      if (image == NULL) {
        printf("GetCameraFrame failed\n");
        continue;
      }

      int ret = Image_Enqueue(&image_queue, image);
      if (ret != 0) {
        ReleaseCameraFrame(pstArgs->tdl_handle, pstArgs->vi_chn);
        TDL_DestroyImage(image);
        continue;
      }

      channel_frame_id[i] += 1;
      ret = TDL_APP_SetFrame(pstArgs->tdl_handle, pstArgs->channel_names[i],
                             image, channel_frame_id[i], 3);
      if (ret != 0) {
        printf("TDL_APP_SetFrame failed with %d\n", ret);
        continue;
      }
    }
  }

  free(channel_frame_id);
  return NULL;
}

void *run_tdl_thread(void *args) {
  RUN_TDL_THREAD_ARG_S *pstArgs = (RUN_TDL_THREAD_ARG_S *)args;
  uint64_t counter = 0;
  uint64_t last_counter = 0;
  uint32_t last_time_ms = get_time_in_ms();

  char identity_txt_path[512];
  char identity_dir[512];
  char image_feature_dir[512];
  char registered_feature_dir[512];
  char registered_info_path[512];

  snprintf(identity_txt_path, sizeof(identity_txt_path), "%s/identity_info.txt",
           pstArgs->data_dir);
  snprintf(identity_dir, sizeof(identity_dir), "%s/identity",
           pstArgs->data_dir);
  snprintf(image_feature_dir, sizeof(image_feature_dir), "%s/image_feature",
           pstArgs->data_dir);
  snprintf(registered_feature_dir, sizeof(registered_feature_dir),
           "%s/registered_feature", pstArgs->data_dir);
  snprintf(registered_info_path, sizeof(registered_info_path),
           "%s/registered_info.txt", pstArgs->data_dir);

  ensure_dir(identity_dir);
  ensure_dir(image_feature_dir);
  ensure_dir(registered_feature_dir);

  FILE *identity_fp = fopen(identity_txt_path, "a");
  FILE *registered_info_fp = fopen(registered_info_path, "a");

  while (!to_exit) {
    for (size_t i = 0; i < pstArgs->channel_size && !to_exit; i++) {
      TDLCaptureInfo capture_info = {0};
      int ret = TDL_APP_Capture(pstArgs->tdl_handle, pstArgs->channel_names[i],
                                &capture_info);
      if (ret == 1) {
        continue;
      } else if (ret == 2) {
        to_exit = true;
        ExitQueue(&image_queue);
        break;
      } else if (ret != 0) {
        printf("TDL_APP_Capture failed with %#x\n", ret);
        to_exit = true;
        ExitQueue(&image_queue);
        break;
      }

      counter++;
      int frm_diff = (int)(counter - last_counter);
      if (frm_diff >= 30) {
        uint32_t cur_ts_ms = get_time_in_ms();
        float infer_time = (float)(cur_ts_ms - last_time_ms) / frm_diff;
        float fps = (infer_time > 0.0f) ? (1000.0f / infer_time) : 0.0f;
        last_time_ms = cur_ts_ms;
        last_counter = counter;
        printf("frame:%" PRIu64 ", infer time:%.2f ms, fps:%.2f\n", counter,
               infer_time, fps);
      }

      for (uint32_t j = 0; j < capture_info.snapshot_size; j++) {
        TDLSnapshotInfo *snapshot = &capture_info.snapshot_info[j];
        TDLFeature *feat = NULL;
        if (capture_info.features != NULL) {
          feat = &capture_info.features[j];
          if (feat->ptr == NULL) {
            feat = NULL;
          }
        }

        int match_id = -1;
        float max_sim = 0.0f;
        if (snapshot->object_type == TDL_OBJECT_TYPE_FACE && feat != NULL) {
          if (pstArgs->gallery_feature != NULL &&
              pstArgs->gallery_feature->size > 0) {
            for (uint32_t k = 0; k < pstArgs->gallery_feature->size; k++) {
              float sim = 0.0f;
              if (TDL_CaculateSimilarity(pstArgs->gallery_feature->feature[k],
                                         *feat, &sim) == 0 &&
                  sim > max_sim) {
                max_sim = sim;
                match_id = (int)k;
              }
            }
            if (max_sim < SIMILARITY_THRESHOLD) {
              match_id = -1;
            }
          } else {
            match_id = match_face_online(feat, &max_sim);
            if (match_id == -1) {
              match_id = register_new_face(feat);
              if (match_id >= 0) {
                char ts[32];
                time_t now = time(NULL);
                struct tm *tm_info = localtime(&now);
                strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", tm_info);

                char reg_subdir[512];
                char reg_bin_file[768];
                if (create_id_dir(registered_feature_dir, match_id, reg_subdir,
                                  sizeof(reg_subdir)) == 0) {
                  snprintf(reg_bin_file, sizeof(reg_bin_file),
                           "%s/%s_registeredID_%d.bin", reg_subdir, ts,
                           match_id);
                  save_feature_bin(reg_bin_file, feat);
                }

                if (registered_info_fp != NULL) {
                  fprintf(registered_info_fp, "人员%d %d\n", match_id + 1,
                          match_id);
                  fflush(registered_info_fp);
                }
                printf("New face registered, id=%d, track_id=%" PRIu64 "\n",
                       match_id, snapshot->track_id);
              }
            }
          }

          if (match_id >= 0) {
            ret = TDL_APP_SetFaceID(
                pstArgs->tdl_handle, pstArgs->channel_names[i],
                snapshot->track_id, snapshot->pair_track_id, match_id);
            if (ret != 0) {
              printf("TDL_APP_SetFaceID failed with %d\n", ret);
            }
#if !defined(__CMODEL_CV181X__) && !defined(__CMODEL_CV184X__)
            TDL_MediaAnalysisServer_AddFaceInfo(match_id,
                                                (int)snapshot->track_id);
#endif
          }
        } else if (snapshot->object_type == TDL_OBJECT_TYPE_PERSON) {
          match_id = snapshot->registered_id;
        }

        const char *obj_type_str =
            (snapshot->object_type == TDL_OBJECT_TYPE_FACE) ? "face" : "person";
        if (identity_fp != NULL &&
            (snapshot->object_type == TDL_OBJECT_TYPE_FACE || match_id != -2)) {
          fprintf(identity_fp, "%" PRIu64 ",%s,%.2f,%.2f,%.2f,%.2f,%d\n",
                  capture_info.frame_id, obj_type_str, snapshot->ori_box.x1,
                  snapshot->ori_box.y1, snapshot->ori_box.x2,
                  snapshot->ori_box.y2, match_id);
          fflush(identity_fp);
        }

        if (snapshot->object_image != NULL) {
          char attr_str[256];
          char ts[32];
          char sub_dir[512];
          char jpg_file[768];
          build_attr_str(snapshot, attr_str, sizeof(attr_str));

          time_t now = time(NULL);
          struct tm *tm_info = localtime(&now);
          strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", tm_info);

          if (create_id_dir(identity_dir, match_id, sub_dir, sizeof(sub_dir)) ==
              0) {
            snprintf(jpg_file, sizeof(jpg_file),
                     "%s/%s_frameID_%" PRIu64 "_registeredID_%d_%sID_%" PRIu64
                     "_pairID_%" PRIu64 "_qua_%.3f%s.jpg",
                     sub_dir, ts, snapshot->snapshot_frame_id, match_id,
                     obj_type_str, snapshot->track_id, snapshot->pair_track_id,
                     snapshot->quality, attr_str);
            ret = TDL_EncodeFrame(pstArgs->tdl_handle, snapshot->object_image,
                                  jpg_file, 1);
            if (ret != 0) {
              printf("TDL_EncodeFrame failed with %#x\n", ret);
            } else {
              printf("[SNAPSHOT_SAVED] %s\n", jpg_file);
            }
          }

          if (snapshot->object_type == TDL_OBJECT_TYPE_FACE && feat != NULL) {
            char feat_sub_dir[512];
            char feat_file[768];
            if (create_id_dir(image_feature_dir, match_id, feat_sub_dir,
                              sizeof(feat_sub_dir)) == 0) {
              snprintf(feat_file, sizeof(feat_file),
                       "%s/%s_frameID_%" PRIu64
                       "_registeredID_%d_faceID_%" PRIu64 "_pairID_%" PRIu64
                       "_qua_%.3f%s.bin",
                       feat_sub_dir, ts, snapshot->snapshot_frame_id, match_id,
                       snapshot->track_id, snapshot->pair_track_id,
                       snapshot->quality, attr_str);
              save_feature_bin(feat_file, feat);
            }
          }
        }
      }

      // ---- Behavior video recording: track person presence ----
      {
        bool person_detected = (capture_info.person_meta.size > 0);
        if (person_detected) {
          g_behavior_recorder.person_lost_counter = 0;
          if (!g_behavior_recorder.is_recording) {
            // Find the first person's info to start recording
            int pid = TDL_APP_GetRegisteredID(
                pstArgs->tdl_handle, capture_info.person_meta.info[0].track_id);
            uint64_t tid = capture_info.person_meta.info[0].track_id;
            behavior_recorder_start(pid, tid, pstArgs->data_dir);
          }
        } else if (g_behavior_recorder.is_recording) {
          g_behavior_recorder.person_lost_counter++;
          if (g_behavior_recorder.person_lost_counter >=
              BEHAVIOR_TIMEOUT_FRAMES) {
            printf("BehaviorRecorder: person lost timeout, stopping\n");
            behavior_recorder_stop_and_submit(pstArgs->data_dir);
          }
        }
        // Also check max frames duration (from encode function side)
        if (g_behavior_recorder.is_recording &&
            g_behavior_recorder.skip_counter / BEHAVIOR_FRAME_INTERVAL >=
                BEHAVIOR_VIDEO_MAX_FRAMES) {
          printf("BehaviorRecorder: max duration reached, stopping\n");
          behavior_recorder_stop_and_submit(pstArgs->data_dir);
        }
      }

      if (capture_info.image) {
        // Encode to H.264 for behavior analysis (every 5th frame)
        behavior_recorder_encode_frame(pstArgs->tdl_handle, capture_info.image);

        uint8_t *encoded_data = NULL;
        uint32_t encoded_size = 0;
        ret = TDL_EncodeFrameRaw(pstArgs->tdl_handle, capture_info.image, 1,
                                 &encoded_data, &encoded_size);
        if (ret == 0 && encoded_data != NULL && encoded_size > 0) {
#if !defined(__CMODEL_CV181X__) && !defined(__CMODEL_CV184X__)
          int channel_id = atoi(pstArgs->channel_names[i]);
          uint64_t timestamp = (uint64_t)get_time_in_ms();
          RegisteredNameEntry reg_name_entries[512];
          int reg_name_count = load_registered_name_map(registered_info_path,
                                                        reg_name_entries, 512);

          char metadata_json[8192] = {0};
          int pos =
              snprintf(metadata_json, sizeof(metadata_json),
                       "{\"width\":%u,\"height\":%u,\"source_width\":%u,"
                       "\"source_height\":%u,\"faces\":[",
                       capture_info.frame_width, capture_info.frame_height,
                       capture_info.source_width, capture_info.source_height);

          for (uint32_t j = 0; j < capture_info.face_meta.size; j++) {
            int registered_id = TDL_APP_GetRegisteredID(
                pstArgs->tdl_handle, capture_info.face_meta.info[j].track_id);
            const char *name = find_registered_name(
                reg_name_entries, reg_name_count, registered_id);
            char reg_label[256];
            char reg_label_escaped[512];
            if (registered_id != -1 && name != NULL) {
              snprintf(reg_label, sizeof(reg_label), "%s", name);
            } else {
              snprintf(reg_label, sizeof(reg_label), "%d", registered_id);
            }
            json_escape_copy(reg_label, reg_label_escaped,
                             sizeof(reg_label_escaped));
            pos += snprintf(
                metadata_json + pos, sizeof(metadata_json) - pos,
                "{\"x1\":%.2f,\"y1\":%.2f,\"x2\":%.2f,\"y2\":%.2f,"
                "\"track_id\":%" PRIu64 ",\"registered_id\":\"%s\"}%s",
                capture_info.face_meta.info[j].box.x1,
                capture_info.face_meta.info[j].box.y1,
                capture_info.face_meta.info[j].box.x2,
                capture_info.face_meta.info[j].box.y2,
                capture_info.face_meta.info[j].track_id, reg_label_escaped,
                (j < capture_info.face_meta.size - 1) ? "," : "");
          }
          pos += snprintf(metadata_json + pos, sizeof(metadata_json) - pos,
                          "],\"persons\":[");

          for (uint32_t j = 0; j < capture_info.person_meta.size; j++) {
            int registered_id = TDL_APP_GetRegisteredID(
                pstArgs->tdl_handle, capture_info.person_meta.info[j].track_id);
            if (registered_id != -1) {
              move_unknown_person_images(
                  identity_dir, registered_id,
                  capture_info.person_meta.info[j].track_id);
            }
            const char *name = find_registered_name(
                reg_name_entries, reg_name_count, registered_id);
            char reg_label[256];
            char reg_label_escaped[512];
            if (registered_id != -1 && name != NULL) {
              snprintf(reg_label, sizeof(reg_label), "%s", name);
            } else {
              snprintf(reg_label, sizeof(reg_label), "%d", registered_id);
            }
            json_escape_copy(reg_label, reg_label_escaped,
                             sizeof(reg_label_escaped));
            pos += snprintf(
                metadata_json + pos, sizeof(metadata_json) - pos,
                "{\"x1\":%.2f,\"y1\":%.2f,\"x2\":%.2f,\"y2\":%.2f,"
                "\"track_id\":%" PRIu64 ",\"registered_id\":\"%s\"}%s",
                capture_info.person_meta.info[j].box.x1,
                capture_info.person_meta.info[j].box.y1,
                capture_info.person_meta.info[j].box.x2,
                capture_info.person_meta.info[j].box.y2,
                capture_info.person_meta.info[j].track_id, reg_label_escaped,
                (j < capture_info.person_meta.size - 1) ? "," : "");
          }
          pos +=
              snprintf(metadata_json + pos, sizeof(metadata_json) - pos, "]}");

          int send_ret = TDL_MediaAnalysisServer_SendImage(
              encoded_data, encoded_size, timestamp, channel_id,
              capture_info.frame_id, metadata_json);
          if (send_ret != 0) {
            printf("TDL_MediaAnalysisServer_SendImage failed with %d\n",
                   send_ret);
          }
#endif
        }
        free(encoded_data);
      }

      TDL_ReleaseCaptureInfo(&capture_info);
      ReleaseCameraFrame(pstArgs->tdl_handle, pstArgs->vi_chn);
      TDLImage img = Image_Dequeue(&image_queue);
      if (img) {
        TDL_DestroyImage(img);
      }
    }
  }

  // Stop any ongoing behavior recording before exit
  if (g_behavior_recorder.is_recording) {
    behavior_recorder_stop_and_submit(pstArgs->data_dir);
  }

  if (identity_fp) fclose(identity_fp);
  if (registered_info_fp) fclose(registered_info_fp);
  return NULL;
}

int main(int argc, char *argv[]) {
  char *config_file = NULL;
  char data_dir[512] = {0};
  int chn = 0;
  int ret = 0;
  char gallery_dir[512] = {0};

  struct option long_options[] = {
      {"config_file", required_argument, 0, 'c'},
      {"data_dir", required_argument, 0,
       'd'},  // 和config_file中的data_path相同（免去解析config_file）
      {"vi_chn", required_argument, 0, 'v'},
      {"help", no_argument, 0, 'h'},
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "c:d:v:h", long_options, NULL)) != -1) {
    switch (opt) {
      case 'c':
        config_file = optarg;
        break;
      case 'd':
        strncpy(data_dir, optarg, sizeof(data_dir) - 1);
        break;
      case 'v':
        chn = atoi(optarg);
        break;
      case 'h':
        print_usage(argv[0]);
        return 0;
      default:
        print_usage(argv[0]);
        return -1;
    }
  }

  if (config_file == NULL || data_dir[0] == '\0') {
    fprintf(stderr, "Error: config_file and data_dir are required\n");
    print_usage(argv[0]);
    return -1;
  }

  signal(SIGINT, handle_signal);
  signal(SIGTERM, handle_signal);

  printf("Running with:\n");
  printf("  config_file: %s\n", config_file);
  printf("  data_dir: %s\n", data_dir);
  printf("  vi_chn: %d\n", chn);

  if (ensure_dir(data_dir) != 0) {
    fprintf(stderr, "Error: create data_dir failed: %s\n", data_dir);
    return -1;
  }
  snprintf(gallery_dir, sizeof(gallery_dir), "%s/registered_feature", data_dir);
  ensure_dir(gallery_dir);
  InitQueue(&image_queue);

  TDLHandle tdl_handle = TDL_CreateHandle(0);
  if (tdl_handle == NULL) {
    ret = -1;
    goto exit0;
  }

  ret = InitCamera(tdl_handle, VI_WIDTH, VI_HEIGHT, IMAGE_YUV420SP_UV, 3);
  if (ret != 0) {
    printf("InitCamera failed with %#x\n", ret);
    goto exit1;
  }

#if !defined(__CMODEL_CV181X__) && !defined(__CMODEL_CV184X__)
  ret = TDL_MediaAnalysisServer_Init(config_file);
  if (ret != 0) {
    printf("TDL_MediaAnalysisServer_Init failed with %#x\n", ret);
    goto exit2;
  }
#endif

  char **channel_names = NULL;
  uint8_t channel_size = 0;
  ret = TDL_APP_Init(tdl_handle, "face_pet_capture", config_file,
                     &channel_names, &channel_size, false);
  if (ret != 0) {
    printf("TDL_APP_Init failed with %#x\n", ret);
    goto exit3;
  }

  TDLFeatureInfo *gallery_feature = NULL;
  gallery_feature = (TDLFeatureInfo *)malloc(sizeof(TDLFeatureInfo));
  if (gallery_feature != NULL) {
    memset(gallery_feature, 0, sizeof(TDLFeatureInfo));
    if (TDL_GetGalleryFeature(gallery_dir, gallery_feature, FEATURE_SIZE) !=
        0) {
      free(gallery_feature);
      gallery_feature = NULL;
    }
  }

  struct termios oldt, newt;
  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  printf("Press any key to exit...\n");

  pthread_t frame_thread, tdl_thread;
  SEND_FRAME_THREAD_ARG_S frame_args = {.tdl_handle = tdl_handle,
                                        .vi_chn = chn,
                                        .channel_size = channel_size,
                                        .channel_names = channel_names};
  RUN_TDL_THREAD_ARG_S tdl_args = {.tdl_handle = tdl_handle,
                                   .vi_chn = chn,
                                   .channel_size = channel_size,
                                   .channel_names = channel_names,
                                   .gallery_feature = gallery_feature};
  strncpy(tdl_args.data_dir, data_dir, sizeof(tdl_args.data_dir) - 1);

  pthread_create(&frame_thread, NULL, send_frame_thread, &frame_args);
  pthread_create(&tdl_thread, NULL, run_tdl_thread, &tdl_args);

  pthread_join(frame_thread, NULL);
  to_exit = true;
  ExitQueue(&image_queue);
  pthread_join(tdl_thread, NULL);

  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);

  if (gallery_feature != NULL) {
    for (uint32_t i = 0; i < gallery_feature->size; i++) {
      TDL_ReleaseFeatureMeta(&gallery_feature->feature[i]);
    }
    free(gallery_feature);
  }
  for (uint32_t i = 0; i < channel_size; i++) {
    free(channel_names[i]);
  }
  free(channel_names);

exit3:
#if !defined(__CMODEL_CV181X__) && !defined(__CMODEL_CV184X__)
  TDL_MediaAnalysisServer_Stop();
#endif
exit2:
  DestoryCamera(tdl_handle);
exit1:
  TDL_DestroyHandle(tdl_handle);
exit0:
  DestroyQueue(&image_queue);
  return ret;
}
