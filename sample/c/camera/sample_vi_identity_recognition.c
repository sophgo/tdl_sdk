#include <getopt.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#include "cvi_comm_video.h"
#include "cvi_vi.h"
#include "meta_visualize.h"
#include "pthread_utils.h"
#include "sample_utils.h"
#include "tdl_sdk.h"

#ifndef __CV184X__
#define ENABLE_RTSP
#endif

#define VI_WIDTH 1920
#define VI_HEIGHT 1080
#define FEATURE_SIZE 256
#define SIMILARITY_THRESHOLD 0.45
#define MAX_REGISTERED_FACES 100
#define MAX_TRACKS 200

static const char *emotionStr[] = {"Anger",   "Disgust", "Fear",    "Happy",
                                   "Neutral", "Sad",     "Surprise"};

// Global variables for online face recognition
static float g_registered_faces[MAX_REGISTERED_FACES][FEATURE_SIZE];
static int g_registered_face_ids[MAX_REGISTERED_FACES];
static int g_registered_face_count = 0;
static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;

// Map track_id to face_id
typedef struct {
  uint64_t track_id;
  int face_id;
  uint64_t last_update_time;
} TrackIDInfo;

static TrackIDInfo g_track_map[MAX_TRACKS];
static pthread_mutex_t g_track_map_mutex = PTHREAD_MUTEX_INITIALIZER;

static volatile bool to_exit = false;
static ImageQueue image_queue;

static uint32_t get_time_in_ms() {
  struct timeval tv;
  if (gettimeofday(&tv, NULL) < 0) {
    return 0;
  }
  return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

typedef struct {
  TDLHandle tdl_handle;
  int vi_chn;
  uint8_t channel_size;
  char **channel_names;
} SEND_FRAME_THREAD_ARG_S;

typedef struct {
  int vi_chn;
  TDLHandle tdl_handle;
  uint8_t channel_size;
  char **channel_names;
  const char *output_dir;
  TDLFeatureInfo *gallery_feature;
} RUN_TDL_THREAD_ARG_S;

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf("  %s -c <config_file> -g <gallery_dir> -o <output_dir> -v <vi_chn>\n",
         prog_name);
  printf(
      "  %s --config_file <path> --gallery_dir <dir> --output_dir <dir> "
      "--vi_chn <int> \n",
      prog_name);
  printf("Options:\n");
  printf(
      "  -c, --config_file : json config file\n"
      "  -g, --gallery_dir : (optional) the face feature directory\n"
      "  -o, --output_dir : output dir to save snapshot and identity info\n"
      "  -v, --vi_chn : optional , defult 0\n");
}

void *send_frame_thread(void *args) {
  printf("Enter send frame thread\n");
  SEND_FRAME_THREAD_ARG_S *pstArgs = (SEND_FRAME_THREAD_ARG_S *)args;

  uint64_t *channel_frame_id =
      (uint64_t *)malloc(pstArgs->channel_size * sizeof(uint64_t));
  int ret = 0;
  memset(channel_frame_id, 0, pstArgs->channel_size * sizeof(uint64_t));

  while (to_exit == false) {
    // Check keyboard input
    fd_set rfds;
    struct timeval tv = {0, 0};
    FD_ZERO(&rfds);
    FD_SET(STDIN_FILENO, &rfds);
    int key_pressed = select(STDIN_FILENO + 1, &rfds, NULL, NULL, &tv);
    if (key_pressed > 0 && FD_ISSET(STDIN_FILENO, &rfds)) {
      to_exit = true;
      break;
    }

    for (size_t i = 0; i < pstArgs->channel_size; i++) {
      TDLImage image = NULL;
      image = GetCameraFrame(pstArgs->tdl_handle, pstArgs->vi_chn);
      if (image == NULL) {
        continue;
      }

      if (image == NULL) {
        printf("GetCameraFrame falied\n");
        continue;
      }

      ret = Image_Enqueue(&image_queue, image);
      if (ret != 0) {
        printf("Image_Enqueue falied\n");
        ReleaseCameraFrame(pstArgs->tdl_handle, pstArgs->vi_chn);
        TDL_DestroyImage(image);
        continue;
      }
      channel_frame_id[i] += 1;

      // Use TDL_APP_SetFrame
      ret = TDL_APP_SetFrame(pstArgs->tdl_handle, pstArgs->channel_names[i],
                             image, channel_frame_id[i]++, 3);  // buffer size 3
      if (ret != 0) {
        printf("TDL_APP_SetFrame failed with %d\n", ret);
        continue;
      }
    }
  }

  free(channel_frame_id);
  printf("Exit send frame thread\n");
  return NULL;
}

float compute_similarity_mixed(const float *feat1_float,
                               const TDLFeature *feat2) {
  // Convert feat2 to float for calculation or vice versa.
  // Assuming feat1_float is already normalized or we do dot product.
  // Let's implement generic matching.
  float dot = 0, n1 = 0, n2 = 0;

  // We'll just cast to float for simplicity
  for (int i = 0; i < FEATURE_SIZE; i++) {
    float v1 = feat1_float[i];
    float v2 = 0;
    if (feat2->type == TDL_TYPE_INT8) {
      v2 = (float)((int8_t *)feat2->ptr)[i];
    } else {
      v2 = ((float *)feat2->ptr)[i];  // Assuming float
    }

    dot += v1 * v2;
    n1 += v1 * v1;
    n2 += v2 * v2;
  }
  if (n1 == 0 || n2 == 0) return 0.0f;
  return dot / (sqrt(n1) * sqrt(n2));
}

int match_face_online(const TDLFeature *query_feature) {
  int match_id = -1;
  float max_similarity = 0.0f;

  pthread_mutex_lock(&g_mutex);
  for (int i = 0; i < g_registered_face_count; ++i) {
    float sim = compute_similarity_mixed(g_registered_faces[i], query_feature);
    if (sim > max_similarity) {
      max_similarity = sim;
      match_id = g_registered_face_ids[i];
    }
  }
  pthread_mutex_unlock(&g_mutex);

  if (max_similarity >= SIMILARITY_THRESHOLD) {
    return match_id;
  }
  return -1;
}

void register_new_face(const TDLFeature *feature) {
  pthread_mutex_lock(&g_mutex);
  if (g_registered_face_count < MAX_REGISTERED_FACES) {
    for (int i = 0; i < FEATURE_SIZE; i++) {
      if (feature->type == TDL_TYPE_INT8)
        g_registered_faces[g_registered_face_count][i] =
            (float)((int8_t *)feature->ptr)[i];
      else
        g_registered_faces[g_registered_face_count][i] =
            ((float *)feature->ptr)[i];
    }
    g_registered_face_ids[g_registered_face_count] = g_registered_face_count;
    g_registered_face_count++;
  }
  pthread_mutex_unlock(&g_mutex);
}

void *run_tdl_thread(void *args) {
  printf("Enter run tdl thread\n");
  RUN_TDL_THREAD_ARG_S *pstArgs = (RUN_TDL_THREAD_ARG_S *)args;

  uint64_t counter = 0;
  uint64_t last_counter = 0;
  uint32_t last_time_ms = get_time_in_ms();

#ifdef ENABLE_RTSP

  VIDEO_FRAME_INFO_S *frame = NULL;

  // 初始化RTSP参数
  RtspContext rtsp_context = {0};
  rtsp_context.chn = 0;
  rtsp_context.pay_load_type = PT_H264;
  rtsp_context.frame_width = VI_WIDTH;
  rtsp_context.frame_height = VI_HEIGHT;
#endif

  // Create output file
  char identity_txt_path[512];
  snprintf(identity_txt_path, sizeof(identity_txt_path), "%s/identity_info.txt",
           pstArgs->output_dir);
  char identity_dir[512];
  snprintf(identity_dir, sizeof(identity_dir), "%s/identity",
           pstArgs->output_dir);
  mkdir(identity_dir, 0777);

  while (to_exit == false) {
    counter++;
    int frm_diff = counter - last_counter;
    if (frm_diff > 30) {
      uint32_t cur_ts_ms = get_time_in_ms();
      float infer_time = (float)(cur_ts_ms - last_time_ms) / frm_diff;
      float fps = 1000.0 / infer_time;

      last_time_ms = cur_ts_ms;
      last_counter = counter;
      printf(
          "+++++++++++++++++++++++++++++++++++ frame:%d, infer time:%.2f, "
          "fps:%.2f\n",
          (int)counter, infer_time, fps);
    }

    TDLCaptureInfo capture_info = {0};
    int ret = TDL_APP_Capture(pstArgs->tdl_handle, pstArgs->channel_names[0],
                              &capture_info);
    if (ret == 1) {  // Try again
      continue;
    } else if (ret != 0) {
      printf("TDL_APP_Capture failed: %d\n", ret);
      goto exit0;
    }

    // Open file for appending
    FILE *fp = fopen(identity_txt_path, "a");

    // Process Snapshots for Recognition
    for (uint32_t i = 0; i < capture_info.snapshot_size; i++) {
      TDLSnapshotInfo *snapshot = &capture_info.snapshot_info[i];
      int match_id = -1;
      if (snapshot->object_type == TDL_OBJECT_TYPE_FACE) {
        // Check features
        if (capture_info.features) {  // Assuming features array is valid
          TDLFeature *feat = &capture_info.features[i];  // 1:1 mapping
          if (feat->ptr == NULL) continue;

          if (pstArgs->gallery_feature && pstArgs->gallery_feature->size > 0) {
            // Gallery match
            float max_sim = 0;
            for (uint32_t k = 0; k < pstArgs->gallery_feature->size; k++) {
              float sim = 0;
              TDL_CaculateSimilarity(pstArgs->gallery_feature->feature[k],
                                     *feat, &sim);
              if (sim > max_sim) {
                max_sim = sim;
                // Assuming gallery feature index corresponds to some ID?
                // The C sample just uses index.
                // We can use index as ID.
                match_id = k;
              }
            }
            if (max_sim < SIMILARITY_THRESHOLD) {
              match_id = -1;
            } else {
              printf("Gallery match with ID %d, similarity: %.2f\n", match_id,
                     max_sim);
            }
          } else {
            // Online match
            match_id = match_face_online(feat);
            if (match_id == -1) {
              register_new_face(feat);
              match_id = g_registered_face_count - 1;
              printf("Register new face with ID %d\n", match_id);
            } else {
              printf("Online match with ID %d\n", match_id);
            }
          }

          if (match_id >= 0) {
            // Set face ID for ReID tracking
            ret = TDL_APP_SetFaceID(
                pstArgs->tdl_handle, pstArgs->channel_names[0],
                snapshot->track_id, snapshot->pair_track_id, match_id);
            printf("TDL_APP_SetFaceID: person_id %" PRIu64
                   " registered_id:%d\n",
                   snapshot->pair_track_id, match_id);
            if (ret != 0) {
              printf("TDL_APP_SetFaceID failed: %d\n", ret);
            }
          }
        }

      } else {
        match_id = snapshot->registered_id;
      }

      const char *object_type_str =
          (snapshot->object_type == TDL_OBJECT_TYPE_FACE) ? "face" : "person";

      // Log to TXT
      if (fp) {
        if (snapshot->object_type == TDL_OBJECT_TYPE_FACE || match_id != -2)
          fprintf(fp, "%lu, %s,%.2f, %.2f, %.2f, %.2f,%d\n",
                  capture_info.frame_id, object_type_str, snapshot->ori_box.x1,
                  snapshot->ori_box.y1, snapshot->ori_box.x2,
                  snapshot->ori_box.y2, match_id);
      }

      if (capture_info.snapshot_info[i].object_image) {  // save snapshot
        char filename[512];
        char attr_str[256] = "";
        char timestamp_str[32];
        time_t now = time(NULL);
        struct tm *tm_info = localtime(&now);
        strftime(timestamp_str, sizeof(timestamp_str), "%Y%m%d_%H%M%S",
                 tm_info);

        // 为人脸添加属性信息
        if (snapshot->object_type == TDL_OBJECT_TYPE_FACE) {
          snprintf(attr_str, sizeof(attr_str),
                   "_male[%d]_glass[%d]_age[%d]_emotion[%s]", snapshot->male,
                   snapshot->glass, snapshot->age,
                   emotionStr[snapshot->emotion]);
        }

        // 创建以match_id命名的子目录
        char sub_dir[512];
        snprintf(sub_dir, sizeof(sub_dir), "%s/%d", identity_dir, match_id);
        mkdir(sub_dir, 0777);

        sprintf(filename,
                "%s/%s_frameID_%" PRIu64
                "_registeredID_%d"
                "_%sID_%" PRIu64 "_pairID_%" PRIu64 "_qua_%.3f%s.jpg",
                sub_dir, timestamp_str, snapshot->snapshot_frame_id, match_id,
                object_type_str, snapshot->track_id, snapshot->pair_track_id,
                snapshot->quality, attr_str);

        ret = TDL_EncodeFrame(pstArgs->tdl_handle,
                              capture_info.snapshot_info[i].object_image,
                              filename, 1);
        if (ret != 0) {
          printf("TDL_EncodeFrame failed with %#x!\n", ret);
          continue;
        }
      }
    }

#ifdef ENABLE_RTSP
    TDLImage image = capture_info.image;
    TDL_WrapImage(image, &frame);

    TDLBrush brush = {0};
    brush.size = 5;
    brush.color.r = 0;
    brush.color.g = 255;
    brush.color.b = 0;
    for (int i = 0; i < capture_info.person_meta.size; i++) {
      TDLObjectInfo *obj_info = &capture_info.person_meta.info[i];
      snprintf(obj_info->name, sizeof(obj_info->name), "id:%d",
               obj_info->track_id);
    }
    DrawObjRect(&capture_info.person_meta, frame, true, brush);

    brush.color.r = 255;
    brush.color.g = 0;
    brush.color.b = 0;
    for (int i = 0; i < capture_info.pet_meta.size; i++) {
      TDLObjectInfo *pet_info = &capture_info.pet_meta.info[i];
      snprintf(pet_info->name, sizeof(pet_info->name), "score:%.2f",
               pet_info->score);
    }
    DrawObjRect(&capture_info.pet_meta, frame, true, brush);

    brush.color.r = 0;
    brush.color.g = 0;
    brush.color.b = 255;
    for (int i = 0; i < capture_info.face_meta.size; i++) {
      TDLFaceInfo *face_info = &capture_info.face_meta.info[i];
      snprintf(face_info->name, sizeof(face_info->name), "id:%d",
               face_info->track_id);
    }
    DrawFaceRect(&capture_info.face_meta, frame, true, brush);

    brush.color.g = 255;
    brush.color.b = 0;
    char text[128] = {0};
    snprintf(text, sizeof(text), "frame id:%d", capture_info.frame_id);
    ObjectWriteText(text, 50, 50, frame, brush);

    ret = SendFrameRTSP(frame, &rtsp_context);
    if (ret != 0) {
      printf("SendFrameRTSP failed with %#x!\n", ret);
      continue;
    }
#endif

    if (fp) fclose(fp);

    TDL_ReleaseCaptureInfo(&capture_info);
    ReleaseCameraFrame(pstArgs->tdl_handle, pstArgs->vi_chn);
    TDLImage img = Image_Dequeue(&image_queue);
    if (img) {
      TDL_DestroyImage(img);
    }
    if (to_exit) {
      break;
    }
  }

exit0:
  DestoryCamera(pstArgs->tdl_handle);
  TDL_DestroyHandle(pstArgs->tdl_handle);
  return NULL;
}

int main(int argc, char *argv[]) {
  const char *config_file = NULL;
  const char *gallery_dir = NULL;
  const char *output_dir = NULL;
  int vi_chn = 0;

  struct option long_options[] = {
      {"config_file", required_argument, 0, 'c'},
      {"gallery_dir", required_argument, 0, 'g'},
      {"output_dir", required_argument, 0, 'o'},
      {"vi_chn", no_argument, 0, 'v'},
      {"help", no_argument, 0, 'h'},
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "c:g:o:v:h", long_options, NULL)) !=
         -1) {
    switch (opt) {
      case 'c':
        config_file = optarg;
        break;
      case 'g':
        gallery_dir = optarg;
        break;
      case 'o':
        output_dir = optarg;
        break;
      case 'v':
        vi_chn = atoi(optarg);
        break;
      case 'h':
        print_usage(argv[0]);
        return 0;
      default:
        print_usage(argv[0]);
        return -1;
    }
  }

  if (config_file == NULL || output_dir == NULL) {
    fprintf(stderr, "Error: config_file and output_dir are required\n");
    print_usage(argv[0]);
    return -1;
  }

  // Create output directory
  mkdir(output_dir, 0755);

  InitQueue(&image_queue);
  // Create Handle
  TDLHandle tdl_handle = TDL_CreateHandle(0);

  // Get Channel Names
  char **channel_names = NULL;
  uint8_t channel_size = 0;

  int ret = InitCamera(tdl_handle, VI_WIDTH, VI_HEIGHT, IMAGE_YUV420SP_UV, 3);
  if (ret != 0) {
    printf("InitCamera %#x!\n", ret);
    return ret;
  }

  // Init APP
  ret = TDL_APP_Init(tdl_handle, "face_pet_capture", config_file,
                     &channel_names, &channel_size, false);
  if (ret != 0) {
    printf("TDL_APP_Init failed\n");
    return -1;
  }

  // Load Gallery if needed
  TDLFeatureInfo *gallery_feature = NULL;
  if (gallery_dir) {
    gallery_feature = (TDLFeatureInfo *)malloc(sizeof(TDLFeatureInfo));
    memset(gallery_feature, 0, sizeof(TDLFeatureInfo));
    ret = TDL_GetGalleryFeature(gallery_dir, gallery_feature, FEATURE_SIZE);
    if (ret != 0) {
      printf("Failed to load gallery from %s\n", gallery_dir);
      free(gallery_feature);
      gallery_feature = NULL;
    }
  }

  // 设置终端为非规范模式
  struct termios oldt, newt;
  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);

  printf("按任意键退出...\n");

  // Start threads
  pthread_t send_thread, run_thread;
  SEND_FRAME_THREAD_ARG_S send_args = {tdl_handle, vi_chn, channel_size,
                                       channel_names};
  RUN_TDL_THREAD_ARG_S run_args = {vi_chn,        tdl_handle, channel_size,
                                   channel_names, output_dir, gallery_feature};

  pthread_create(&send_thread, NULL, send_frame_thread, &send_args);
  pthread_create(&run_thread, NULL, run_tdl_thread, &run_args);

  pthread_join(send_thread, NULL);
  pthread_join(run_thread, NULL);

  // 恢复终端设置
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);

exit1:
  for (int i = 0; i < channel_size; i++) {
    free(channel_names[i]);
  }
  free(channel_names);

exit0:
  DestroyQueue(&image_queue);
  if (gallery_feature) {
    for (int i = 0; i < gallery_feature->size; i++) {
      TDL_ReleaseFeatureMeta(&gallery_feature->feature[i]);
    }
    free(gallery_feature);
  }
  DestoryCamera(tdl_handle);
  TDL_DestroyHandle(tdl_handle);

  return 0;
}
