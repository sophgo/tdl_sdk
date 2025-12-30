#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/time.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#include <pthread.h>
#include "cvi_comm_video.h"
#include "cvi_vi.h"
#include "meta_visualize.h"
#include "pthread_utils.h"
#include "sample_utils.h"
#include "tdl_sdk.h"

#ifndef __CV184X__
#define ENABLE_RTSP
#endif

#define VI_WIDTH 960
#define VI_HEIGHT 540
#define FEATURE_SIZE 256

static volatile bool to_exit = false;
static ImageQueue image_queue;
static const char *emotionStr[] = {"Anger",   "Disgust", "Fear",    "Happy",
                                   "Neutral", "Sad",     "Surprise"};

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
      "  -g, --gallery_dir : the face feature directory contains feature files "
      "named 0.bin, 1.bin, 2.bin...(no more than 100)\n"
      "  -o, --output_dir : output dir to save snapshot\n"
      "  -v, --vi_chn : optional , defult 0\n");
}

void *send_frame_thread(void *args) {
  printf("Enter send frame thread\n");
  SEND_FRAME_THREAD_ARG_S *pstArgs = (SEND_FRAME_THREAD_ARG_S *)args;

  uint64_t *channel_frame_id = malloc(pstArgs->channel_size * sizeof(uint64_t));
  int ret = 0;
  memset(channel_frame_id, 0, pstArgs->channel_size * sizeof(uint64_t));

  while (to_exit == false) {
    // 检查键盘输入
    fd_set rfds;
    struct timeval tv = {0, 0};
    FD_ZERO(&rfds);
    FD_SET(STDIN_FILENO, &rfds);
    int key_pressed = select(STDIN_FILENO + 1, &rfds, NULL, NULL, &tv);
    if (key_pressed > 0 && FD_ISSET(STDIN_FILENO, &rfds)) {
      to_exit = true;
      break;  // 有键盘输入，退出循环
    }

    for (size_t i = 0; i < pstArgs->channel_size; i++) {
      TDLImage image = NULL;

      image =
          GetCameraFrame(pstArgs->tdl_handle,
                         pstArgs->vi_chn);  // if channel_size > 1, image should
                                            // be taken from different vi_chn
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

      ret = TDL_APP_SetFrame(pstArgs->tdl_handle, pstArgs->channel_names[i],
                             image, channel_frame_id[i], 3);
      if (ret != 0) {
        printf("TDL_APP_SetFrame failed with %d\n", ret);
        continue;
      }
    }
  }

  free(channel_frame_id);
}

void *run_tdl_thread(void *args) {
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

  while (to_exit == false) {
    for (size_t i = 0; i < pstArgs->channel_size; i++) {
      TDLCaptureInfo capture_info = {0};

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

      int ret = TDL_APP_Capture(pstArgs->tdl_handle, pstArgs->channel_names[i],
                                &capture_info);

      if (ret == 1) {
        continue;
      } else if (ret == 2) {
        to_exit = true;
        break;
      } else if (ret != 0) {
        printf("TDL_APP_Capture failed with %#x!\n", ret);
        goto exit0;
      }

      if (frm_diff > 30) {
        printf("detect person size: %d, pet size: %d\n",
               capture_info.person_meta.size, capture_info.pet_meta.size);
      }

      for (uint32_t j = 0; j < capture_info.snapshot_size; j++) {
        printf("snapshot[%d]: male:%d,glass:%d,age:%d,emotion:%s\n", j,
               capture_info.snapshot_info[j].male,
               capture_info.snapshot_info[j].glass,
               capture_info.snapshot_info[j].age,
               emotionStr[capture_info.snapshot_info[j].emotion]);

        if (capture_info.snapshot_info[j].object_image) {  // save snapshot
          char filename[512];
          sprintf(filename,
                  "%s/%" PRIu64 "_face_%" PRIu64
                  "_qua_%.3f_male[%d]_glass[%d]_age[%d]_emotion[%s].jpg",
                  pstArgs->output_dir,
                  capture_info.snapshot_info[j].snapshot_frame_id,
                  capture_info.snapshot_info[j].track_id,
                  capture_info.snapshot_info[j].quality,
                  capture_info.snapshot_info[j].male,
                  capture_info.snapshot_info[j].glass,
                  capture_info.snapshot_info[j].age,
                  emotionStr[capture_info.snapshot_info[j].emotion]);

          ret = TDL_EncodeFrame(pstArgs->tdl_handle,
                                capture_info.snapshot_info[j].object_image,
                                filename, 1);
          if (ret != 0) {
            printf("TDL_EncodeFrame failed with %#x!\n", ret);
            continue;
          }

          if (capture_info.snapshot_info[j]
                  .encoded_full_image) {  // save full image
            sprintf(filename,
                    "%s/%" PRIu64 "_face_%" PRIu64
                    "_box[%.2f,%.2f,%.2f,%.2f]_full_image.jpg",
                    pstArgs->output_dir,
                    capture_info.snapshot_info[j].snapshot_frame_id,
                    capture_info.snapshot_info[j].track_id,
                    capture_info.snapshot_info[j].ori_box.x1,
                    capture_info.snapshot_info[j].ori_box.y1,
                    capture_info.snapshot_info[j].ori_box.x2,
                    capture_info.snapshot_info[j].ori_box.y2);

            FILE *f;
            f = fopen(filename, "wb");
            if (!f) {
              printf("open file fail: %s\n", filename);
            } else {
              fwrite(capture_info.snapshot_info[j].encoded_full_image, 1,
                     capture_info.snapshot_info[j].full_length, f);
            }
            fclose(f);
          }
        }

        printf("to do TDL_CaculateSimilarity\n");

        float max_similarity = 0;
        float similarity = 0;
        uint8_t top_index;
        for (uint32_t k = 0; k < pstArgs->gallery_feature->size; k++) {
          TDL_CaculateSimilarity(pstArgs->gallery_feature->feature[k],
                                 capture_info.features[j], &similarity);
          if (similarity > max_similarity) {
            max_similarity = similarity;
            top_index = k;
          }
        }

        if (max_similarity > 0.4) {
          printf("match feature %d.bin, track id: %ld, similarity: %.2f\n",
                 top_index, capture_info.snapshot_info[i].track_id,
                 max_similarity);
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

      TDL_ReleaseCaptureInfo(&capture_info);
      ReleaseCameraFrame(pstArgs->tdl_handle, pstArgs->vi_chn);
      TDLImage img = Image_Dequeue(&image_queue);
      if (img) {
        TDL_DestroyImage(img);
      }
    }

    if (to_exit) {
      break;
    }
  }

  return NULL;

exit0:
  DestoryCamera(pstArgs->tdl_handle);
  TDL_DestroyHandle(pstArgs->tdl_handle);
}

int main(int argc, char *argv[]) {
  char *config_file = NULL;  // sample/config/face_pet_cap_app_vi.json
  char *gallery_dir = NULL;
  char *output_dir = NULL;
  char *vi_chn = NULL;
  int chn = 0;

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
        vi_chn = optarg;
        break;
      case 'h':
        print_usage(argv[0]);
        return 0;
      case '?':
        print_usage(argv[0]);
        return -1;
      default:
        print_usage(argv[0]);
        return -1;
    }
  }

  if (!config_file) {
    fprintf(stderr, "Error: config_file are required\n");
    print_usage(argv[0]);
    return -1;
  }

  if (vi_chn) {
    chn = atoi(vi_chn);
  }

  printf("Running with:\n");
  printf("  config_file:    %s\n",
         config_file);  // sample/config/face_pet_cap_app_vi.json
  printf("  gallery_dir:   %s\n", gallery_dir);
  printf("  output_dir:  %s\n", output_dir);
  printf("  vi_chn:        %d\n", chn);

  InitQueue(&image_queue);

  TDLFeatureInfo gallery_feature = {0};
  int ret = TDL_GetGalleryFeature(gallery_dir, &gallery_feature, FEATURE_SIZE);
  if (ret != 0) {
    printf("get gallery feature from %s failed with %#x!\n", gallery_dir, ret);
    goto exit0;
  }

  TDLImage image = NULL;
  TDLHandle tdl_handle = TDL_CreateHandle(0);

  char **channel_names = NULL;
  uint8_t channel_size = 0;

  ret = InitCamera(tdl_handle, VI_WIDTH, VI_HEIGHT, IMAGE_YUV420SP_UV, 3);
  if (ret != 0) {
    printf("InitCamera %#x!\n", ret);
    return ret;
  }

  ret = TDL_APP_Init(tdl_handle, "face_pet_capture", config_file,
                     &channel_names, &channel_size);
  if (ret != 0) {
    printf("TDL_APP_Init failed with %#x!\n", ret);
    goto exit1;
  }

  // 设置终端为非规范模式
  struct termios oldt, newt;
  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);

  printf("按任意键退出...\n");

  pthread_t stFrameThread, stTDLThread;

  SEND_FRAME_THREAD_ARG_S frame_args = {.tdl_handle = tdl_handle,
                                        .vi_chn = chn,
                                        .channel_size = channel_size,
                                        .channel_names = channel_names};

  RUN_TDL_THREAD_ARG_S tdl_args = {.tdl_handle = tdl_handle,
                                   .vi_chn = chn,
                                   .channel_size = channel_size,
                                   .channel_names = channel_names,
                                   .gallery_feature = &gallery_feature,
                                   .output_dir = output_dir};

  pthread_create(&stFrameThread, NULL, send_frame_thread, &frame_args);
  pthread_create(&stTDLThread, NULL, run_tdl_thread, &tdl_args);

  pthread_join(stFrameThread, NULL);
  pthread_join(stTDLThread, NULL);

  // 恢复终端设置
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);

exit1:
  for (int i = 0; i < channel_size; i++) {
    free(channel_names[i]);
  }
  free(channel_names);

exit0:
  DestroyQueue(&image_queue);
  for (int i = 0; i < gallery_feature.size; i++) {
    TDL_ReleaseFeatureMeta(&gallery_feature.feature[i]);
  }
  DestoryCamera(tdl_handle);
  TDL_DestroyHandle(tdl_handle);

  return ret;
}
