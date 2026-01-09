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
} RUN_TDL_THREAD_ARG_S;

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf("  %s -c <config_file> -v <vi_chn>\n", prog_name);
  printf("  %s --config_file <path> --vi_chn <int> \n", prog_name);
  printf("Options:\n");
  printf(
      "  -c, --config_file : json config file\n"
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

      int ret = TDL_APP_HumanPoseSmooth(
          pstArgs->tdl_handle, pstArgs->channel_names[i], &capture_info);

      if (ret == 1) {
        continue;
      } else if (ret == 2) {
        to_exit = true;
        break;
      } else if (ret != 0) {
        printf("TDL_APP_HumanPoseSmooth failed with %#x!\n", ret);
        goto exit0;
      }

      if (frm_diff > 30) {
        printf("detect person size: %d\n", capture_info.person_meta.size);
      }

#ifdef ENABLE_RTSP
      TDLImage image = capture_info.image;
      TDL_WrapImage(image, &frame);

      TDLObject kps_meta = {
          0};  // Temporary solution, use boxes to draw key points.
      memset(&kps_meta, 0, sizeof(TDLObject));
      TDL_InitObjectMeta(&kps_meta, 17, 0);

      for (int j = 0; j < capture_info.person_meta.size; j++) {
        TDLBrush brush = {0};
        brush.size = 2;
        brush.color.r = 0;
        brush.color.g = 255;
        brush.color.b = 0;
        TDLObjectInfo *obj_info = &capture_info.person_meta.info[j];
        snprintf(obj_info->name, sizeof(obj_info->name), "id:%d",
                 obj_info->track_id);
        DrawObjRect(&capture_info.person_meta, frame, true, brush);  // box

        brush.color.g = 0;
        brush.color.b = 255;

        for (int k = 0; k < 17; k++) {
          if (capture_info.person_meta.info[j].landmark_properity[k].score <
              0.5) {  // skip
            kps_meta.info[k].box.x1 = 0;
            kps_meta.info[k].box.y1 = 0;
            kps_meta.info[k].box.x2 = 0;
            kps_meta.info[k].box.y2 = 0;
            continue;
          };
          float x = capture_info.person_meta.info[j]
                        .landmark_properity[k]
                        .x;  // keypoint x
          float y = capture_info.person_meta.info[j]
                        .landmark_properity[k]
                        .y;  // keypoint y

          float deta = 10.0f * (float)capture_info.frame_width / 1920.0f;
          kps_meta.info[k].box.x1 = x - deta > 0 ? x - deta : 0;
          kps_meta.info[k].box.y1 = y - deta > 0 ? y - deta : 0;
          kps_meta.info[k].box.x2 = x + deta < capture_info.frame_width - 1
                                        ? x + deta
                                        : capture_info.frame_width - 1;
          kps_meta.info[k].box.y2 = y + deta < capture_info.frame_height - 1
                                        ? y + deta
                                        : capture_info.frame_height - 1;
        }

        DrawObjRect(&kps_meta, frame, true, brush);
      }

      ret = SendFrameRTSP(frame, &rtsp_context);
      if (ret != 0) {
        printf("SendFrameRTSP failed with %#x!\n", ret);
        continue;
      }
      TDL_ReleaseObjectMeta(&kps_meta);

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
  char *config_file = NULL;  // sample/config/human_pose_smooth_app_vi.json
  char *vi_chn = NULL;
  int chn = 0;

  struct option long_options[] = {
      {"config_file", required_argument, 0, 'c'},
      {"vi_chn", no_argument, 0, 'v'},
      {"help", no_argument, 0, 'h'},
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "c:o:v:h", long_options, NULL)) != -1) {
    switch (opt) {
      case 'c':
        config_file = optarg;
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
  printf("  config_file:    %s\n", config_file);
  printf("  vi_chn:        %d\n", chn);

  InitQueue(&image_queue);

  TDLImage image = NULL;
  TDLHandle tdl_handle = TDL_CreateHandle(0);

  char **channel_names = NULL;
  uint8_t channel_size = 0;

  int ret = InitCamera(tdl_handle, VI_WIDTH, VI_HEIGHT, IMAGE_YUV420SP_UV, 3);
  if (ret != 0) {
    printf("InitCamera %#x!\n", ret);
    return ret;
  }

  ret = TDL_APP_Init(tdl_handle, "human_pose_smooth", config_file,
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
                                   .channel_names = channel_names};

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
  DestoryCamera(tdl_handle);
  TDL_DestroyHandle(tdl_handle);

  return ret;
}
