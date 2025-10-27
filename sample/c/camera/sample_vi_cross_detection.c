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

// #define ENABLE_RTSP

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

      if (channel_frame_id[i] == 1000) {  // reset cross detection line
        ret = TDL_APP_ObjectCountingSetLine(pstArgs->tdl_handle,
                                            pstArgs->channel_names[i], 480, 20,
                                            480, 520, 2);
        if (ret != 0) {
          printf("TDL_APP_ObjectCountingSetLine failed with %d\n", ret);
          continue;
        }
      }

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

  int cross_count = 0;
  while (to_exit == false) {
    for (size_t i = 0; i < pstArgs->channel_size; i++) {
      TDLObjectCountingInfo cross_detection_info = {0};

      int ret =
          TDL_APP_ObjectCounting(pstArgs->tdl_handle, pstArgs->channel_names[i],
                                 &cross_detection_info);

      for (int j = 0; j < cross_detection_info.object_meta.size; j++) {
        TDLObjectInfo *obj_info = &cross_detection_info.object_meta.info[j];

        if (obj_info->is_cross) {
          cross_count += 1;
        }
      }

      if (ret == 1) {
        continue;
      } else if (ret == 2) {
        to_exit = true;
        break;
      } else if (ret != 0) {
        printf("TDL_APP_Capture failed with %#x!\n", ret);
        goto exit0;
      }

      counter++;
      int frm_diff = counter - last_counter;
      if (frm_diff > 30) {
        uint32_t cur_ts_ms = get_time_in_ms();
        float infer_time = (float)(cur_ts_ms - last_time_ms) / frm_diff;
        float fps = 1000.0 / infer_time;

        last_time_ms = cur_ts_ms;
        last_counter = counter;

        printf("cross num: %d\n", cross_count);
        printf(
            "+++++++++++++++++++++++++++++++++++ frame:%d, infer time:%.2f, "
            "fps:%.2f\n",
            (int)counter, infer_time, fps);
        cross_count = 0;
      }

#ifdef ENABLE_RTSP
      TDLImage image = cross_detection_info.image;
      TDL_WrapImage(image, &frame);

      TDLBrush brush = {0};
      brush.size = 5;
      brush.color.r = 0;
      brush.color.g = 255;
      brush.color.b = 0;
      for (int i = 0; i < cross_detection_info.object_meta.size; i++) {
        TDLObjectInfo *obj_info = &cross_detection_info.object_meta.info[i];
        snprintf(obj_info->name, sizeof(obj_info->name), "id:%d,is_cross:%d",
                 obj_info->track_id, (int)obj_info->is_cross);
      }
      DrawObjRect(&cross_detection_info.object_meta, frame, true, brush);

      brush.color.r = 255;
      brush.color.g = 0;

      char line_start[128] = {0};
      snprintf(line_start, sizeof(line_start), "start");  // TODO: TDL_DrawLine
      ObjectWriteText(line_start, cross_detection_info.counting_line[0],
                      cross_detection_info.counting_line[1], frame, brush);

      char line_end[128] = {0};
      snprintf(line_end, sizeof(line_end), "end");
      ObjectWriteText(line_end, cross_detection_info.counting_line[2],
                      cross_detection_info.counting_line[3], frame, brush);

      ret = SendFrameRTSP(frame, &rtsp_context);
      if (ret != 0) {
        printf("SendFrameRTSP failed with %#x!\n", ret);
        continue;
      }
#endif

      TDL_ReleaseObjectCountingInfo(&cross_detection_info);
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
  char *config_file = NULL;  // sample/config/cross_detection_app_vi.json
  int vi_chn = 0;

  struct option long_options[] = {
      {"config_file", required_argument, 0, 'c'},
      {"vi_chn", no_argument, 0, 'v'},
      {"help", no_argument, 0, 'h'},
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "c:v:h", long_options, NULL)) != -1) {
    switch (opt) {
      case 'c':
        config_file = optarg;
        break;
      case 'v':
        vi_chn = atoi(optarg);
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
    fprintf(stderr, "Error: config_file is required\n");
    print_usage(argv[0]);
    return -1;
  }

  printf("Running with:\n");
  printf("  config_file:    %s\n", config_file);
  printf("  vi_chn:        %d\n", vi_chn);

  TDLImage image = NULL;
  TDLHandle tdl_handle = TDL_CreateHandle(0);

  char **channel_names = NULL;
  uint8_t channel_size = 0;
  int ret = InitCamera(tdl_handle, VI_WIDTH, VI_HEIGHT, IMAGE_YUV420SP_VU, 3);
  if (ret != 0) {
    printf("InitCamera %#x!\n", ret);
    return ret;
  }

  ret = TDL_APP_Init(tdl_handle, "cross_detection", config_file, &channel_names,
                     &channel_size);
  if (ret != 0) {
    printf("TDL_APP_Init failed with %#x!\n", ret);
    goto exit0;
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
                                        .vi_chn = vi_chn,
                                        .channel_size = channel_size,
                                        .channel_names = channel_names};

  RUN_TDL_THREAD_ARG_S tdl_args = {.tdl_handle = tdl_handle,
                                   .vi_chn = vi_chn,
                                   .channel_size = channel_size,
                                   .channel_names = channel_names};

  pthread_create(&stFrameThread, NULL, send_frame_thread, &frame_args);
  pthread_create(&stTDLThread, NULL, run_tdl_thread, &tdl_args);

  pthread_join(stFrameThread, NULL);
  pthread_join(stTDLThread, NULL);

  // 恢复终端设置
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);

exit0:
  DestroyQueue(&image_queue);
  for (int i = 0; i < channel_size; i++) {
    free(channel_names[i]);
  }
  free(channel_names);

  DestoryCamera(tdl_handle);
  TDL_DestroyHandle(tdl_handle);

  return ret;
}
