#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
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

#define VI_WIDTH 960
#define VI_HEIGHT 540

static volatile bool to_exit = false;
static ImageQueue image_queue;
TDLObject g_obj_meta = {0};
static uint32_t g_frame_id = 0;
MUTEXAUTOLOCK_INIT(ResultMutex);

// Global state enum
typedef enum { DETECTION = 0, TRACKING = 2 } SystemStatus;

RtspContext rtsp_context = {.chn = 0,
                            .pay_load_type = PT_H264,
                            .frame_width = VI_WIDTH,
                            .frame_height = VI_HEIGHT};

// Global variables
SystemStatus g_status = DETECTION;   // Initial state is detection
uint32_t g_lost_start_time;          // Start time when the object is lost
bool g_lost_timer_started = false;   // Whether the lost timer has started
const int LOST_TIMEOUT_SECONDS = 5;  // Timeout for object lost

static uint32_t get_time_in_ms() {
  struct timeval tv;
  if (gettimeofday(&tv, NULL) < 0) {
    return 0;
  }
  return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}
// 设置终端为非阻塞模式
void set_non_blocking_input() {
  struct termios oldt, newt;
  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  int old_flags = fcntl(STDIN_FILENO, F_GETFL, 0);
  fcntl(STDIN_FILENO, F_SETFL, old_flags | O_NONBLOCK);
}

// 恢复终端设置
void restoreTerminal() {
  struct termios oldt;
  tcgetattr(STDIN_FILENO, &oldt);
  oldt.c_lflag |= (ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  fcntl(STDIN_FILENO, F_SETFL, 0);
}

char check_key_input() {
  char ch = 0;
  if (read(STDIN_FILENO, &ch, 1) > 0) {
    return ch;
  }
  return 0;
}

typedef struct {
  TDLHandle tdl_handle;
  int vi_chn;
  TDLModel det_model_id;
  TDLModel sot_model_id;
} THREAD_ARG_S;

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf("  %s -d <det_model_path> -s <sot_model_path>\n", prog_name);
  printf("  %s --det_model_path <path> --sot_model_path <path> \n", prog_name);
  printf("Options:\n");
  printf(
      "  -d, --det_model_path : person vehicle detection model path\n"
      "  -s, --sot_model_path : sot model path\n"
      "  -h, --help : print help\n");
}

void *get_frame_thread(void *args) {
  printf("Enter send frame thread\n");
  THREAD_ARG_S *pstArgs = (THREAD_ARG_S *)args;
  TDLObject obj_meta = {0};

  int ret;

  while (to_exit == false) {
    TDLImage image = NULL;

    image =
        GetCameraFrame(pstArgs->tdl_handle,
                       pstArgs->vi_chn);  // if channel_size > 1, image should
                                          // be taken from different vi_chn
    if (image == NULL) {
      printf("GetCameraFrame falied\n");
      continue;
    }
    g_frame_id++;

    ret = Image_Enqueue(&image_queue, image);
    if (ret != 0) {
      printf("Image_Enqueue falied\n");
      ReleaseCameraFrame(pstArgs->tdl_handle, pstArgs->vi_chn);
      TDL_DestroyImage(image);
      continue;
    }
  }

  return NULL;
}

void *run_det_thread(void *args) {
  THREAD_ARG_S *pstArgs = (THREAD_ARG_S *)args;

  TDLObject obj_meta = {0};
  VIDEO_FRAME_INFO_S *frame = NULL;
  TDLImage image = NULL;

  while (to_exit == false) {
    if (g_status == DETECTION) {
      // printf("to do TDL_Detection\n");

      {
        MutexAutoLock(ResultMutex, lock);
        image = Image_Dequeue(&image_queue);
      }
      if (image) {
        int ret = TDL_Detection(pstArgs->tdl_handle, pstArgs->det_model_id,
                                image, &obj_meta);
        if (ret != 0) {
          printf("TDL_Detection failed with %#x!\n", ret);
        }

        {
          MutexAutoLock(ResultMutex, lock);
          TDL_CopyObjectMeta(&obj_meta, &g_obj_meta);
        }

        TDL_WrapImage(image, &frame);

        TDLBrush brush = {0};
        brush.size = 5;
        brush.color.r = 0;
        brush.color.g = 255;
        brush.color.b = 0;
        for (int i = 0; i < obj_meta.size; i++) {
          TDLObjectInfo *obj_info = &obj_meta.info[i];
          snprintf(obj_info->name, sizeof(obj_info->name), "index:%d", i);
        }
        DrawObjRect(&obj_meta, frame, true, brush);
        TDL_ReleaseObjectMeta(&obj_meta);

        ret = SendFrameRTSP(frame, &rtsp_context);
        if (ret != 0) {
          printf("SendFrameRTSP failed with %#x!\n", ret);
          continue;
        }

        TDL_DestroyImage(image);
        ReleaseCameraFrame(pstArgs->tdl_handle, pstArgs->vi_chn);
      }
    } else {
      usleep(10 * 1000);
    }
  }

  return NULL;

exit0:
  DestoryCamera(pstArgs->tdl_handle);
  TDL_DestroyHandle(pstArgs->tdl_handle);
}

void *run_sot_thread(void *args) {
  THREAD_ARG_S *pstArgs = (THREAD_ARG_S *)args;

  TDLTracker track_meta = {0};
  TDLObject track_obj_meta = {0};
  TDL_InitObjectMeta(&track_obj_meta, 1, 0);
  uint64_t counter = 0;
  uint64_t last_counter = 0;
  uint32_t last_time_ms = get_time_in_ms();
  TDLImage image = NULL;
  VIDEO_FRAME_INFO_S *frame = NULL;

  printf("Usage: input i or I to start tracking ......\n");
  while (to_exit == false) {
    char key = check_key_input();
    if (key == 'i' || key == 'I') {
      restoreTerminal();  // 临时恢复终端设置以便输入
      if (g_obj_meta.size > 0) {
        printf(
            "Enter  box index 0-%d or box x1,y1,x2,y2 or a point x,y to track: "
            "\n",
            g_obj_meta.size - 1);
      } else {
        printf(
            "Enter bbox x1,y1,x2,y2 or a point x,y or bbox index to track: \n");
      }
      char input[100];
      while (true) {
        char *p = fgets(input, sizeof(input), stdin);
        if (p != NULL) {
          int values[4];
          int num_values = sscanf(input, "%d,%d,%d,%d", &values[0], &values[1],
                                  &values[2], &values[3]);

          if (num_values != 1 && num_values != 2 && num_values != 4) {
            printf("num_values should be 1 or 2 or 4\n");
            continue;
          }

          if (num_values == 1 && values[0] > g_obj_meta.size - 1) {
            printf("box index out of range\n");
            continue;
          }

          for (int i = 0; i < num_values; i++) {
            printf("values[%d] = %d\n", i, values[i]);
          }

          bool init_success = false;
          while (true) {
            g_status = TRACKING;
            {
              MutexAutoLock(ResultMutex, lock);
              image = Image_Dequeue(&image_queue);
            }
            if (image) {
              printf("g_obj_meta.size = %d\n", g_obj_meta.size);

              int ret = TDL_SetSingleObjectTracking(
                  pstArgs->tdl_handle, image, &g_obj_meta, values, num_values);
              if (ret != 0) {
                printf("TDL_SetSingleObjectTracking failed with %#x!\n", ret);
                TDL_DestroyImage(image);
                ReleaseCameraFrame(pstArgs->tdl_handle, pstArgs->vi_chn);
                break;
              }
              init_success = true;
              TDL_DestroyImage(image);
              ReleaseCameraFrame(pstArgs->tdl_handle, pstArgs->vi_chn);

              set_non_blocking_input();  // 重新设置非阻塞输入

              break;
            }
          }
          if (init_success) {
            break;
          }
        }
      }
    }

    if (g_status == TRACKING) {
      {
        MutexAutoLock(ResultMutex, lock);
        image = Image_Dequeue(&image_queue);
      }

      if (image) {
        int ret = TDL_SingleObjectTracking(pstArgs->tdl_handle, image,
                                           &track_meta, g_frame_id);

        if (ret != 0) {
          printf("TDL_SingleObjectTracking failed with %#x!\n", ret);
        }

        TDL_WrapImage(image, &frame);

        if (track_meta.info) {
          TDLBrush brush = {0};
          brush.size = 5;
          brush.color.r = 255;
          brush.color.g = 0;
          brush.color.b = 0;

          track_obj_meta.info[0].box.x1 = track_meta.info[0].bbox.x1;
          track_obj_meta.info[0].box.y1 = track_meta.info[0].bbox.y1;
          track_obj_meta.info[0].box.x2 = track_meta.info[0].bbox.x2;
          track_obj_meta.info[0].box.y2 = track_meta.info[0].bbox.y2;

          DrawObjRect(&track_obj_meta, frame, true, brush);

          g_lost_timer_started = false;
        } else {
          // Start or continue lost timing
          if (!g_lost_timer_started) {
            g_lost_start_time = get_time_in_ms();
            g_lost_timer_started = true;
          } else {
            uint32_t current_time = get_time_in_ms();
            uint32_t elapsed_time = current_time - g_lost_start_time;
            if (elapsed_time >= LOST_TIMEOUT_SECONDS * 1000) {
              printf(
                  "The target has been lost for more than [%d] seconds, "
                  "switching to detection state\n",
                  LOST_TIMEOUT_SECONDS);
              g_status = DETECTION;
              g_lost_timer_started = false;
            }
          }
        }

        ret = SendFrameRTSP(frame, &rtsp_context);
        if (ret != 0) {
          printf("SendFrameRTSP failed with %#x!\n", ret);
          continue;
        }

        TDL_DestroyImage(image);
        ReleaseCameraFrame(pstArgs->tdl_handle, pstArgs->vi_chn);
      }
    }

    TDL_ReleaseTrackMeta(&track_meta);
  }

  return NULL;

exit0:
  TDL_ReleaseObjectMeta(&track_obj_meta);
  DestoryCamera(pstArgs->tdl_handle);
  TDL_DestroyHandle(pstArgs->tdl_handle);
}

int main(int argc, char *argv[]) {
  char *det_model_path = NULL;
  char *sot_model_path = NULL;
  int vi_chn = 0;

  struct option long_options[] = {
      {"det_model_path", required_argument, 0, 'd'},
      {"sot_model_path", required_argument, 0, 's'},
      {"help", no_argument, 0, 'h'},
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "d:s:h", long_options, NULL)) != -1) {
    switch (opt) {
      case 'd':
        det_model_path = optarg;
        break;
      case 's':
        sot_model_path = optarg;
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

  if (!det_model_path || !sot_model_path) {
    fprintf(stderr, "Error: det_model_path and sot_model_path are required\n");
    print_usage(argv[0]);
    return -1;
  }

  printf("Running with:\n");
  printf("det_model_path:    %s\n", det_model_path);
  printf("sot_model_path:    %s\n", sot_model_path);
  printf("vi_chn:        %d\n", vi_chn);

  InitQueue(&image_queue);

  TDLImage image = NULL;
  TDLHandle tdl_handle = TDL_CreateHandle(0);

  int ret = InitCamera(tdl_handle, VI_WIDTH, VI_HEIGHT, IMAGE_YUV420SP_UV, 3);
  if (ret != 0) {
    printf("InitCamera %#x!\n", ret);
    goto exit0;
  }

  TDLModel det_model_id = TDL_MODEL_YOLOV8N_DET_PERSON_VEHICLE;
  ret = TDL_OpenModel(tdl_handle, det_model_id, det_model_path, NULL);
  if (ret != 0) {
    printf("open detection model failed with %#x!\n", ret);
    goto exit1;
  }

  TDLModel sot_model_id = TDL_MODEL_TRACKING_FEARTRACK;
  ret = TDL_OpenModel(tdl_handle, sot_model_id, sot_model_path, NULL);
  if (ret != 0) {
    printf("open sot model failed with %#x!\n", ret);
    goto exit2;
  }

  set_non_blocking_input();

  pthread_t stFrameThread, stDetThread, stSotThread;

  THREAD_ARG_S total_args = {.tdl_handle = tdl_handle,
                             .det_model_id = det_model_id,
                             .sot_model_id = sot_model_id,
                             .vi_chn = vi_chn};

  pthread_create(&stFrameThread, NULL, get_frame_thread, &total_args);
  pthread_create(&stDetThread, NULL, run_det_thread, &total_args);
  pthread_create(&stSotThread, NULL, run_sot_thread, &total_args);

  pthread_join(stFrameThread, NULL);
  pthread_join(stDetThread, NULL);
  pthread_join(stSotThread, NULL);

exit2:
  TDL_CloseModel(tdl_handle, det_model_id);

exit1:
  TDL_CloseModel(tdl_handle, sot_model_id);

exit0:
  DestroyQueue(&image_queue);

  DestoryCamera(tdl_handle);
  TDL_DestroyHandle(tdl_handle);

  return ret;
}
