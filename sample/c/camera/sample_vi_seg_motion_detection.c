#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/time.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#include "meta_visualize.h"
#include "pthread_utils.h"
#include "sample_utils.h"
#include "tdl_sdk.h"

#define WIDTH 1280
#define HEIGHT 720

static volatile bool to_exit = false;
static ImageQueue image_queue;
MUTEXAUTOLOCK_INIT(ResultMutex);

RtspContext rtsp_context = {.chn = 0,
                            .pay_load_type = PT_H264,
                            .frame_width = WIDTH,
                            .frame_height = HEIGHT};

typedef struct {
  TDLHandle tdl_handle;
  int vi_chn;
  TDLModel model_id;
  uint32_t min_area;
  RtspContext *rtsp_context;
} ProcessArgs;

static void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf("  %s -m <model_path> [-a <min_area>]\n", prog_name);
  printf("  %s --model_path <path> [--min_area <min_area>]\n\n", prog_name);
  printf("Options:\n");
  printf("  -m, --model_path  Path to cvimodel eg. <topformer_seg_motion>\n");
  printf(
      "  -a, --min_area    Minimum connected-component area (default: 256)\n");
  printf("  -h, --help        Show this help message\n");
}

static void *capture_thread(void *arg) {
  ProcessArgs *args = (ProcessArgs *)arg;
  TDLImage image = NULL;

  while (!to_exit) {
    image = GetCameraFrame(args->tdl_handle, args->vi_chn);
    if (image == NULL) {
      usleep(10000);
      continue;
    }

    if (Image_Enqueue(&image_queue, image) == -1) {
      if (to_exit) {
        ReleaseCameraFrame(args->tdl_handle, args->vi_chn);
        TDL_DestroyImage(image);
        break;
      }
      ReleaseCameraFrame(args->tdl_handle, args->vi_chn);
      TDL_DestroyImage(image);
      continue;
    }
  }
  return NULL;
}

static void *process_thread(void *arg) {
  ProcessArgs *args = (ProcessArgs *)arg;
  TDLImage image = NULL;
  VIDEO_FRAME_INFO_S *frame = NULL;
  TDLObject obj_meta = {0};
  int ret;
  struct timeval start_time, end_time;
  static unsigned int frame_count = 0;

  while (!to_exit) {
    {
      MutexAutoLock(ResultMutex, lock);
      image = Image_Dequeue(&image_queue);
      if (image == NULL) {
        if (to_exit) {
          break;
        }
        continue;
      }
    }

    memset(&obj_meta, 0, sizeof(TDLObject));
    gettimeofday(&start_time, NULL);
    ret = TDL_SegMotionDetection(args->tdl_handle, args->model_id, image,
                                 args->min_area, &obj_meta);
    gettimeofday(&end_time, NULL);
    double elapsed_ms = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                        (end_time.tv_usec - start_time.tv_usec) / 1000.0;

    if (ret != 0) {
      fprintf(stderr, "Error: TDL_SegMotionDetection failed!\n");
    } else {
      if (frame_count % 20 == 1) {
        printf("Model inference time : %.2f ms, box_num: %u\n", elapsed_ms,
               obj_meta.size);
      }
    }
    frame_count++;

    TDL_WrapImage(image, &frame);

    TDLBrush brush = {0};
    brush.size = 5;
    brush.color.r = 255;
    brush.color.g = 0;
    brush.color.b = 0;

    DrawObjRect(&obj_meta, frame, true, brush);

    ret = SendFrameRTSP(frame, args->rtsp_context);
    if (ret != 0) {
      fprintf(stderr, "Error: SendFrameRTSP failed!\n");
    }

    TDL_ReleaseObjectMeta(&obj_meta);
    TDL_DestroyImage(image);
    ReleaseCameraFrame(args->tdl_handle, args->vi_chn);
  }
  return NULL;
}

int main(int argc, char *argv[]) {
  int ret = 0;
  char *model_path = NULL;
  int rtsp_chn = 0;
  int vi_chn = 0;
  uint32_t min_area = 256;

  struct option long_options[] = {{"model_path", required_argument, 0, 'm'},
                                  {"min_area", required_argument, 0, 'a'},
                                  {"help", no_argument, 0, 'h'},
                                  {NULL, 0, NULL, 0}};

  int opt;
  while ((opt = getopt_long(argc, argv, "m:a:h", long_options, NULL)) != -1) {
    switch (opt) {
      case 'm':
        model_path = optarg;
        break;
      case 'a':
        min_area = (uint32_t)strtoul(optarg, NULL, 10);
        break;
      case 'h':
        print_usage(argv[0]);
        return 0;
      case '?':
      default:
        print_usage(argv[0]);
        return 0;
    }
  }

  if (!model_path) {
    fprintf(stderr, "Error: Model path is required\n");
    print_usage(argv[0]);
    return -1;
  }

  printf("Running with:\n");
  printf("  Model path:    %s\n", model_path);
  printf("  min_area:      %u\n", min_area);
  printf("  vi_chn:        %d\n", vi_chn);
  printf("  rtsp_chn:      %d\n", rtsp_chn);

  TDLModel model_id = TDL_MODEL_TOPFORMER_SEG_MOTION;
  TDLHandle tdl_handle = TDL_CreateHandle(0);

  ret = InitCamera(tdl_handle, WIDTH, HEIGHT, IMAGE_YUV420SP_UV, 3);
  if (ret != 0) {
    fprintf(stderr, "Error: InitCamera failed!\n");
    TDL_DestroyHandle(tdl_handle);
    return ret;
  }

  ret = TDL_OpenModel(tdl_handle, model_id, model_path, NULL, 0);
  if (ret != 0) {
    fprintf(stderr, "Error: open model failed!\n");
    DestoryCamera(tdl_handle);
    TDL_DestroyHandle(tdl_handle);
    return ret;
  }

  ProcessArgs process_args;
  process_args.tdl_handle = tdl_handle;
  process_args.vi_chn = vi_chn;
  process_args.model_id = model_id;
  process_args.min_area = min_area;
  process_args.rtsp_context = &rtsp_context;

  InitQueue(&image_queue);

  pthread_t capture_tid, process_tid;
  printf("Starting threads...\n");
  if (pthread_create(&capture_tid, NULL, capture_thread, &process_args) != 0) {
    fprintf(stderr, "Error: Failed to create capture thread!\n");
    DestroyQueue(&image_queue);
    TDL_CloseModel(tdl_handle, model_id);
    DestoryCamera(tdl_handle);
    TDL_DestroyHandle(tdl_handle);
    return -1;
  }
  if (pthread_create(&process_tid, NULL, process_thread, &process_args) != 0) {
    fprintf(stderr, "Error: Failed to create process thread!\n");
    DestroyQueue(&image_queue);
    TDL_CloseModel(tdl_handle, model_id);
    DestoryCamera(tdl_handle);
    TDL_DestroyHandle(tdl_handle);
    return -1;
  }

  struct termios oldt, newt;
  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);

  printf("按任意键退出...\n");
  while (1) {
    fd_set rfds;
    struct timeval tv = {0, 0};
    FD_ZERO(&rfds);
    FD_SET(STDIN_FILENO, &rfds);
    int key_pressed = select(STDIN_FILENO + 1, &rfds, NULL, NULL, &tv);
    if (key_pressed > 0 && FD_ISSET(STDIN_FILENO, &rfds)) {
      break;
    }
    usleep(100000);
  }

  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);

  printf("Stopping threads...\n");
  to_exit = true;
  ExitQueue(&image_queue);
  pthread_join(capture_tid, NULL);
  pthread_join(process_tid, NULL);

  DestroyQueue(&image_queue);
  TDL_CloseModel(tdl_handle, model_id);
  DestoryCamera(tdl_handle);
  TDL_DestroyHandle(tdl_handle);
  return ret;
}
