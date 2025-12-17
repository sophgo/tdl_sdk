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

// --- Threading Structures ---
RtspContext rtsp_context = {.chn = 0,
                            .pay_load_type = PT_H264,
                            .frame_width = WIDTH,
                            .frame_height = HEIGHT};

typedef struct {
  TDLHandle tdl_handle;
  int vi_chn;
  TDLModel model_id;
  RtspContext *rtsp_context;
} ProcessArgs;

int get_model_info(char *model_path, TDLModel *model_index) {
  if (strstr(model_path, "yolov5m_det_coco80_640_640") != NULL) {
    *model_index = TDL_MODEL_YOLOV5_DET_COCO80;
  } else if (strstr(model_path, "yolov5s_det_coco80_640_640") != NULL) {
    *model_index = TDL_MODEL_YOLOV5_DET_COCO80;
  } else if (strstr(model_path, "yolov6n_det_coco80_640_640") != NULL) {
    *model_index = TDL_MODEL_YOLOV6_DET_COCO80;
  } else if (strstr(model_path, "yolov6s_det_coco80_640_640") != NULL) {
    *model_index = TDL_MODEL_YOLOV6_DET_COCO80;
  } else if (strstr(model_path, "yolov7_tiny_det_coco80_640_640") != NULL) {
    *model_index = TDL_MODEL_YOLOV7_DET_COCO80;
  } else if (strstr(model_path, "yolov8n_det_coco80_640_640") != NULL) {
    *model_index = TDL_MODEL_YOLOV8_DET_COCO80;
  } else if (strstr(model_path, "yolov8s_det_coco80_640_640") != NULL) {
    *model_index = TDL_MODEL_YOLOV8_DET_COCO80;
  } else if (strstr(model_path, "yolox_m_det_coco80_640_640") != NULL) {
    *model_index = TDL_MODEL_YOLOX_DET_COCO80;
  } else if (strstr(model_path, "yolox_s_det_coco80_640_640") != NULL) {
    *model_index = TDL_MODEL_YOLOX_DET_COCO80;
  } else if (strstr(model_path, "yolov10n_det_coco80_640_640") != NULL) {
    *model_index = TDL_MODEL_YOLOV10_DET_COCO80;
  } else if (strstr(model_path, "ppyoloe_det_coco80_640_640") != NULL) {
    *model_index = TDL_MODEL_PPYOLOE_DET_COCO80;
  } else if (strstr(model_path, "yolov8") != NULL) {
    *model_index = TDL_MODEL_YOLOV8;
  } else if (strstr(model_path, "yolov10") != NULL) {
    *model_index = TDL_MODEL_YOLOV10;
  } else {
    return -1;
  }
}

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf("  %s -m <model_path>\n", prog_name);
  printf("  %s --model_path <path>\n\n", prog_name);
  printf("Options:\n");
  printf(
      "  -m, --model_path  Path to cvimodel eg. "
      "<yolov8n_det_person_vehicle>\n");
  printf("  -h, --help        Show this help message\n");
}

// --- Worker Threads ---
void *capture_thread(void *arg) {
  ProcessArgs *args = (ProcessArgs *)arg;
  TDLImage image = NULL;

  while (to_exit == false) {
    // Capture frame (this might block waiting for VSYNC)
    image = GetCameraFrame(args->tdl_handle, args->vi_chn);
    if (image == NULL) {
      fprintf(stderr, "Error: GetCameraFrame failed!\n");
      // Avoid busy loop if camera fails
      usleep(10000);
      continue;
    }

    if (Image_Enqueue(&image_queue, image) == -1) {
      if (to_exit) {
        ReleaseCameraFrame(args->tdl_handle, args->vi_chn);
        TDL_DestroyImage(image);
        break;
      }
      fprintf(stderr, "Error: Image_Enqueue failed!\n");
      ReleaseCameraFrame(args->tdl_handle, args->vi_chn);
      TDL_DestroyImage(image);
      continue;
    }
  }

  return NULL;
}

void *process_thread(void *arg) {
  ProcessArgs *args = (ProcessArgs *)arg;
  TDLImage image = NULL;
  VIDEO_FRAME_INFO_S *frame = NULL;
  TDLObject obj_meta = {0};
  int ret;
  struct timeval start_time, end_time;
  static unsigned int frame_count = 0;

  while (to_exit == false) {
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
    // Object Detection
    memset(&obj_meta, 0, sizeof(TDLObject));
    gettimeofday(&start_time, NULL);
    // Note: Assuming TDL_Detection is thread-safe relative to GetCameraFrame
    ret = TDL_Detection(args->tdl_handle, args->model_id, image, &obj_meta);
    if (ret != 0) {
      fprintf(stderr, "Error: TDL_Detection failed!\n");
    }

    gettimeofday(&end_time, NULL);
    double elapsed_ms = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                        (end_time.tv_usec - start_time.tv_usec) / 1000.0;

    if (frame_count % 20 == 1) {
      printf("Model inference time : %.2f ms\n", elapsed_ms);
    }
    frame_count++;

    // Process Frame
    TDL_WrapImage(image, &frame);

    TDLBrush brush = {0};
    brush.size = 5;
    brush.color.r = 255;
    brush.color.g = 0;
    brush.color.b = 0;
    for (int i = 0; i < obj_meta.size; i++) {
      // 如果需要打印模型输出将以下注释取消
      // printf("obj_meta_index : %d, ", i);
      // printf("class_id : %d, ", obj_meta.info[i].class_id);
      // printf("score : %f, ", obj_meta.info[i].score);
      // printf("bbox : [%f %f %f %f]\n", obj_meta.info[i].box.x1,
      //         obj_meta.info[i].box.x2, obj_meta.info[i].box.y1,
      //         obj_meta.info[i].box.y2);

      TDLObjectInfo *obj_info = &obj_meta.info[i];
      snprintf(obj_info->name, sizeof(obj_info->name), "class:%d score:%.2f",
               obj_info->class_id, obj_info->score);
    }
    DrawObjRect(&obj_meta, frame, true, brush);

    // Send RTSP Frame
    ret = SendFrameRTSP(frame, args->rtsp_context);
    if (ret != 0) {
      fprintf(stderr, "Error: SendFrameRTSP failed!\n");
    }

    // Cleanup
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

  struct option long_options[] = {{"model_path", required_argument, 0, 'm'},
                                  {"help", no_argument, 0, 'h'},
                                  {NULL, 0, NULL, 0}};

  int opt;
  while ((opt = getopt_long(argc, argv, "m:h", long_options, NULL)) != -1) {
    switch (opt) {
      case 'm':
        model_path = optarg;
        break;
      case 'h':
        print_usage(argv[0]);
        return 0;
      case '?':
        print_usage(argv[0]);
        return 0;
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
  printf("  vi_chn:        %d\n", vi_chn);
  printf("  rtsp_chn:      %d\n", rtsp_chn);

  TDLModel model_id;
  TDLHandle tdl_handle = TDL_CreateHandle(0);

  if (get_model_info(model_path, &model_id) == -1) {
    fprintf(stderr, "Error: unsupported model: %s\n", model_path);
    return -1;
  }

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

  // Set terminal to non-canonical mode
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
    usleep(100000);  // Check input every 100ms
  }

  // Restore terminal
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);

  printf("Stopping threads...\n");
  // Signal threads to stop
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
