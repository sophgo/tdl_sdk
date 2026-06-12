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

#ifndef __CV184X__
#define ENABLE_RTSP
#endif

#define WIDTH 1280
#define HEIGHT 720

static volatile bool to_exit = false;
static ImageQueue image_queue;
MUTEXAUTOLOCK_INIT(ResultMutex);

#ifdef ENABLE_RTSP
RtspContext rtsp_context = {.chn = 0,
                            .pay_load_type = PT_H264,
                            .frame_width = WIDTH,
                            .frame_height = HEIGHT};
#endif

typedef struct {
  TDLHandle tdl_handle;
  int vi_chn;
  TDLModel model_id_detect;
  TDLModel model_id_keypoint;
  TDLModel model_id_recognition;
#ifdef ENABLE_RTSP
  RtspContext *rtsp_context;
#endif
} ProcessArgs;

int get_model_info(char *model_path, TDLModel *model_index) {
  if (strstr(model_path, "recognition_license_plate") != NULL) {
    *model_index = TDL_MODEL_RECOGNITION_LICENSE_PLATE;
  } else if (strstr(model_path, "keypoint_license_plate") != NULL) {
    *model_index = TDL_MODEL_KEYPOINT_LICENSE_PLATE;
  } else if (strstr(model_path, "yolov8n_det_license_plate") != NULL) {
    *model_index = TDL_MODEL_YOLOV8N_DET_LICENSE_PLATE;
  } else {
    return -1;
  }
  return 0;
}

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf(
      "  %s -m "
      "<detect_model_path>,<keypoint_model_path>,<recognition_model_path>\n",
      prog_name);
  printf(
      "  %s --model_path "
      "<detect_model_path>,<keypoint_model_path>,<recognition_model_path>\n\n",
      prog_name);
  printf("Options:\n");
  printf(
      "  -m, --model_path  Path to detect, keypoint and recognition model\n");
  printf("  -h, --help        Show this help message\n");
}

void *capture_thread(void *arg) {
  ProcessArgs *args = (ProcessArgs *)arg;
  TDLImage image = NULL;

  while (to_exit == false) {
    image = GetCameraFrame(args->tdl_handle, args->vi_chn);
    if (image == NULL) {
      fprintf(stderr, "Error: GetCameraFrame failed!\n");
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

    memset(&obj_meta, 0, sizeof(TDLObject));
    gettimeofday(&start_time, NULL);

    ret = TDL_Detection(args->tdl_handle, args->model_id_detect, image,
                        &obj_meta);
    if (ret != 0) {
      fprintf(stderr, "Error: TDL_Detection failed!\n");
      goto cleanup;
    }

    if (obj_meta.size > 0) {
      TDLImage *crop_image =
          (TDLImage *)malloc(sizeof(TDLImage) * obj_meta.size);
      ret = TDL_DetectionKeypoint(args->tdl_handle, args->model_id_keypoint,
                                  image, &obj_meta, crop_image);
      if (ret != 0) {
        fprintf(stderr, "Error: TDL_DetectionKeypoint failed!\n");
        free(crop_image);
        goto cleanup;
      }

      for (int32_t i = 0; i < obj_meta.size; i++) {
        TDLText ocr_meta = {0};
        ret = TDL_CharacterRecognition(args->tdl_handle,
                                       args->model_id_recognition,
                                       crop_image[i], &ocr_meta);
        if (ret == 0 && ocr_meta.size > 0 && ocr_meta.text_info != NULL) {
          snprintf(obj_meta.info[i].name, sizeof(obj_meta.info[i].name),
                   "plate:%s", ocr_meta.text_info);
        }
        TDL_ReleaseCharacterMeta(&ocr_meta);
      }

      for (int32_t i = 0; i < obj_meta.size; i++) {
        TDL_DestroyImage(crop_image[i]);
      }
      free(crop_image);
    }

    gettimeofday(&end_time, NULL);
    double elapsed_ms = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                        (end_time.tv_usec - start_time.tv_usec) / 1000.0;

    if (frame_count % 20 == 1) {
      printf("Model inference time : %.2f ms", elapsed_ms);
      if (obj_meta.size > 0) {
        printf(", %s", obj_meta.info[0].name);
      }
      printf("\n");
    }
    frame_count++;

    TDL_WrapImage(image, &frame);

    TDLBrush brush = {0};
    brush.size = 15;
    brush.color.r = 53;
    brush.color.g = 208;
    brush.color.b = 217;
    DrawObjRect(&obj_meta, frame, true, brush);  // opencv 的原因，中文显示乱码

#ifdef ENABLE_RTSP
    ret = SendFrameRTSP(frame, args->rtsp_context);
    if (ret != 0) {
      fprintf(stderr, "Error: SendFrameRTSP failed!\n");
    }
#endif

  cleanup:
    TDL_ReleaseObjectMeta(&obj_meta);
    TDL_DestroyImage(image);
    ReleaseCameraFrame(args->tdl_handle, args->vi_chn);
  }
  return NULL;
}

int main(int argc, char *argv[]) {
  int ret = 0;
  char *model_path = NULL;
#ifdef ENABLE_RTSP
  int rtsp_chn = 0;
#endif
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

  const char *first_comma = strchr(model_path, ',');
  if (!first_comma || first_comma == model_path || first_comma[1] == '\0') {
    fprintf(stderr,
            "Error: Models must be in format "
            "'model_path_detect,model_path_keypoint,model_path_recognition'\n");
    return -1;
  }
  const char *second_comma = strchr(first_comma + 1, ',');
  if (!second_comma || second_comma == first_comma + 1 ||
      second_comma[1] == '\0') {
    fprintf(stderr,
            "Error: Models must be in format "
            "'model_path_detect,model_path_keypoint,model_path_recognition'\n");
    return -1;
  }
  if (strchr(second_comma + 1, ',')) {
    fprintf(stderr, "Error: Exactly three model paths are required\n");
    return -1;
  }

  char *comm1 = (char *)first_comma;
  char *comm2 = (char *)second_comma;

  char *model_path_detect = model_path;
  *comm1 = '\0';
  char *model_path_keypoint = comm1 + 1;
  *comm2 = '\0';
  char *model_path_recognition = comm2 + 1;

  printf("Running with:\n");
  printf("  Model path_detect:      %s\n", model_path_detect);
  printf("  Model path_keypoint:    %s\n", model_path_keypoint);
  printf("  Model path_recognition: %s\n", model_path_recognition);
  printf("  vi_chn:                 %d\n", vi_chn);
#ifdef ENABLE_RTSP
  printf("  rtsp_chn:               %d\n", rtsp_chn);
#endif

  TDLModel model_id_detect;
  if (get_model_info(model_path_detect, &model_id_detect) != 0) {
    printf("unsupported detect model: %s\n", model_path_detect);
    return -1;
  }
  TDLModel model_id_keypoint;
  if (get_model_info(model_path_keypoint, &model_id_keypoint) != 0) {
    printf("unsupported keypoint model: %s\n", model_path_keypoint);
    return -1;
  }
  TDLModel model_id_recognition;
  if (get_model_info(model_path_recognition, &model_id_recognition) != 0) {
    printf("unsupported recognition model: %s\n", model_path_recognition);
    return -1;
  }

  TDLHandle tdl_handle = TDL_CreateHandle(0);

  ret = InitCamera(tdl_handle, WIDTH, HEIGHT, IMAGE_YUV420SP_UV, 3);
  if (ret != 0) {
    fprintf(stderr, "Error: InitCamera failed!\n");
    TDL_DestroyHandle(tdl_handle);
    return ret;
  }

  ret = TDL_OpenModel(tdl_handle, model_id_detect, model_path_detect, NULL, 0);
  if (ret != 0) {
    printf("open detect model failed with %#x!\n", ret);
    DestoryCamera(tdl_handle);
    TDL_DestroyHandle(tdl_handle);
    return ret;
  }

  ret = TDL_OpenModel(tdl_handle, model_id_keypoint, model_path_keypoint, NULL,
                      0);
  if (ret != 0) {
    printf("open keypoint model failed with %#x!\n", ret);
    TDL_CloseModel(tdl_handle, model_id_detect);
    DestoryCamera(tdl_handle);
    TDL_DestroyHandle(tdl_handle);
    return ret;
  }

  ret = TDL_OpenModel(tdl_handle, model_id_recognition, model_path_recognition,
                      NULL, 0);
  if (ret != 0) {
    printf("open recognition model failed with %#x!\n", ret);
    TDL_CloseModel(tdl_handle, model_id_keypoint);
    TDL_CloseModel(tdl_handle, model_id_detect);
    DestoryCamera(tdl_handle);
    TDL_DestroyHandle(tdl_handle);
    return ret;
  }

  ProcessArgs process_args;
  process_args.tdl_handle = tdl_handle;
  process_args.vi_chn = vi_chn;
  process_args.model_id_detect = model_id_detect;
  process_args.model_id_keypoint = model_id_keypoint;
  process_args.model_id_recognition = model_id_recognition;
#ifdef ENABLE_RTSP
  process_args.rtsp_context = &rtsp_context;
#endif

  InitQueue(&image_queue);

  pthread_t capture_tid, process_tid;

  printf("Starting threads...\n");
  if (pthread_create(&capture_tid, NULL, capture_thread, &process_args) != 0) {
    fprintf(stderr, "Error: Failed to create capture thread!\n");
    DestroyQueue(&image_queue);
    TDL_CloseModel(tdl_handle, model_id_recognition);
    TDL_CloseModel(tdl_handle, model_id_keypoint);
    TDL_CloseModel(tdl_handle, model_id_detect);
    DestoryCamera(tdl_handle);
    TDL_DestroyHandle(tdl_handle);
    return -1;
  }
  if (pthread_create(&process_tid, NULL, process_thread, &process_args) != 0) {
    fprintf(stderr, "Error: Failed to create process thread!\n");
    DestroyQueue(&image_queue);
    TDL_CloseModel(tdl_handle, model_id_recognition);
    TDL_CloseModel(tdl_handle, model_id_keypoint);
    TDL_CloseModel(tdl_handle, model_id_detect);
    DestoryCamera(tdl_handle);
    TDL_DestroyHandle(tdl_handle);
    return -1;
  }

  struct termios oldt, newt;
  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);

  printf("Press any key to exit...\n");

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
  TDL_CloseModel(tdl_handle, model_id_recognition);
  TDL_CloseModel(tdl_handle, model_id_keypoint);
  TDL_CloseModel(tdl_handle, model_id_detect);
  DestoryCamera(tdl_handle);
  TDL_DestroyHandle(tdl_handle);
  return ret;
}
