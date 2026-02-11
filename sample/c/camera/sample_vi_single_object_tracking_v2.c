#include <fcntl.h>
#include <getopt.h>
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

// 预选区配置
#define SELECTION_BOX_SIZE 200
#define CENTER_X (VI_WIDTH / 2)
#define CENTER_Y (VI_HEIGHT / 2)
#define BOX_LEFT (CENTER_X - SELECTION_BOX_SIZE / 2)
#define BOX_TOP (CENTER_Y - SELECTION_BOX_SIZE / 2)
#define BOX_RIGHT (CENTER_X + SELECTION_BOX_SIZE / 2)
#define BOX_BOTTOM (CENTER_Y + SELECTION_BOX_SIZE / 2)

// 颜色定义
#define GREEN_R 0
#define GREEN_G 255
#define GREEN_B 0
#define YELLOW_R 255
#define YELLOW_G 255
#define YELLOW_B 0

TDLObject p_selected_area = {0};
void init_selected_area() {
  // 初始化对象元数据，创建一个包含1个对象的预选区
  TDL_InitObjectMeta(&p_selected_area, 1, 0);

  // 设置预选区的边界框坐标
  p_selected_area.info[0].box.x1 = BOX_LEFT;
  p_selected_area.info[0].box.y1 = BOX_TOP;
  p_selected_area.info[0].box.x2 = BOX_RIGHT;
  p_selected_area.info[0].box.y2 = BOX_BOTTOM;

  // 设置预选区的类别和置信度
  p_selected_area.info[0].class_id = 0;
  p_selected_area.info[0].score = 1.0f;
}

bool init_success = false;
bool just_initialized = false;  // 标记是否刚刚初始化，用于绘制第一帧的框选结果
static volatile bool to_exit = false;
static ImageQueue image_queue;
TDLObject g_obj_meta = {0};
static uint32_t g_frame_id = 0;
TDLBrush brush_green = {{0, 255, 0}, 2};
TDLBrush brush_yellow = {{255, 255, 0}, 2};
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

// 检查检测框中心点是否在预选区内
static bool is_box_in_selection_area(const TDLObjectInfo *obj_info) {
  float box_cx = (obj_info->box.x1 + obj_info->box.x2) / 2.0f;
  float box_cy = (obj_info->box.y1 + obj_info->box.y2) / 2.0f;
  return (box_cx >= BOX_LEFT && box_cx <= BOX_RIGHT && box_cy >= BOX_TOP &&
          box_cy <= BOX_BOTTOM);
}

// 计算检测框中心点到预选区中心的距离平方
static float get_distance_sq_to_center(const TDLObjectInfo *obj_info) {
  float box_cx = (obj_info->box.x1 + obj_info->box.x2) / 2.0f;
  float box_cy = (obj_info->box.y1 + obj_info->box.y2) / 2.0f;
  float dx = box_cx - CENTER_X;
  float dy = box_cy - CENTER_Y;
  return dx * dx + dy * dy;
}

// 找到预选区内最接近中心的目标
static int find_closest_box_in_selection(const TDLObject *obj_meta) {
  int closest_idx = -1;
  float min_distance_sq = 1e10f;

  for (int i = 0; i < obj_meta->size; i++) {
    if (is_box_in_selection_area(&obj_meta->info[i])) {
      float dist_sq = get_distance_sq_to_center(&obj_meta->info[i]);
      if (dist_sq < min_distance_sq) {
        min_distance_sq = dist_sq;
        closest_idx = i;
      }
    }
  }
  return closest_idx;
}

// 绘制中心+号
static void draw_center_cross(VIDEO_FRAME_INFO_S *frame, TDLBrush brush) {
  const int cross_size = 20;      // +号大小
  const int cross_thickness = 3;  // +号线宽

  // 创建横线
  TDLObject cross_h = {0};
  TDL_InitObjectMeta(&cross_h, 1, 0);
  cross_h.info[0].box.x1 = CENTER_X - cross_size / 2;
  cross_h.info[0].box.y1 = CENTER_Y - cross_thickness / 2;
  cross_h.info[0].box.x2 = CENTER_X + cross_size / 2;
  cross_h.info[0].box.y2 = CENTER_Y + cross_thickness / 2;
  DrawObjRect(&cross_h, frame, false, brush);
  TDL_ReleaseObjectMeta(&cross_h);

  // 创建竖线
  TDLObject cross_v = {0};
  TDL_InitObjectMeta(&cross_v, 1, 0);
  cross_v.info[0].box.x1 = CENTER_X - cross_thickness / 2;
  cross_v.info[0].box.y1 = CENTER_Y - cross_size / 2;
  cross_v.info[0].box.x2 = CENTER_X + cross_thickness / 2;
  cross_v.info[0].box.y2 = CENTER_Y + cross_size / 2;
  DrawObjRect(&cross_v, frame, false, brush);
  TDL_ReleaseObjectMeta(&cross_v);
}

// 绘制预选区实线框
static void draw_selection_rect(VIDEO_FRAME_INFO_S *frame, TDLBrush brush) {
  DrawObjRect(&p_selected_area, frame, false, brush);
}

typedef struct {
  TDLHandle tdl_handle;
  int vi_chn;
  TDLModel det_model_id;
  TDLModel sot_model_id;
  const char *sam_model_path;
  TDLTargetSearchTypeE target_search_type;
} THREAD_ARG_S;

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf(
      "  %s -d <det_model_path> -s <sot_model_path> -t <target_search_type> "
      "[-f <sam_model_path>]\n",
      prog_name);
  printf(
      "  %s --det_model_path <path> --sot_model_path <path> "
      "--target_search_type <type> [--sam_model_path <path>]\n",
      prog_name);
  printf("Options:\n");
  printf(
      "  -d, --det_model_path : person vehicle detection model path\n"
      "  -s, --sot_model_path : sot model path\n"
      "  -t, --target_search_type : target search method type (0-3, see "
      "below)\n"
      "  -f, --sam_model_path : FastSAM segment model path (required when "
      "target_search_type=3)\n"
      "  -h, --help : print help\n");
  printf("\nTarget Search Methods (target_search_type):\n");
  printf("  0 (TDL_REJECT): 不使用框选方法\n");
  printf("  1 (TDL_GRABCUT): 基于GrabCut框选方法（耗时高）\n");
  printf("  2 (TDL_COLOR): 基于颜色阈值框选方法\n");
  printf("  3 (TDL_FASTSAM): 基于FastSAM框选方法（需要传模型路径）\n");
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
  init_selected_area();
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

        // 找到预选区内最接近中心的目标
        int closest_idx = find_closest_box_in_selection(&obj_meta);
        bool has_target_in_selection = (closest_idx >= 0);

        // 绘制检测框：预选区内最接近中心的目标用黄色，其他用绿色
        TDLBrush brush_green_det = {{0, 255, 0}, 5};
        TDLBrush brush_yellow_det = {{255, 255, 0}, 5};

        for (int i = 0; i < obj_meta.size; i++) {
          TDLObjectInfo *obj_info = &obj_meta.info[i];
          snprintf(obj_info->name, sizeof(obj_info->name), "index:%d", i);

          // 创建单个对象的TDLObject用于绘制
          TDLObject single_obj = {0};
          TDL_InitObjectMeta(&single_obj, 1, 0);
          single_obj.info[0] = *obj_info;

          // 如果是最接近中心的目标，用黄色，否则用绿色
          if (i == closest_idx) {
            DrawObjRect(&single_obj, frame, true, brush_yellow_det);
          } else {
            DrawObjRect(&single_obj, frame, true, brush_green_det);
          }

          TDL_ReleaseObjectMeta(&single_obj);
        }

        // 绘制预选区实线框：如果有目标在预选区内，用黄色，否则用绿色
        if (has_target_in_selection) {
          draw_selection_rect(frame, brush_yellow);
          // 绘制中心+号（黄色）
          draw_center_cross(frame, brush_yellow);
        } else {
          draw_selection_rect(frame, brush_green);
          // 绘制中心+号（绿色）
          draw_center_cross(frame, brush_green);
        }

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

  printf("Usage: input i or I to start tracking (use pre-selection box).\n");
  while (to_exit == false) {
    char key = check_key_input();
    if (key == 'i' || key == 'I') {
      /* 固定使用预选框坐标，不再从用户输入读取 */
      int values[4] = {
          (int)p_selected_area.info[0].box.x1,
          (int)p_selected_area.info[0].box.y1,
          (int)p_selected_area.info[0].box.x2,
          (int)p_selected_area.info[0].box.y2,
      };
      const int num_values = 4;

      {
        MutexAutoLock(ResultMutex, lock);
        image = Image_Dequeue(&image_queue);
      }
      if (image) {
        const char *model_path = (pstArgs->target_search_type == TDL_FASTSAM)
                                     ? pstArgs->sam_model_path
                                     : NULL;
        int ret = TDL_SetSingleObjectTracking(
            pstArgs->tdl_handle, image, &g_obj_meta, values, num_values,
            pstArgs->target_search_type, model_path);

        if (ret != 0) {
          init_success = false;
          printf("TDL_SetSingleObjectTracking failed: %#x\n", ret);
          TDL_DestroyImage(image);
          ReleaseCameraFrame(pstArgs->tdl_handle, pstArgs->vi_chn);
        } else {
          init_success = true;
          g_status = TRACKING;
          g_lost_timer_started = false;  // 重置丢失计时器

          // 初始化成功后，立即用同一帧图像调用一次跟踪来获取框选结果
          TDLTracker init_track_meta = {0};
          int track_ret = TDL_SingleObjectTracking(
              pstArgs->tdl_handle, image, &init_track_meta, g_frame_id);

          TDL_WrapImage(image, &frame);
          if (frame) {
            if (track_ret == 0 && init_track_meta.info) {
              // 成功获取到跟踪结果，绘制框选算法的结果
              printf("Got initial track result: bbox=[%.1f,%.1f,%.1f,%.1f]\n",
                     init_track_meta.info[0].bbox.x1,
                     init_track_meta.info[0].bbox.y1,
                     init_track_meta.info[0].bbox.x2,
                     init_track_meta.info[0].bbox.y2);

              TDLBrush brush_init_track = {0};
              brush_init_track.size = 5;
              brush_init_track.color.r = 255;
              brush_init_track.color.g = 0;
              brush_init_track.color.b = 0;

              TDLObject init_track_obj = {0};
              TDL_InitObjectMeta(&init_track_obj, 1, 0);
              init_track_obj.info[0].box.x1 = init_track_meta.info[0].bbox.x1;
              init_track_obj.info[0].box.y1 = init_track_meta.info[0].bbox.y1;
              init_track_obj.info[0].box.x2 = init_track_meta.info[0].bbox.x2;
              init_track_obj.info[0].box.y2 = init_track_meta.info[0].bbox.y2;

              DrawObjRect(&init_track_obj, frame, true, brush_init_track);

              TDL_ReleaseObjectMeta(&init_track_obj);
            } else {
              printf(
                  "Initial track call failed: track_ret=%d, info=%p, will "
                  "retry next frame\n",
                  track_ret, init_track_meta.info);
            }

            // 无论是否有跟踪结果，都绘制预选区框和+号
            draw_selection_rect(frame, brush_yellow);
            draw_center_cross(frame, brush_yellow);

            SendFrameRTSP(frame, &rtsp_context);
          }

          TDL_ReleaseTrackMeta(&init_track_meta);

          // 释放图像
          TDL_DestroyImage(image);
          ReleaseCameraFrame(pstArgs->tdl_handle, pstArgs->vi_chn);
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

        // 如果刚刚初始化，记录状态
        if (just_initialized) {
          just_initialized = false;
          printf("First frame after init: track_ret=%d, track_meta.info=%p\n",
                 ret, track_meta.info);
          if (track_meta.info) {
            printf("Got track result: bbox=[%.1f,%.1f,%.1f,%.1f]\n",
                   track_meta.info[0].bbox.x1, track_meta.info[0].bbox.y1,
                   track_meta.info[0].bbox.x2, track_meta.info[0].bbox.y2);
          } else {
            printf("No track result in first frame, status may be LOST\n");
          }
        }

        if (track_meta.info) {
          // 绘制跟踪框
          TDLBrush brush_track = {0};
          brush_track.size = 5;
          brush_track.color.r = 255;
          brush_track.color.g = 0;
          brush_track.color.b = 0;

          track_obj_meta.info[0].box.x1 = track_meta.info[0].bbox.x1;
          track_obj_meta.info[0].box.y1 = track_meta.info[0].bbox.y1;
          track_obj_meta.info[0].box.x2 = track_meta.info[0].bbox.x2;
          track_obj_meta.info[0].box.y2 = track_meta.info[0].bbox.y2;

          DrawObjRect(&track_obj_meta, frame, true, brush_track);

          // 绘制预选区实线框
          draw_selection_rect(frame, brush_yellow);
          // 绘制中心+号
          draw_center_cross(frame, brush_yellow);

          g_lost_timer_started = false;
        } else {
          // Start or continue lost timing
          if (!g_lost_timer_started) {
            g_lost_start_time = get_time_in_ms();
            g_lost_timer_started = true;
            printf("Target lost detected at frame %u\n", g_frame_id);
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
              init_success = false;
            }
          }
          // 即使目标丢失，也绘制预选区实线框和+号
          draw_selection_rect(frame, brush_yellow);
          draw_center_cross(frame, brush_yellow);
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
  char *sam_model_path = NULL;
  int target_search_type = -1;
  int vi_chn = 0;

  struct option long_options[] = {
      {"det_model_path", required_argument, 0, 'd'},
      {"sot_model_path", required_argument, 0, 's'},
      {"target_search_type", required_argument, 0, 't'},
      {"sam_model_path", required_argument, 0, 'f'},
      {"help", no_argument, 0, 'h'},
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "d:s:t:f:h", long_options, NULL)) !=
         -1) {
    switch (opt) {
      case 'd':
        det_model_path = optarg;
        break;
      case 's':
        sot_model_path = optarg;
        break;
      case 't':
        target_search_type = atoi(optarg);
        break;
      case 'f':
        sam_model_path = optarg;
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
  if (target_search_type < 0 || target_search_type > 3) {
    fprintf(stderr, "Error: target_search_type is required and must be 0-3\n");
    print_usage(argv[0]);
    return -1;
  }
  if (target_search_type == TDL_FASTSAM && !sam_model_path) {
    fprintf(stderr,
            "Error: sam_model_path is required when target_search_type=3 "
            "(TDL_FASTSAM)\n");
    print_usage(argv[0]);
    return -1;
  }

  printf("Running with:\n");
  printf("det_model_path:    %s\n", det_model_path);
  printf("sot_model_path:    %s\n", sot_model_path);
  printf("target_search_type: %d\n", target_search_type);
  if (sam_model_path) {
    printf("sam_model_path: %s\n", sam_model_path);
  }
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
  ret = TDL_OpenModel(tdl_handle, det_model_id, det_model_path, NULL, 0);
  if (ret != 0) {
    printf("open detection model failed with %#x!\n", ret);
    goto exit1;
  }

  TDLModel sot_model_id = TDL_MODEL_TRACKING_FEARTRACK;
  ret = TDL_OpenModel(tdl_handle, sot_model_id, sot_model_path, NULL, 0);
  if (ret != 0) {
    printf("open sot model failed with %#x!\n", ret);
    goto exit2;
  }

  set_non_blocking_input();

  pthread_t stFrameThread, stDetThread, stSotThread;

  THREAD_ARG_S total_args = {
      .tdl_handle = tdl_handle,
      .det_model_id = det_model_id,
      .sot_model_id = sot_model_id,
      .vi_chn = vi_chn,
      .sam_model_path = sam_model_path,
      .target_search_type = (TDLTargetSearchTypeE)target_search_type};

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
