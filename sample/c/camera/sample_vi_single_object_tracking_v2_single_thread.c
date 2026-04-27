#include <fcntl.h>
#include <getopt.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/time.h>
#include <termios.h>

static bool to_exit = false;

static void handle_signal(int signal) {
  if (signal == SIGINT) {
    to_exit = true;
  }
}
#include <time.h>
#include <unistd.h>
#include "cvi_comm_video.h"
#include "cvi_vi.h"
#include "meta_visualize.h"
#include "sample_utils.h"
#include "tdl_sdk.h"

#ifndef __CV184X__
#define ENABLE_RTSP
#endif

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

TDLObject g_obj_meta = {0};
static uint32_t g_frame_id = 0;
TDLBrush brush_green = {{0, 255, 0}, 2};
TDLBrush brush_yellow = {{255, 255, 0}, 2};

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

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf(
      "  %s -d <det_model_path> -s <sot_model_path> -t <target_search_type> "
      "[-f <sam_model_path>]\n",
      prog_name);
  printf("Options:\n");
  printf(
      "  -d, --det_model_path : person vehicle detection model path\n"
      "  -s, --sot_model_path : sot model path\n"
      "  -t, --target_search_type : target search method type (0-3)\n"
      "  -f, --sam_model_path : FastSAM segment model path\n"
      "  -h, --help : print help\n");
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
      default:
        print_usage(argv[0]);
        return -1;
    }
  }

  if (!det_model_path || !sot_model_path || target_search_type < 0) {
    print_usage(argv[0]);
    return -1;
  }

  TDLHandle tdl_handle = TDL_CreateHandle(0);
  int ret = InitCamera(tdl_handle, VI_WIDTH, VI_HEIGHT, IMAGE_YUV420SP_UV, 3);
  if (ret != 0) {
    printf("InitCamera failed!\n");
    return -1;
  }

  TDLModel det_model_id = TDL_MODEL_YOLOV8N_DET_PERSON_VEHICLE;
  ret = TDL_OpenModel(tdl_handle, det_model_id, det_model_path, NULL, 0);
  if (ret != 0) {
    printf("open detection model failed!\n");
    return -1;
  }

  TDLModel sot_model_id = TDL_MODEL_TRACKING_FEARTRACK;
  ret = TDL_OpenModel(tdl_handle, sot_model_id, sot_model_path, NULL, 0);
  if (ret != 0) {
    printf("open sot model failed!\n");
    return -1;
  }

  init_selected_area();
  set_non_blocking_input();
  signal(SIGINT, handle_signal);

  TDLTracker track_meta = {0};
  TDLObject track_obj_meta = {0};
  TDL_InitObjectMeta(&track_obj_meta, 1, 0);
  TDLObject det_obj_meta = {0};

  printf("Usage: input i or I to start tracking, q or Q to exit.\n");

  while (!to_exit) {
    char key = check_key_input();
    if (key == 'q' || key == 'Q') {
      to_exit = true;
      break;
    }

    TDLImage image = GetCameraFrame(tdl_handle, vi_chn);
    if (image == NULL) {
      usleep(10 * 1000);
      continue;
    }
    g_frame_id++;

    VIDEO_FRAME_INFO_S *frame = NULL;
    TDL_WrapImage(image, &frame);

    if (g_status == DETECTION) {
      ret = TDL_Detection(tdl_handle, det_model_id, image, &det_obj_meta);
      if (ret == 0) {
        printf("det_obj_meta: %d\n", det_obj_meta.size);

        TDL_CopyObjectMeta(&det_obj_meta, &g_obj_meta);
        int closest_idx = find_closest_box_in_selection(&det_obj_meta);
        printf("g_obj_meta: %d closest_idx: %d\n", g_obj_meta.size,
               closest_idx);
        bool has_target = (closest_idx >= 0);

#ifdef ENABLE_RTSP
        TDLBrush brush_green_det = {{0, 255, 0}, 5};
        TDLBrush brush_yellow_det = {{255, 255, 0}, 5};
        for (int i = 0; i < det_obj_meta.size; i++) {
          TDLObject single_obj = {0};
          TDL_InitObjectMeta(&single_obj, 1, 0);
          single_obj.info[0] = det_obj_meta.info[i];
          DrawObjRect(&single_obj, frame, true,
                      (i == closest_idx) ? brush_yellow_det : brush_green_det);
          TDL_ReleaseObjectMeta(&single_obj);
        }
        draw_selection_rect(frame, has_target ? brush_yellow : brush_green);
        draw_center_cross(frame, has_target ? brush_yellow : brush_green);
#endif

        if (key == 'i' || key == 'I') {
          int values[4] = {(int)BOX_LEFT, (int)BOX_TOP, (int)BOX_RIGHT,
                           (int)BOX_BOTTOM};
          const char *model_path =
              (target_search_type == TDL_FASTSAM) ? sam_model_path : NULL;
          ret = TDL_SetSingleObjectTracking(tdl_handle, image, &g_obj_meta,
                                            values, 4, g_frame_id,
                                            target_search_type, model_path);
          if (ret == 0) {
            g_status = TRACKING;
            g_lost_timer_started = false;
            printf("Switching to TRACKING mode\n");
          }
        }
        TDL_ReleaseObjectMeta(&det_obj_meta);
      }
    } else if (g_status == TRACKING) {
      ret =
          TDL_SingleObjectTracking(tdl_handle, image, &track_meta, g_frame_id);
      if (track_meta.info && track_meta.info[0].score > 0.3) {
#ifdef ENABLE_RTSP
        TDLBrush brush_track = {{255, 0, 0}, 5};
        track_obj_meta.info[0].box.x1 = track_meta.info[0].bbox.x1;
        track_obj_meta.info[0].box.y1 = track_meta.info[0].bbox.y1;
        track_obj_meta.info[0].box.x2 = track_meta.info[0].bbox.x2;
        track_obj_meta.info[0].box.y2 = track_meta.info[0].bbox.y2;
        DrawObjRect(&track_obj_meta, frame, true, brush_track);
        draw_selection_rect(frame, brush_yellow);
        draw_center_cross(frame, brush_yellow);
#endif
        // 每隔30帧打印跟踪框信息
        if (g_frame_id % 30 == 0) {
          printf("Frame %u: Track bbox=[%.1f,%.1f,%.1f,%.1f]\n", g_frame_id,
                 track_meta.info[0].bbox.x1, track_meta.info[0].bbox.y1,
                 track_meta.info[0].bbox.x2, track_meta.info[0].bbox.y2);
        }
        g_lost_timer_started = false;
      } else {
        if (!g_lost_timer_started) {
          g_lost_start_time = get_time_in_ms();
          g_lost_timer_started = true;
          printf("Target lost at frame %u\n", g_frame_id);
        } else if (get_time_in_ms() - g_lost_start_time >=
                   LOST_TIMEOUT_SECONDS * 1000) {
          g_status = DETECTION;
          g_lost_timer_started = false;
          printf("Lost timeout, switching to DETECTION mode\n");
        }
#ifdef ENABLE_RTSP
        draw_selection_rect(frame, brush_yellow);
        draw_center_cross(frame, brush_yellow);
#endif
      }
      TDL_ReleaseTrackMeta(&track_meta);
    }

#ifdef ENABLE_RTSP
    SendFrameRTSP(frame, &rtsp_context);
#endif
    TDL_DestroyImage(image);
    ReleaseCameraFrame(tdl_handle, vi_chn);
  }

  restoreTerminal();
  TDL_ReleaseObjectMeta(&track_obj_meta);
  TDL_CloseModel(tdl_handle, det_model_id);
  TDL_CloseModel(tdl_handle, sot_model_id);
  DestoryCamera(tdl_handle);
  TDL_DestroyHandle(tdl_handle);

  return 0;
}
