#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#include "meta_visualize.h"
#include "rtsp_utils.h"
#include "tdl_sdk.h"

#define VI_WIDTH 640
#define VI_HEIGHT 360

int get_model_info(char *model_path, TDLModel *model_index) {
  int ret = 0;
  if (strstr(model_path, "yolov8n_det_person_vehicle") != NULL) {
    *model_index = TDL_MODEL_YOLOV8N_DET_PERSON_VEHICLE;
  } else if (strstr(model_path, "yolov8n_det_hand_384_640") != NULL) {
    *model_index = TDL_MODEL_YOLOV8N_DET_HAND;
  } else {
    ret = -1;
  }
  return ret;
}

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf("  %s -m <model_path> -c <vi_chn>\n", prog_name);
  printf("  %s --model_path <path> --chn <vi_chn>\n\n", prog_name);
  printf("Options:\n");
  printf(
      "  -m, --model_path  Path to cvimodel eg. "
      "<yolov8n_det_person_vehicle>\n");
  printf("  -c, --chn         Vi chn\n");
  printf("  -h, --help        Show this help message\n");
}

int main(int argc, char *argv[]) {
  char *model_path = NULL;
  int vi_chn = 0;
  int rtsp_chn = 0;

  struct option long_options[] = {{"model_path", required_argument, 0, 'm'},
                                  {"pipe", required_argument, 0, 'c'},
                                  {"help", no_argument, 0, 'h'},
                                  {NULL, 0, NULL, 0}};

  int opt;
  while ((opt = getopt_long(argc, argv, "m:c:t:h", long_options, NULL)) != -1) {
    switch (opt) {
      case 'm':
        model_path = optarg;
        break;
      case 'c':
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

  // 检查必需参数
  if (!model_path) {
    fprintf(stderr, "Error: Model path is required\n");
    print_usage(argv[0]);
    return -1;
  }

  printf("Running with:\n");
  printf("  Model path:    %s\n", model_path);
  printf("  vi_chn:        %d\n", vi_chn);
  printf("  rtsp_chn:      %d\n", rtsp_chn);

  int ret = 0;
  TDLImage image = NULL;
  TDLObject obj_meta = {0};
  TDLModel model_id;
  TDLHandle tdl_handle = TDL_CreateHandle(vi_chn);

  if (get_model_info(model_path, &model_id) == -1) {
    printf("unsupported model: %s\n", model_path);
    return -1;
  }

  ret = TDL_InitCamera(tdl_handle);
  if (ret != 0) {
    printf("TDL_InitCamera %#x!\n", ret);
    TDL_DestroyHandle(tdl_handle);
    return ret;
  }

  // system("cat /proc/soph/vi_dbg");
  // system("cat /proc/soph/mipi-rx");
  // system("cat /proc/soph/vi");

  ret = TDL_OpenModel(tdl_handle, model_id, model_path, NULL);
  if (ret != 0) {
    printf("open model failed with %#x!\n", ret);
    TDL_DestoryCamera(tdl_handle);
    TDL_DestroyHandle(tdl_handle);
    return ret;
  }

  // 初始化RTSP参数
  TDLRTSPContext rtsp_context = {0};
  rtsp_context.chn = rtsp_chn;
  rtsp_context.pay_load_type = PT_H264;
  rtsp_context.frame_width = VI_WIDTH;
  rtsp_context.frame_height = VI_HEIGHT;

  // 设置终端为非规范模式
  struct termios oldt, newt;
  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);

  VIDEO_FRAME_INFO_S frame = {0};
  printf("按任意键退出...\n");
  int is_rtsp_running = 0;

  while (1) {
    // 检查键盘输入
    fd_set rfds;
    struct timeval tv = {0, 0};
    FD_ZERO(&rfds);
    FD_SET(STDIN_FILENO, &rfds);
    int key_pressed = select(STDIN_FILENO + 1, &rfds, NULL, NULL, &tv);
    if (key_pressed > 0 && FD_ISSET(STDIN_FILENO, &rfds)) {
      break;  // 有键盘输入，退出循环
    }

    image = TDL_GetCameraFrame(tdl_handle, vi_chn);
    if (image == NULL) {
      printf("TDL_GetViFrame failed\n");
      continue;
    }

    TDL_WrapImage(image, &frame);

    // 执行目标检测
    ret = TDL_Detection(tdl_handle, model_id, image, &obj_meta);
    if (ret != 0) {
      printf("TDL_Detection failed with %#x!\n", ret);
    } else {
      // 为每个检测到的目标画框
      for (int i = 0; i < obj_meta.size; i++) {
        TDLBrush brush = {0};
        brush.color.r = 255;
        brush.color.g = 0;
        brush.color.b = 0;
        brush.size = 5;

        // 创建TDLObject结构
        TDLObjectInfo obj_info = {0};
        obj_info.box = obj_meta.info[i].box;
        snprintf(obj_info.name, sizeof(obj_info.name), "class:%d score:%.2f",
                 obj_meta.info[i].class_id, obj_meta.info[i].score);

        TDLObject meta = {0};
        meta.info = &obj_info;
        meta.size = 1;

        // 使用TDL_DrawRect绘制
        TDL_DrawRect(&meta, &frame, true, brush);
        for (int i = 0; i < obj_meta.size; i++) {
          printf("obj_meta_index : %d, ", i);
          printf("class_id : %d, ", obj_meta.info[i].class_id);
          printf("score : %f, ", obj_meta.info[i].score);
          printf("bbox : [%f %f %f %f]\n", obj_meta.info[i].box.x1,
                 obj_meta.info[i].box.x2, obj_meta.info[i].box.y1,
                 obj_meta.info[i].box.y2);
        }
      }
    }

    // 发送帧
    ret = TDL_SendFrameRTSP(&frame, &rtsp_context);
    if (ret != 0) {
      printf("TDL_SendFrameRTSP failed with %#x!\n", ret);
      continue;
    }
    if (is_rtsp_running == 0) {
      is_rtsp_running = 1;
      printf("rtsp connected!\n");
    }

    TDL_ReleaseObjectMeta(&obj_meta);
    TDL_ReleaseCameraFrame(tdl_handle, vi_chn);
    TDL_DestroyImage(image);
    usleep(40 * 1000);  // Match Vi Frame Rate
  }

  // 恢复终端设置
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  printf("rtsp disconnected!\n");
  TDL_CloseModel(tdl_handle, model_id);
  TDL_DestoryCamera(tdl_handle);
  TDL_DestroyHandle(tdl_handle);
  return ret;
}
