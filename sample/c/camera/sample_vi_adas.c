#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/time.h>
#include <termios.h>
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
  printf("  %s --config_file <path> --vi_chn <int>\n", prog_name);
  printf("Options:\n");
  printf(
      "  -c, --config_file : json config file\n"
      "  -v, --vi_chn       : optional, default 0\n");
}

void *send_frame_thread(void *args) {
  printf("Enter send frame thread\n");
  SEND_FRAME_THREAD_ARG_S *pstArgs = (SEND_FRAME_THREAD_ARG_S *)args;

  uint64_t *channel_frame_id = malloc(pstArgs->channel_size * sizeof(uint64_t));
  int ret = 0;
  if (pstArgs->channel_size > 0 && channel_frame_id == NULL) {
    printf("malloc channel_frame_id failed\n");
    to_exit = true;
    ExitQueue(&image_queue);
    return NULL;
  }
  if (channel_frame_id && pstArgs->channel_size > 0) {
    memset(channel_frame_id, 0, pstArgs->channel_size * sizeof(uint64_t));
  }

  while (to_exit == false) {
    fd_set rfds;
    struct timeval tv = {0, 0};
    FD_ZERO(&rfds);
    FD_SET(STDIN_FILENO, &rfds);
    int key_pressed = select(STDIN_FILENO + 1, &rfds, NULL, NULL, &tv);
    if (key_pressed > 0 && FD_ISSET(STDIN_FILENO, &rfds)) {
      to_exit = true;
      ExitQueue(&image_queue);
      break;
    }

    for (size_t i = 0; i < pstArgs->channel_size; i++) {
      TDLImage image = NULL;

      if (Image_GetQueueSize(&image_queue) == 0) {
        image = GetCameraFrame(pstArgs->tdl_handle, pstArgs->vi_chn);
      } else {
        usleep(10000);
        continue;
      }
      if (image == NULL) {
        printf("GetCameraFrame failed\n");
        continue;
      }

      channel_frame_id[i] += 1;

      ret = TDL_APP_SetFrame(pstArgs->tdl_handle, pstArgs->channel_names[i],
                             image, channel_frame_id[i], 3);
      if (ret != 0) {
        printf("TDL_APP_SetFrame failed with %d\n", ret);
        ReleaseCameraFrame(pstArgs->tdl_handle, pstArgs->vi_chn);
        TDL_DestroyImage(image);
        continue;
      }

      ret = Image_Enqueue(&image_queue, image);
      if (ret != 0) {
        printf("Image_Enqueue failed\n");
        ReleaseCameraFrame(pstArgs->tdl_handle, pstArgs->vi_chn);
        TDL_DestroyImage(image);
        continue;
      }
    }
  }

  free(channel_frame_id);
  return NULL;
}

void *run_tdl_thread(void *args) {
  RUN_TDL_THREAD_ARG_S *pstArgs = (RUN_TDL_THREAD_ARG_S *)args;

  uint64_t counter = 0;
  uint64_t last_counter = 0;
  uint32_t last_time_ms = get_time_in_ms();

  static const char *state_str[] = {"NORMAL", "START", "WARNING"};

#ifdef ENABLE_RTSP
  VIDEO_FRAME_INFO_S *frame = NULL;

  RtspContext rtsp_context = {0};
  rtsp_context.chn = 0;
  rtsp_context.pay_load_type = PT_H264;
  rtsp_context.frame_width = VI_WIDTH;
  rtsp_context.frame_height = VI_HEIGHT;
#endif

  while (to_exit == false) {
    for (size_t i = 0; i < pstArgs->channel_size; i++) {
      TDLVehicleAdasInfo adas_info = {0};

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

      int ret = TDL_APP_VehicleAdas(pstArgs->tdl_handle,
                                    pstArgs->channel_names[i], &adas_info);

      if (ret == 1) {
        usleep(1000);
        continue;
      } else if (ret == 2) {
        to_exit = true;
        ExitQueue(&image_queue);
        break;
      } else if (ret != 0) {
        printf("TDL_APP_VehicleAdas failed with %#x!\n", ret);
        TDL_ReleaseVehicleAdasInfo(&adas_info);
        ReleaseCameraFrame(pstArgs->tdl_handle, pstArgs->vi_chn);
        TDLImage img = Image_Dequeue(&image_queue);
        if (img) {
          TDL_DestroyImage(img);
        }
        to_exit = true;
        ExitQueue(&image_queue);
        return NULL;
      }

      uint32_t obj_count = adas_info.adas_objects.size;

      if (frm_diff > 30) {
        printf("frame_id:%" PRIu64 " objects:%u lane_state:%d lanes:%u\n",
               adas_info.frame_id, obj_count, adas_info.lane_meta.lane_state,
               adas_info.lane_meta.size);
      }

      // // Print ADAS info

      // for (uint32_t j = 0; j < obj_count; j++) {
      //   TDLVehicleAdasObjectInfo *obj = &adas_info.adas_objects.info[j];
      //   printf(
      //       "  [%u] cls:%d bbox:%.1f %.1f %.1f %.1f dis:%.1fm speed:%.1fm/s "
      //       "state:%s\n",
      //       (unsigned int)obj->track_id, obj->class_id, obj->box.x1,
      //       obj->box.y1, obj->box.x2, obj->box.y2, obj->distance, obj->speed,
      //       state_str[obj->state]);
      // }
      // if (adas_info.lane_meta.lane_state == 1) {
      //   printf("  LANE DEPARTURE WARNING!\n");
      // }

#ifdef ENABLE_RTSP
      TDLImage image = adas_info.image;
      TDL_WrapImage(image, &frame);

      TDLBrush brush = {0};
      brush.size = 1;

      // Count normal vs warning objects
      uint32_t normal_count = 0;
      uint32_t warning_count = 0;
      for (uint32_t j = 0; j < obj_count; j++) {
        if (adas_info.adas_objects.info[j].state == 0) {
          normal_count++;
        } else {
          warning_count++;
        }
      }

      // Draw NORMAL objects (green)
      TDLObject normal_meta = {0};
      if (normal_count > 0) {
        normal_meta.size = normal_count;
        normal_meta.info =
            (TDLObjectInfo *)malloc(normal_count * sizeof(TDLObjectInfo));
        uint32_t idx = 0;
        for (uint32_t j = 0; j < obj_count; j++) {
          TDLVehicleAdasObjectInfo *obj = &adas_info.adas_objects.info[j];
          if (obj->state == 0) {
            TDLObjectInfo *info = &normal_meta.info[idx];
            memset(info, 0, sizeof(TDLObjectInfo));
            info->box = obj->box;
            info->track_id = obj->track_id;
            info->class_id = obj->class_id;
            info->score = obj->score;
            snprintf(info->name, sizeof(info->name), "[%d][%s]S:%.1f V:%.1f",
                     obj->class_id, state_str[obj->state], obj->distance,
                     obj->speed);
            idx++;
          }
        }
        brush.color.r = 0;
        brush.color.g = 255;
        brush.color.b = 0;
        DrawObjRect(&normal_meta, frame, true, brush);
        free(normal_meta.info);
      }

      // Draw START / COLLISION_WARNING objects (red)
      TDLObject warning_meta = {0};
      if (warning_count > 0) {
        warning_meta.size = warning_count;
        warning_meta.info =
            (TDLObjectInfo *)malloc(warning_count * sizeof(TDLObjectInfo));
        uint32_t idx = 0;
        for (uint32_t j = 0; j < obj_count; j++) {
          TDLVehicleAdasObjectInfo *obj = &adas_info.adas_objects.info[j];
          if (obj->state != 0) {
            TDLObjectInfo *info = &warning_meta.info[idx];
            memset(info, 0, sizeof(TDLObjectInfo));
            info->box = obj->box;
            info->track_id = obj->track_id;
            info->class_id = obj->class_id;
            info->score = obj->score;
            snprintf(info->name, sizeof(info->name), "[%d][%s]S:%.1f V:%.1f",
                     obj->class_id, state_str[obj->state], obj->distance,
                     obj->speed);
            idx++;
          }
        }
        brush.color.r = 0;
        brush.color.g = 0;
        brush.color.b = 255;
        DrawObjRect(&warning_meta, frame, true, brush);
        free(warning_meta.info);
      }

      // Draw lane lines (green)
      uint32_t lane_size = adas_info.lane_meta.size;
      if (lane_size > 0) {
        box_t *lane_boxes = (box_t *)malloc(lane_size * sizeof(box_t));
        for (uint32_t j = 0; j < lane_size; j++) {
          lane_boxes[j].x1 = adas_info.lane_meta.lines[j].x1;
          lane_boxes[j].y1 = adas_info.lane_meta.lines[j].y1;
          lane_boxes[j].x2 = adas_info.lane_meta.lines[j].x2;
          lane_boxes[j].y2 = adas_info.lane_meta.lines[j].y2;
        }
        brush.color.r = 0;
        brush.color.g = 255;
        brush.color.b = 0;
        DrawLine(lane_boxes, (int32_t)lane_size, frame, brush);
        free(lane_boxes);
      }

      // Draw lane state text (mid-lower area, matching C++ sample)
      int lane_text_x = VI_WIDTH * 0.4;
      int lane_text_y = VI_HEIGHT * 0.9;
      if (adas_info.lane_meta.lane_state == 1) {
        brush.color.r = 0;
        brush.color.g = 0;
        brush.color.b = 255;
        ObjectWriteText("LANE DEPARTURE WARNING!", lane_text_x, lane_text_y,
                        frame, brush);
      } else {
        brush.color.r = 0;
        brush.color.g = 255;
        brush.color.b = 0;
        ObjectWriteText("NORMAL", lane_text_x, lane_text_y, frame, brush);
      }

      ret = SendFrameRTSP(frame, &rtsp_context);
      if (ret != 0) {
        printf("SendFrameRTSP failed with %#x!\n", ret);
      }
#endif

      TDL_ReleaseVehicleAdasInfo(&adas_info);
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
}

int main(int argc, char *argv[]) {
  char *config_file = NULL;
  char *vi_chn = NULL;
  int chn = 0;
  bool termios_changed = false;
  TDLHandle tdl_handle = NULL;

  struct option long_options[] = {
      {"config_file", required_argument, 0, 'c'},
      {"vi_chn", required_argument, 0, 'v'},
      {"help", no_argument, 0, 'h'},
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "c:v:h", long_options, NULL)) != -1) {
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

  int ret = 0;

  if (!config_file) {
    fprintf(stderr, "Error: config_file is required\n");
    print_usage(argv[0]);
    return -1;
  }

  if (vi_chn) {
    chn = atoi(vi_chn);
  }

  printf("Running with:\n");
  printf("  config_file:    %s\n", config_file);
  printf("  vi_chn:         %d\n", chn);

  InitQueue(&image_queue);

  tdl_handle = TDL_CreateHandle(0);
  if (tdl_handle == NULL) {
    ret = -1;
    printf("TDL_CreateHandle failed\n");
    goto exit0;
  }

  char **channel_names = NULL;
  uint8_t channel_size = 0;

  ret = InitCamera(tdl_handle, VI_WIDTH, VI_HEIGHT, IMAGE_YUV420SP_UV, 3);
  if (ret != 0) {
    printf("InitCamera failed with %#x!\n", ret);
    goto exit0;
  }

  ret = TDL_APP_Init(tdl_handle, "vehicle_adas", config_file, &channel_names,
                     &channel_size, false);
  if (ret != 0) {
    printf("TDL_APP_Init failed with %#x!\n", ret);
    goto exit1;
  }

  for (int i = 0; i < channel_size; i++) {
    printf("channel[%d]: %s\n", i, channel_names[i]);
  }

  // Set terminal to non-canonical mode
  struct termios oldt, newt;
  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  termios_changed = true;

  printf("Press any key to exit...\n");

  pthread_t stFrameThread, stTDLThread;
  bool frame_thread_created = false;
  bool tdl_thread_created = false;

  SEND_FRAME_THREAD_ARG_S frame_args = {.tdl_handle = tdl_handle,
                                        .vi_chn = chn,
                                        .channel_size = channel_size,
                                        .channel_names = channel_names};

  RUN_TDL_THREAD_ARG_S tdl_args = {.tdl_handle = tdl_handle,
                                   .vi_chn = chn,
                                   .channel_size = channel_size,
                                   .channel_names = channel_names};

  ret = pthread_create(&stFrameThread, NULL, send_frame_thread, &frame_args);
  if (ret != 0) {
    printf("pthread_create send_frame_thread failed with %d\n", ret);
    goto exit3;
  }
  frame_thread_created = true;

  ret = pthread_create(&stTDLThread, NULL, run_tdl_thread, &tdl_args);
  if (ret != 0) {
    printf("pthread_create run_tdl_thread failed with %d\n", ret);
    to_exit = true;
    ExitQueue(&image_queue);
    goto exit3;
  }
  tdl_thread_created = true;

exit3:
  if (frame_thread_created) {
    pthread_join(stFrameThread, NULL);
  }
  if (tdl_thread_created) {
    pthread_join(stTDLThread, NULL);
  }
  if (termios_changed) {
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  }

exit2:
  for (int i = 0; i < channel_size; i++) {
    free(channel_names[i]);
  }
  free(channel_names);

exit1:
  if (tdl_handle) {
    DestoryCamera(tdl_handle);
  }

exit0:
  DestroyQueue(&image_queue);
  if (tdl_handle) {
    TDL_DestroyHandle(tdl_handle);
  }

  return ret;
}
