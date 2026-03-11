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
#include "meta_visualize.h"
#include "pthread_utils.h"
#include "sample_utils.h"
#include "tdl_sdk.h"

static uint32_t g_frame_id = 0;

int read_init_box(const char *path, int values[4]) {
  FILE *fp = fopen(path, "r");
  if (!fp) {
    perror("fopen");
    return -1;
  }

  char buf[256];
  if (!fgets(buf, sizeof(buf), fp)) {
    fclose(fp);
    fprintf(stderr, "read error or empty file\n");
    return -2;
  }
  fclose(fp);

  // 仅接受空白分隔的四个整数
  int count = sscanf(buf, "%d %d %d %d", &values[0], &values[1], &values[2],
                     &values[3]);
  if (count != 4) {
    fprintf(stderr, "parse error: expected 4 integers in one line\n");
    return -3;
  }
  return 0;
}

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf(
      "  %s -v <video_path> -m <sot_model_path> -t <txt_init_path> -s "
      "<save_dir>\n",
      prog_name);
  printf(
      "  %s --video_path <path> --sot_model_path <path> --txt_init_path <path> "
      "--save_dir <dir>\n",
      prog_name);
  printf("Options:\n");
  printf(
      "  -v, --video_path : video files supported by OpenCV\n"
      "  -m, --sot_model_path : sot model path\n"
      "  -t, --txt_init_path : one line, initial box info (int) like \"x1 x2 "
      "y1 y2\"\n"
      "  -s, --save_dir : save dir\n"
      "  -h, --help : print help\n");
}

int main(int argc, char *argv[]) {
  char *video_path = NULL;
  char *sot_model_path = NULL;
  char *txt_init_path = NULL;
  char *save_dir = NULL;
  int vi_chn = 0;

  struct option long_options[] = {
      {"video_path", required_argument, 0, 'v'},
      {"sot_model_path", required_argument, 0, 'm'},
      {"txt_init_path", required_argument, 0, 't'},
      {"save_dir", required_argument, 0, 's'},
      {"help", no_argument, 0, 'h'},
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "v:m:t:s:h", long_options, NULL)) !=
         -1) {
    switch (opt) {
      case 'v':
        video_path = optarg;
        break;
      case 'm':
        sot_model_path = optarg;
        break;
      case 't':
        txt_init_path = optarg;
        break;
      case 's':
        save_dir = optarg;
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

  if (!video_path || !sot_model_path || !txt_init_path || !save_dir) {
    fprintf(stderr,
            "Error: video_path, sot_model_path, txt_init_path, and save_dir "
            "are required\n");
    print_usage(argv[0]);
    return -1;
  }

  printf("Running with:\n");
  printf("video_path:     %s\n", video_path);
  printf("sot_model_path:    %s\n", sot_model_path);
  printf("txt_init_path:    %s\n", txt_init_path);
  printf("save_dir:    %s\n", save_dir);

  TDLHandle tdl_handle = TDL_CreateHandle(0);

  int ret;
  TDLModel sot_model_id = TDL_MODEL_TRACKING_FEARTRACK;
  ret = TDL_OpenModel(tdl_handle, sot_model_id, sot_model_path, NULL, 0);
  if (ret != 0) {
    printf("open sot model failed with %#x!\n", ret);
    goto exit1;
  }

  TDLObject obj_meta = {0};
  int values[4];
  ret = read_init_box(txt_init_path, values);
  if (ret != 0) {
    printf("read init box failed with %#x!\n", ret);
    goto exit1;
  }

  TDLImage image = GetVideoFrame(tdl_handle, video_path);

  if (image == NULL) {
    printf("get video frame failed with %#x!\n", ret);
    goto exit1;
  }

  ret = TDL_SetSingleObjectTracking(tdl_handle, image, &obj_meta, values, 4,
                                    TDL_REJECT);

  if (ret != 0) {
    printf("TDL_SetSingleObjectTracking failed with %#x!\n", ret);
    return -1;
  }
  TDL_DestroyImage(image);

  TDLTracker track_meta = {0};
  while (true) {
    TDLImage image = GetVideoFrame(tdl_handle, video_path);
    if (image) {
      g_frame_id++;

      if (g_frame_id % 10 == 0) {
        printf("processing frame %d\n", g_frame_id);
      }

      ret =
          TDL_SingleObjectTracking(tdl_handle, image, &track_meta, g_frame_id);

      if (ret != 0) {
        printf("TDL_SingleObjectTracking failed with %#x!\n", ret);
        return -1;
      }

      if (track_meta.info) {
        box_t boxes[1];
        boxes[0].x1 = track_meta.info[0].bbox.x1;
        boxes[0].y1 = track_meta.info[0].bbox.y1;
        boxes[0].x2 = track_meta.info[0].bbox.x2;
        boxes[0].y2 = track_meta.info[0].bbox.y2;

        char outpath[128];
        snprintf(outpath, 128, "%s/%07d.jpg", save_dir, g_frame_id);

        int colors[3] = {255, 0, 0};  // 蓝色
        if (VisualizeRectangle(boxes, 1, image, outpath, colors) != 0) {
          printf("VisualizeRectangle failed with %#x!\n", ret);
          return -1;
        }
      }

      TDL_DestroyImage(image);

    } else {
      printf("process done!\n");
      break;
    }
  }

exit1:
  TDL_CloseModel(tdl_handle, sot_model_id);

exit0:
  TDL_DestroyHandle(tdl_handle);

  return ret;
}
