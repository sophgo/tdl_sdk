#include <ctype.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "meta_visualize.h"
#include "tdl_sdk.h"
#include "tdl_utils.h"

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf("  %s -c <config_file> -o <output_dir>\n", prog_name);
  printf("  %s --config_file <path> --output_dir <dir>\n", prog_name);
  printf("Options:\n");
  printf(
      "  -c, --config_file : json config file\n"
      "  -o, --output_dir : output dir to save snapshot\n");
}

int main(int argc, char *argv[]) {
  char *config_file = NULL;  // sample/config/face_pet_cap_app.json
  char *gallery_dir = NULL;
  char *output_dir = NULL;

  struct option long_options[] = {
      {"config_file", required_argument, 0, 'c'},
      {"output_dir", required_argument, 0, 'o'},
      {"help", no_argument, 0, 'h'},
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "c:o:h", long_options, NULL)) != -1) {
    switch (opt) {
      case 'c':
        config_file = optarg;
        break;
      case 'o':
        output_dir = optarg;
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

  if (!config_file || !output_dir) {
    fprintf(stderr, "Error: config_file and output_dir are required\n");
    print_usage(argv[0]);
    return -1;
  }

  printf("Running with:\n");
  printf("  config_file:    %s\n", config_file);
  printf("  output_dir:  %s\n", output_dir);

  TDLHandle tdl_handle = TDL_CreateHandle(0);

  char **channel_names = NULL;
  uint8_t channel_size = 0;
  int ret = TDL_APP_Init(tdl_handle, "human_pose_smooth", config_file,
                         &channel_names, &channel_size);
  if (ret != 0) {
    printf("TDL_APP_Init failed with %#x!\n", ret);
    goto exit1;
  }

  bool to_exit = false;
  while (true) {
    for (size_t i = 0; i < channel_size; i++) {
      TDLCaptureInfo capture_info = {0};
      ret =
          TDL_APP_HumanPoseSmooth(tdl_handle, channel_names[i], &capture_info);
      if (ret == 1) {
        continue;
      } else if (ret == 2) {
        to_exit = true;
        break;
      } else if (ret != 0) {
        printf("TDL_APP_Capture failed with %#x!\n", ret);
        goto exit1;
      }

      box_t boxes[capture_info.person_meta.size];
      printf("person_meta.size: %d\n", capture_info.person_meta.size);

      point_t point[capture_info.person_meta.size * 17];
      for (int i = 0; i < capture_info.person_meta.size; i++) {
        for (int j = 0; j < 17; j++) {
          if (capture_info.person_meta.info[i].landmark_properity[j].score <
              0.5)
            continue;
          point[i * 17 + j].x =
              capture_info.person_meta.info[i].landmark_properity[j].x;
          point[i * 17 + j].y =
              capture_info.person_meta.info[i].landmark_properity[j].y;
        }
      }
      char outpath[128];
      snprintf(outpath, 128, "%s/%07d.jpg", output_dir, capture_info.frame_id);
      ret = VisualizePoint(point, capture_info.person_meta.size * 17,
                           capture_info.image, outpath);
      if (ret != 0) {
        printf("VisualizeRectangle failed with %#x!\n", ret);
        goto exit1;
      }

      TDL_ReleaseCaptureInfo(&capture_info);
    }

    if (to_exit) {
      break;
    }
  }

exit1:
  for (int i = 0; i < channel_size; i++) {
    free(channel_names[i]);
  }
  free(channel_names);

exit0:
  TDL_DestroyHandle(tdl_handle);
}