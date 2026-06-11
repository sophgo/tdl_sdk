#include <ctype.h>
#include <dirent.h>
#include <inttypes.h>
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

static const char *state_str[] = {"NORMAL", "START", "WARNING"};

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf("  %s -c <config_file> -o <output_dir>\n", prog_name);
  printf("  %s --config_file <path> --output_dir <dir>\n", prog_name);
  printf("Options:\n");
  printf(
      "  -c, --config_file : json config file\n"
      "  -o, --output_dir : output dir to save results\n");
}

static int make_dir(const char *path) {
  struct stat st;
  if (stat(path, &st) == 0) {
    return 0;
  }
  return mkdir(path, 0755);
}

int main(int argc, char *argv[]) {
  char *config_file = NULL;
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
  printf("  config_file: %s\n", config_file);
  printf("  output_dir:  %s\n", output_dir);

  make_dir(output_dir);

  TDLHandle tdl_handle = TDL_CreateHandle(0);

  char **channel_names = NULL;
  uint8_t channel_size = 0;
  int ret = TDL_APP_Init(tdl_handle, "vehicle_adas", config_file,
                         &channel_names, &channel_size, false);
  if (ret != 0) {
    printf("TDL_APP_Init failed with %#x!\n", ret);
    goto exit1;
  }

  for (int i = 0; i < channel_size; i++) {
    printf("channel[%d]: %s\n", i, channel_names[i]);
  }

  bool to_exit = false;
  while (true) {
    for (size_t i = 0; i < channel_size; i++) {
      TDLVehicleAdasInfo adas_info = {0};
      ret = TDL_APP_VehicleAdas(tdl_handle, channel_names[i], &adas_info);
      if (ret == 1) {
        continue;
      } else if (ret == 2) {
        to_exit = true;
        break;
      } else if (ret != 0) {
        printf("TDL_APP_VehicleAdas failed with %#x!\n", ret);
        goto exit1;
      }

      uint32_t obj_count = adas_info.adas_objects.size;
      printf("frame_id:%" PRIu64 " objects:%u lane_state:%d lanes:%u\n",
             adas_info.frame_id, obj_count, adas_info.lane_meta.lane_state,
             adas_info.lane_meta.size);

      // Print ADAS info
      for (uint32_t j = 0; j < obj_count; j++) {
        TDLVehicleAdasObjectInfo *obj = &adas_info.adas_objects.info[j];
        float ctx = (obj->box.x1 + obj->box.x2) / 2.0f / adas_info.frame_width;
        float cty = (obj->box.y1 + obj->box.y2) / 2.0f / adas_info.frame_height;
        float w = (obj->box.x2 - obj->box.x1) / adas_info.frame_width;
        float h = (obj->box.y2 - obj->box.y1) / adas_info.frame_height;
        printf(
            "  [%u] cls:%d bbox:%.2f %.2f %.2f %.2f dis:%.1fm speed:%.1fm/s "
            "state:%s\n",
            (unsigned int)obj->track_id, obj->class_id, ctx, cty, w, h,
            obj->distance, obj->speed, state_str[obj->state]);
      }

      if (adas_info.lane_meta.lane_state == 1) {
        printf("  LANE DEPARTURE WARNING!\n");
      }

      // Draw boxes
      int *colors = NULL;
      box_t *boxes = NULL;
      if (obj_count > 0) {
        boxes = (box_t *)malloc(obj_count * sizeof(box_t));
        colors = (int *)malloc(obj_count * 3 * sizeof(int));

        for (uint32_t j = 0; j < obj_count; j++) {
          TDLVehicleAdasObjectInfo *obj = &adas_info.adas_objects.info[j];
          boxes[j].x1 = obj->box.x1;
          boxes[j].y1 = obj->box.y1;
          boxes[j].x2 = obj->box.x2;
          boxes[j].y2 = obj->box.y2;

          if (obj->state != 0) {
            // Red for START / COLLISION_WARNING
            colors[j * 3] = 0;
            colors[j * 3 + 1] = 0;
            colors[j * 3 + 2] = 255;
          } else {
            // Green for NORMAL
            colors[j * 3] = 0;
            colors[j * 3 + 1] = 255;
            colors[j * 3 + 2] = 0;
          }
        }
      }

      char outpath[256];
      snprintf(outpath, sizeof(outpath), "%s/%08" PRIu64 ".jpg", output_dir,
               adas_info.frame_id);
      ret = VisualizeRectangle(boxes, (int32_t)obj_count, adas_info.image,
                               outpath, colors);
      if (ret != 0) {
        printf("VisualizeRectangle failed with %#x!\n", ret);
      }

      if (boxes) free(boxes);
      if (colors) free(colors);

      TDL_ReleaseVehicleAdasInfo(&adas_info);
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
  return 0;
}
