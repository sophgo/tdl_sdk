#include <ctype.h>
#include <dirent.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "meta_visualize.h"
#include "tdl_sdk.h"
#include "tdl_utils.h"

#define MAX_FILE_COUNT 1000

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf("  %s -m <model_path> -i <input_image> [-o <output_dir>]\n",
         prog_name);
  printf("  %s --model_path <path> --input <image> [--output <dir>]\n\n",
         prog_name);
  printf("Options:\n");
  printf(
      "  -m, --model_path  Path to object detection model\n"
      "                    "
      "<yolov8n_det_person_vehicle|mbv2_det_person|yolov8n_det_coco80|yolov10n_"
      "det_coco80|...>\n");
  printf(
      "  -i, --input       Path to input images dir, such as \"-i input\",\n"
      "  The images in the folder must be named in the format of xxx_d.xxx, "
      "  such as image_0.jpg, image_1.jpg......\n"
      "  such as input_0.jpg, input_1.jpg......\n"
      "  MAX_FILE_COUNT is 1000\n");
  printf("  -o, --output      Path to output image dir\n");
  printf("  -h, --help        Show this help message\n");
  printf("\n  (Uses TDL_Tracking: object-only tracking, no face fusion)\n");
}

int get_model_info(char *model_path, TDLModel *model_index) {
  int ret = 0;
  if (strstr(model_path, "yolov8n_det_person_vehicle") != NULL) {
    *model_index = TDL_MODEL_YOLOV8N_DET_PERSON_VEHICLE;
  } else if (strstr(model_path, "yolov8n_det_head_hardhat") != NULL) {
    *model_index = TDL_MODEL_YOLOV8N_DET_HEAD_HARDHAT;
  } else if (strstr(model_path, "yolov10n_det_coco80") != NULL) {
    *model_index = TDL_MODEL_YOLOV10_DET_COCO80;
  } else if (strstr(model_path, "yolov6n_det_coco80") != NULL) {
    *model_index = TDL_MODEL_YOLOV6_DET_COCO80;
  } else if (strstr(model_path, "yolov8n_det_coco80") != NULL) {
    *model_index = TDL_MODEL_YOLOV8_DET_COCO80;
  } else if (strstr(model_path, "ppyoloe_det_coco80") != NULL) {
    *model_index = TDL_MODEL_PPYOLOE_DET_COCO80;
  } else if (strstr(model_path, "yolov8n_det_fire") != NULL) {
    *model_index = TDL_MODEL_YOLOV8N_DET_FIRE;
  } else if (strstr(model_path, "yolov8n_det_fire_smoke") != NULL) {
    *model_index = TDL_MODEL_YOLOV8N_DET_FIRE_SMOKE;
  } else if (strstr(model_path, "yolov8n_det_hand_384_640") != NULL ||
             strstr(model_path, "yolov8n_det_hand_mv3") != NULL) {
    *model_index = TDL_MODEL_YOLOV8N_DET_HAND;
  } else if (strstr(model_path, "yolov8n_det_hand_face_person") != NULL) {
    *model_index = TDL_MODEL_YOLOV8N_DET_HAND_FACE_PERSON;
  } else if (strstr(model_path, "yolov8n_det_head_hardhat") != NULL) {
    *model_index = TDL_MODEL_YOLOV8N_DET_HEAD_HARDHAT;
  } else if (strstr(model_path, "yolov8n_det_head_shoulder") != NULL) {
    *model_index = TDL_MODEL_YOLOV8N_DET_HEAD_SHOULDER;
  } else if (strstr(model_path, "yolov8n_det_ir_person") != NULL ||
             strstr(model_path, "yolov8n_det_monitor_person") != NULL ||
             strstr(model_path, "yolov8n_det_overlook_person") != NULL) {
    *model_index = TDL_MODEL_YOLOV8N_DET_MONITOR_PERSON;
  } else if (strstr(model_path, "yolov8n_det_license_plate") != NULL) {
    *model_index = TDL_MODEL_YOLOV8N_DET_LICENSE_PLATE;
  } else if (strstr(model_path, "yolov8n_det_pet_person") != NULL) {
    *model_index = TDL_MODEL_YOLOV8N_DET_PET_PERSON;
  } else if (strstr(model_path, "yolov8n_det_bicycle_motor_ebicycle") != NULL) {
    *model_index = TDL_MODEL_YOLOV8N_DET_BICYCLE_MOTOR_EBICYCLE;
  } else if (strstr(model_path, "yolov8n_det_traffic_light") != NULL) {
    *model_index = TDL_MODEL_YOLOV8N_DET_TRAFFIC_LIGHT;
  } else if (strstr(model_path, "mbv2_det_person") != NULL) {
    *model_index = TDL_MODEL_MBV2_DET_PERSON;
  } else {
    ret = -1;
  }
  return ret;
}

int extractNumber(const char *str) {
  const char *p = strrchr(str, '_');
  if (p != NULL) {
    p++;
    if (isdigit(*p)) {
      return atoi(p);
    }
  }
  p = str;
  while (*p) {
    if (isdigit(*p)) {
      return atoi(p);
    }
    p++;
  }
  return 0;
}

int compareFileNames(const void *a, const void *b) {
  const char *name1 = *(const char **)a;
  const char *name2 = *(const char **)b;

  int num1 = extractNumber(name1);
  int num2 = extractNumber(name2);

  if (num1 != num2) {
    return num1 - num2;
  }
  return strcmp(name1, name2);
}

int main(int argc, char **argv) {
  char *model_path = NULL;
  char *video_file = NULL;
  char *output_file = NULL;
  struct dirent *entry;
  int image_num = 0;

  struct option long_options[] = {
      {"model_path", required_argument, 0, 'm'},
      {"input", required_argument, 0, 'i'},
      {"output", required_argument, 0, 'o'},
      {"help", no_argument, 0, 'h'},
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "m:i:o:h", long_options, NULL)) != -1) {
    switch (opt) {
      case 'm':
        model_path = optarg;
        break;
      case 'i':
        video_file = optarg;
        break;
      case 'o':
        output_file = optarg;
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

  if (!model_path || !video_file) {
    fprintf(stderr, "Error: All arguments are required\n");
    print_usage(argv[0]);
    return -1;
  }

  printf("Running with:\n");
  printf("  Model path:    %s\n", model_path);
  printf("  Input image:   %s\n", video_file);
  printf("  Output image:  %s\n", output_file);

  TDLModel model_id_obj;
  if (get_model_info(model_path, &model_id_obj) == -1) {
    printf("unsupported model: %s\n", model_path);
    return -1;
  }
  int ret = 0;
  char *files[MAX_FILE_COUNT];

  TDLHandle tdl_handle = TDL_CreateHandle(0);

  ret = TDL_OpenModel(tdl_handle, model_id_obj, model_path, NULL, 0);
  if (ret != 0) {
    printf("open object detection model failed with %#x!\n", ret);
    goto exit0;
  }

  DIR *dir = opendir(video_file);
  if (dir == NULL) {
    printf("open dir fail\n");
    goto exit1;
  }

  while ((entry = readdir(dir)) != NULL) {
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
      continue;
    }

    // Support both xxx_0.jpg and 00000001.jpg naming formats
    bool valid = false;
    const char *underscore = strrchr(entry->d_name, '_');
    if (underscore != NULL && isdigit(*(underscore + 1))) {
      valid = true;
    } else if (isdigit(entry->d_name[0])) {
      valid = true;
    }
    if (valid) {
      int file_size = strlen(video_file) + strlen(entry->d_name) + 2;
      files[image_num] = malloc(file_size);
      snprintf(files[image_num], file_size, "%s/%s", video_file, entry->d_name);
      image_num++;
      if (image_num >= MAX_FILE_COUNT) {
        break;
      }
    }
  }
  closedir(dir);
  qsort(files, image_num, sizeof(char *), compareFileNames);

  TDLImage image = NULL;
  for (int i = 0; i < image_num; i++) {
    printf("file path is %s\n", files[i]);
    image = TDL_ReadImage(files[i]);

    TDLObject obj_meta = {0};
    ret = TDL_Detection(tdl_handle, model_id_obj, image, &obj_meta);
    if (ret != 0) {
      printf("TDL_Detection failed with %#x!\n", ret);
    }

    TDLTracker track_meta = {0};
    ret = TDL_Tracking(tdl_handle, i, &obj_meta, &track_meta);
    if (obj_meta.size <= 0) {
      printf("none to detect\n");
    }
    if (track_meta.out_num > 0) {
      box_t boxes[track_meta.out_num];
      int rect_colors[track_meta.out_num * 3];
      char outpath[128];
      if (output_file != NULL) {
        size_t len = strlen(output_file);
        if (len > 0 && output_file[len - 1] == '/') {
          snprintf(outpath, 128, "%soutput_%d.jpg", output_file, i);
        } else {
          snprintf(outpath, 128, "%s/output_%d.jpg", output_file, i);
        }
      }
      for (int c = 0; c < track_meta.out_num; c++) {
        printf("frame_id = %d, obj: %d, track_id = %" PRIu64
               ", box = [%f, %f, %f, %f]\n",
               i, c, track_meta.info[c].id, track_meta.info[c].bbox.x1,
               track_meta.info[c].bbox.y1, track_meta.info[c].bbox.x2,
               track_meta.info[c].bbox.y2);
        boxes[c].x1 = track_meta.info[c].bbox.x1;
        boxes[c].y1 = track_meta.info[c].bbox.y1;
        boxes[c].x2 = track_meta.info[c].bbox.x2;
        boxes[c].y2 = track_meta.info[c].bbox.y2;
        rect_colors[c * 3] = 0;        // B
        rect_colors[c * 3 + 1] = 0;    // G
        rect_colors[c * 3 + 2] = 255;  // R (red)
        if (output_file != NULL) {
          char text[5] = {0};
          snprintf(text, 5, "%" PRIu64, track_meta.info[c].id);
          int text_colors[3] = {0, 255, 0};  // B, G, R (green)
          DrawText(image,
                   (int32_t)(boxes[c].x1 + (boxes[c].x2 - boxes[c].x1) / 2),
                   (int32_t)(boxes[c].y1 + (boxes[c].y2 - boxes[c].y1) / 2),
                   text, text_colors);
        }
      }
      if (output_file != NULL) {
        VisualizeRectangle(boxes, track_meta.out_num, image, outpath,
                           rect_colors);
      }
    }

    TDL_DestroyImage(image);
    TDL_ReleaseTrackMeta(&track_meta);
    TDL_ReleaseObjectMeta(&obj_meta);
    free(files[i]);
    files[i] = NULL;
  }

exit1:
  TDL_CloseModel(tdl_handle, model_id_obj);

exit0:
  TDL_DestroyHandle(tdl_handle);
}