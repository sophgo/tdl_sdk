#include <dirent.h>
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
  printf("  %s -m <model_path_face>,<model_path_obj> -i <input_image>\n",
         prog_name);
  printf("Options:\n");
  printf(
      "  -m, --model_path  Path to model"
      "<scrfd_det_face_xxx>,<mbv2_det_person_xxx>\n");
  printf(
      "  -i, --input       Path to input images dir, such as \"-i input\",\n"
      "  The images in the folder must be named in the format of xxx_d.xxx, "
      "  such as image_0.jpg, image_1.jpg......\n"
      "  such as input_0.jpg, input_1.jpg......\n"
      "  MAX_FILE_COUNT is 1000\n");
  printf("  -h, --help        Show this help message\n");
}

int extractNumber(const char *str) {
  const char *p = strrchr(str, '_');
  if (p != NULL) {
    p++;  // 跳过下划线
    if (isdigit(*p)) {
      return atoi(p);
    }
  }
  // 如果没有下划线数字模式，则扫描整个字符串找第一个数字
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
  char *models = NULL;
  char *model1 = NULL;
  char *model2 = NULL;
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
        models = optarg;
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

  if (!models || !video_file) {
    fprintf(stderr, "Error: All arguments are required\n");
    print_usage(argv[0]);
    return -1;
  }

  char *comma = strchr(models, ',');
  if (!comma || comma == models || !*(comma + 1)) {
    fprintf(stderr, "Error: Models must be in format '<face>,<obj>'\n");
    return -1;
  }
  model1 = models;
  *comma = '\0';
  model2 = comma + 1;

  printf("Running with:\n");
  printf("  Model path:    %s\n", models);
  printf("  Input image:   %s\n", video_file);
  printf("  Output image:  %s\n", output_file);

  TDLModel model_id_face = TDL_MODEL_SCRFD_DET_FACE;
  TDLModel model_id_obj = TDL_MODEL_MBV2_DET_PERSON_256_448;
  int ret = 0;
  char *files[MAX_FILE_COUNT];

  TDLHandle tdl_handle = TDL_CreateHandle(0);

  ret = TDL_OpenModel(tdl_handle, model_id_face, model1, NULL);
  if (ret != 0) {
    printf("open face detection model failed with %#x!\n", ret);
    goto exit0;
  }

  ret = TDL_OpenModel(tdl_handle, model_id_obj, model2, NULL);
  if (ret != 0) {
    printf("open face attribute model failed with %#x!\n", ret);
    goto exit1;
  }

  DIR *dir = opendir(video_file);
  if (dir == NULL) {
    printf("open dir fail\n");
    goto exit2;
  }

  while ((entry = readdir(dir)) != NULL) {
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
      continue;
    }

    const char *underscore = strrchr(entry->d_name, '_');
    if (underscore != NULL && isdigit(*(underscore + 1))) {
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
    TDLFace face_meta = {0};
    ret = TDL_FaceDetection(tdl_handle, model_id_face, image, &face_meta);
    if (ret != 0) {
      printf("face detection failed with %#x!\n", ret);
    }

    TDLObject obj_meta = {0};
    ret = TDL_Detection(tdl_handle, model_id_obj, image, &obj_meta);
    if (ret != 0) {
      printf("TDL_Detection failed with %#x!\n", ret);
    }

    if (face_meta.size <= 0 && obj_meta.size <= 0) {
      printf("none to detect\n");
      TDL_DestroyImage(image);
      TDL_ReleaseFaceMeta(&face_meta);
      TDL_ReleaseObjectMeta(&obj_meta);
      free(files[i]);
      files[i] = NULL;
    }

    TDLTracker track_meta;
    ret = TDL_Tracking(tdl_handle, i, &face_meta, &obj_meta, &track_meta);
    if (track_meta.out_num > 0) {
      box_t boxes[track_meta.out_num];
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
        printf(
            "frame_id = %d, obj: %d, track_id = %d, box = [%f, %f, %f, %f]\n",
            i, c, track_meta.info[c].id, track_meta.info[c].bbox.x1,
            track_meta.info[c].bbox.y1, track_meta.info[c].bbox.x2,
            track_meta.info[c].bbox.y2);
        boxes[c].x1 = track_meta.info[c].bbox.x1;
        boxes[c].y1 = track_meta.info[c].bbox.y1;
        boxes[c].x2 = track_meta.info[c].bbox.x2;
        boxes[c].y2 = track_meta.info[c].bbox.y2;
        if (output_file != NULL) {
          char text[5] = {0};
          snprintf(text, 5, "%d", track_meta.info[c].id);
          if (c == 0) {
            TDL_VisualizText(boxes[c].x1 + (boxes[c].x2 - boxes[c].x1) / 2,
                             boxes[c].y1 + (boxes[c].y2 - boxes[c].y1) / 2,
                             text, files[i], outpath);
          } else {
            TDL_VisualizText(boxes[c].x1 + (boxes[c].x2 - boxes[c].x1) / 2,
                             boxes[c].y1 + (boxes[c].y2 - boxes[c].y1) / 2,
                             text, outpath, outpath);
          }
        }
      }
      if (output_file != NULL) {
        TDL_VisualizeRectangle(boxes, track_meta.out_num, outpath, outpath);
      }
    }

    TDL_DestroyImage(image);
    TDL_ReleaseTrackMeta(&track_meta);
    TDL_ReleaseFaceMeta(&face_meta);
    TDL_ReleaseObjectMeta(&obj_meta);
    free(files[i]);
    files[i] = NULL;
  }

exit2:
  TDL_CloseModel(tdl_handle, model_id_obj);

exit1:
  TDL_CloseModel(tdl_handle, model_id_face);

exit0:
  TDL_DestroyHandle(tdl_handle);
}