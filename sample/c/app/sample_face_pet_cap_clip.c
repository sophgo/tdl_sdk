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
#include "tdl_object_def.h"
#include "tdl_sdk.h"
#include "tdl_utils.h"

#define FEATURE_SIZE 256

static const char *emotionStr[] = {"Anger",   "Disgust", "Fear",    "Happy",
                                   "Neutral", "Sad",     "Surprise"};

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf("  %s -c <config_file> -g <gallery_dir> -o <output_dir>\n", prog_name);
  printf("  %s --config_file <path> --gallery_dir <dir> --output_dir <dir>\n",
         prog_name);
  printf("Options:\n");
  printf(
      "  -c, --config_file : json config file\n"
      "  -g, --gallery_dir : the face feature directory contains feature files "
      "named 0.bin, 1.bin, 2.bin...(no more than 100)\n"
      "  -o, --output_dir : output dir to save snapshot\n");
}

int main(int argc, char *argv[]) {
  char *config_file = NULL;  // sample/config/face_pet_cap_app.json
  char *gallery_dir = NULL;
  char *output_dir = NULL;

  struct option long_options[] = {
      {"config_file", required_argument, 0, 'c'},
      {"gallery_dir", required_argument, 0, 'g'},
      {"output_dir", required_argument, 0, 'o'},
      {"help", no_argument, 0, 'h'},
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "c:g:o:h", long_options, NULL)) != -1) {
    switch (opt) {
      case 'c':
        config_file = optarg;
        break;
      case 'g':
        gallery_dir = optarg;
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

  if (!config_file || !gallery_dir || !output_dir) {
    fprintf(stderr,
            "Error: config_file and gallery_dir and output_dir are required\n");
    print_usage(argv[0]);
    return -1;
  }

  printf("Running with:\n");
  printf("  config_file:    %s\n", config_file);
  printf("  gallery_dir:   %s\n", gallery_dir);
  printf("  output_dir:  %s\n", output_dir);

  TDLFeatureInfo gallery_feature = {0};
  int ret = TDL_GetGalleryFeature(gallery_dir, &gallery_feature, FEATURE_SIZE);
  if (ret != 0) {
    printf("get gallery feature from %s failed with %#x!\n", gallery_dir, ret);
    goto exit1;
  }

  TDLHandle tdl_handle = TDL_CreateHandle(0);

  char **channel_names = NULL;
  uint8_t channel_size = 0;
  ret = TDL_APP_Init(tdl_handle, "face_pet_capture", config_file,
                     &channel_names, &channel_size, false);
  if (ret != 0) {
    printf("TDL_APP_Init failed with %#x!\n", ret);
    goto exit1;
  }

  bool to_exit = false;
  while (true) {
    for (size_t i = 0; i < channel_size; i++) {
      TDLCaptureInfo capture_info = {0};
      ret = TDL_APP_Capture(tdl_handle, channel_names[i], &capture_info);
      if (ret == 1) {
        continue;
      } else if (ret == 2) {
        to_exit = true;
        break;
      } else if (ret != 0) {
        printf("TDL_APP_Capture failed with %#x!\n", ret);
        goto exit0;
      }

      printf("detect person size: %d, pet size: %d\n",
             capture_info.person_meta.size, capture_info.pet_meta.size);

      for (uint32_t j = 0; j < capture_info.snapshot_size; j++) {
        if (capture_info.snapshot_info[j].object_image) {  // save snapshot
          char filename[512];

          if (capture_info.snapshot_info[j].object_type ==
              TDL_OBJECT_TYPE_PERSON) {
            sprintf(filename,
                    "%s/%" PRIu64 "_person_ID_%" PRIu64 "_pairID_%" PRIu64
                    "_qua_%.3f.jpg",
                    output_dir, capture_info.snapshot_info[j].snapshot_frame_id,
                    capture_info.snapshot_info[j].track_id,
                    capture_info.snapshot_info[j].pair_track_id,
                    capture_info.snapshot_info[j].quality);
          } else {
            sprintf(filename,
                    "%s/%" PRIu64 "_face_ID_%" PRIu64 "_pairID_%" PRIu64
                    "_qua_%.3f_male[%d]_glass[%d]_age[%d]_emotion[%s].jpg",
                    output_dir, capture_info.snapshot_info[j].snapshot_frame_id,
                    capture_info.snapshot_info[j].track_id,
                    capture_info.snapshot_info[j].pair_track_id,
                    capture_info.snapshot_info[j].quality,
                    capture_info.snapshot_info[j].male,
                    capture_info.snapshot_info[j].glass,
                    capture_info.snapshot_info[j].age,
                    emotionStr[capture_info.snapshot_info[j].emotion]);
          }

          ret = TDL_EncodeFrame(tdl_handle,
                                capture_info.snapshot_info[j].object_image,
                                filename, 1);
          if (ret != 0) {
            printf("TDL_EncodeFrame failed with %#x!\n", ret);
            continue;
          }

          if (capture_info.snapshot_info[j]
                  .encoded_full_image) {  // save full image
            sprintf(filename,
                    "%s/%" PRIu64 "_%" PRIu64
                    "_box[%.2f,%.2f,%.2f,%.2f]_full_image.jpg",
                    output_dir, capture_info.snapshot_info[j].snapshot_frame_id,
                    capture_info.snapshot_info[j].track_id,
                    capture_info.snapshot_info[j].ori_box.x1,
                    capture_info.snapshot_info[j].ori_box.y1,
                    capture_info.snapshot_info[j].ori_box.x2,
                    capture_info.snapshot_info[j].ori_box.y2);

            FILE *f;
            f = fopen(filename, "wb");
            if (!f) {
              printf("open file fail: %s\n", filename);
            } else {
              fwrite(capture_info.snapshot_info[j].encoded_full_image, 1,
                     capture_info.snapshot_info[j].full_length, f);
            }
            fclose(f);
          }
        }

        if (capture_info.snapshot_info[j].object_type ==
                TDL_OBJECT_TYPE_PERSON &&
            capture_info.features[j].size > 0) {
          char feature_filename[512];
          sprintf(feature_filename,
                  "%s/%" PRIu64 "_person_ID_%" PRIu64 "_feature.bin",
                  output_dir, capture_info.snapshot_info[j].snapshot_frame_id,
                  capture_info.snapshot_info[j].track_id);

          FILE *f = fopen(feature_filename, "wb");
          if (f) {
            // 将int8_t*指针转换为float*以正确访问数据
            float *feature_data = (float *)capture_info.features[j].ptr;
            size_t data_size = capture_info.features[j].size * sizeof(float);
            fwrite(feature_data, 1, data_size, f);
            fclose(f);
            printf("Saved person feature to %s\n", feature_filename);
          } else {
            printf("Failed to open feature file: %s\n", feature_filename);
          }
        }
      }

      TDL_ReleaseCaptureInfo(&capture_info);
    }

    if (to_exit) {
      break;
    }
  }

exit2:
  for (int i = 0; i < channel_size; i++) {
    free(channel_names[i]);
  }
  free(channel_names);

exit1:
  for (int i = 0; i < gallery_feature.size; i++) {
    TDL_ReleaseFeatureMeta(&gallery_feature.feature[i]);
  }
exit0:
  TDL_DestroyHandle(tdl_handle);
}