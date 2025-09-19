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

#define FEATURE_SIZE 256

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
                     &channel_names, &channel_size);
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

      // todo: save snapshot img

      for (uint32_t j = 0; j < capture_info.snapshot_size; j++) {
        printf("to do TDL_CaculateSimilarity\n");

        float max_similarity = 0;
        float similarity = 0;
        uint8_t top_index;
        for (uint32_t k = 0; k < gallery_feature.size; k++) {
          TDL_CaculateSimilarity(gallery_feature.feature[k],
                                 capture_info.features[j], &similarity);
          if (similarity > max_similarity) {
            max_similarity = similarity;
            top_index = k;
          }
        }

        if (max_similarity > 0.4) {
          printf("match feature %d.bin, track id: %ld, similarity: %.2f\n",
                 top_index, capture_info.snapshot_info[i].track_id,
                 max_similarity);
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