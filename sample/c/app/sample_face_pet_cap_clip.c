#include <ctype.h>
#include <dirent.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
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

bool make_dir(const char *path, mode_t mode) {
  if (mkdir(path, mode) == 0) {
    return true;  // 创建成功
  }
  if (errno == EEXIST) {
    return true;  // 已经存在
  }
  // 其他错误
  fprintf(stderr, "mkdir failed: %s,dir:%s\n", strerror(errno), path);
  return false;
}

// 创建文件夹的函数
bool create_id_folder(const char *dir_path, uint64_t id1, uint64_t id2,
                      char *dst_dir, size_t dst_dir_size) {
  char temp_dir[512];
  char old_dir[512];

  if (id2 <= 0) {
    // id2为0，创建id1_-1文件夹
    snprintf(temp_dir, sizeof(temp_dir), "%s/%" PRIu64 "_-1", dir_path, id1);
    make_dir(temp_dir, 0755);
    strncpy(dst_dir, temp_dir, dst_dir_size);
    return true;
  } else {
    // id2大于0，先检查id1_-1文件夹是否存在
    snprintf(old_dir, sizeof(old_dir), "%s/%" PRIu64 "_-1", dir_path, id1);
    snprintf(temp_dir, sizeof(temp_dir), "%s/%" PRIu64 "_%" PRIu64, dir_path,
             id1, id2);

    struct stat st;
    if (stat(old_dir, &st) == 0 && S_ISDIR(st.st_mode)) {
      // id1_-1文件夹存在，重命名为id1_id2
      if (rename(old_dir, temp_dir) != 0) {
        fprintf(stderr, "Failed to rename directory from %s to %s: %s\n",
                old_dir, temp_dir, strerror(errno));
        return false;
      }
    } else {
      // id1_-1文件夹不存在，直接创建id1_id2文件夹
      make_dir(temp_dir, 0755);
    }
    strncpy(dst_dir, temp_dir, dst_dir_size);
    return true;
  }
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

  // 创建output_dir/face文件夹
  char face_dir[512], person_dir[512], image_feature_dir[512];
  snprintf(face_dir, sizeof(face_dir), "%s/face", output_dir);
  make_dir(face_dir, 0755);
  snprintf(person_dir, sizeof(person_dir), "%s/person", output_dir);
  make_dir(person_dir, 0755);
  snprintf(image_feature_dir, sizeof(image_feature_dir), "%s/image_feature",
           output_dir);
  make_dir(image_feature_dir, 0755);

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

  time_t rawtime;
  struct tm *timeinfo;
  char timestamp_str[20];

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

      char dst_dir[512];
      char filename[512];

      time(&rawtime);
      timeinfo = localtime(&rawtime);
      strftime(timestamp_str, sizeof(timestamp_str), "%Y%m%d_%H%M%S", timeinfo);

      for (uint32_t j = 0; j < capture_info.snapshot_size; j++) {
        if (capture_info.snapshot_info[j].object_image) {  // save snapshot
          if (capture_info.snapshot_info[j].object_type ==
              TDL_OBJECT_TYPE_PERSON) {
            create_id_folder(person_dir, capture_info.snapshot_info[j].track_id,
                             capture_info.snapshot_info[j].pair_track_id,
                             dst_dir, sizeof(dst_dir));

            sprintf(filename,
                    "%s/%s_frameID_%" PRIu64 "_personID_%" PRIu64
                    "_pairID_%" PRIu64 "_qua_%.3f.jpg",
                    dst_dir, timestamp_str,
                    capture_info.snapshot_info[j].snapshot_frame_id,
                    capture_info.snapshot_info[j].track_id,
                    capture_info.snapshot_info[j].pair_track_id,
                    capture_info.snapshot_info[j].quality);
          } else {
            create_id_folder(face_dir, capture_info.snapshot_info[j].track_id,
                             capture_info.snapshot_info[j].pair_track_id,
                             dst_dir, sizeof(dst_dir));

            sprintf(filename,
                    "%s/%s_frameID_%" PRIu64 "_faceID_%" PRIu64
                    "_pairID_%" PRIu64
                    "_qua_%.3f_male[%d]_glass[%d]_age[%d]_emotion[%s].jpg",
                    dst_dir, timestamp_str,
                    capture_info.snapshot_info[j].snapshot_frame_id,
                    capture_info.snapshot_info[j].track_id,
                    capture_info.snapshot_info[j].pair_track_id,
                    capture_info.snapshot_info[j].quality,
                    capture_info.snapshot_info[j].male,
                    capture_info.snapshot_info[j].glass,
                    capture_info.snapshot_info[j].age,
                    emotionStr[capture_info.snapshot_info[j].emotion]);
          }
          printf("!!![1]filename: %s\n", filename);

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
          create_id_folder(image_feature_dir,
                           capture_info.snapshot_info[j].track_id,
                           capture_info.snapshot_info[j].pair_track_id, dst_dir,
                           sizeof(dst_dir));

          sprintf(filename,
                  "%s/%s_frameID_%" PRIu64 "_personID_%" PRIu64
                  "_pairID_%" PRIu64 "_qua_%.3f.bin",
                  dst_dir, timestamp_str,
                  capture_info.snapshot_info[j].snapshot_frame_id,
                  capture_info.snapshot_info[j].track_id,
                  capture_info.snapshot_info[j].pair_track_id,
                  capture_info.snapshot_info[j].quality);
          printf("!!![2]filename: %s\n", filename);

          // sprintf(filename, "%s/%s_image_feature.bin", dst_dir,
          // timestamp_str);

          FILE *f = fopen(filename, "wb");
          if (f) {
            // 将int8_t*指针转换为float*以正确访问数据
            float *feature_data = (float *)capture_info.features[j].ptr;
            size_t data_size = capture_info.features[j].size * sizeof(float);
            fwrite(feature_data, 1, data_size, f);
            fclose(f);
            printf("Saved person feature to %s\n", filename);
          } else {
            printf("Failed to open feature file: %s\n", filename);
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