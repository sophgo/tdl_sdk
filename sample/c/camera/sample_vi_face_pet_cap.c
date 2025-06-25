#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/time.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#include <pthread.h>
#include "cvi_comm_video.h"
#include "cvi_vi.h"
#include "tdl_sdk.h"

#define FEATURE_SIZE 256
static volatile bool to_exit = false;

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
  TDLHandle tdl_handle;
  uint8_t channel_size;
  char **channel_names;
  TDLFeatureInfo *gallery_feature;
} RUN_TDL_THREAD_ARG_S;

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf("  %s -c <config_file> -g <gallery_dir> -o <output_dir> -v <vi_chn>\n",
         prog_name);
  printf(
      "  %s --config_file <path> --gallery_dir <dir> --output_dir <dir> "
      "--vi_chn <int> \n",
      prog_name);
  printf("Options:\n");
  printf(
      "  -c, --config_file : json config file\n"
      "  -g, --gallery_dir : the face feature directory contains feature files "
      "named 0.bin, 1.bin, 2.bin...(no more than 100)\n"
      "  -o, --output_dir : output dir to save snapshot\n"
      "  -v, --vi_chn : optional , defult 0\n");
}

void *send_frame_thread(void *args) {
  printf("Enter send frame thread\n");
  SEND_FRAME_THREAD_ARG_S *pstArgs = (SEND_FRAME_THREAD_ARG_S *)args;

  uint64_t *channel_frame_id = malloc(pstArgs->channel_size * sizeof(uint64_t));
  memset(channel_frame_id, 0, pstArgs->channel_size * sizeof(uint64_t));

  while (to_exit == false) {
    for (size_t i = 0; i < pstArgs->channel_size; i++) {
      TDLImage image = NULL;

      image = TDL_GetCameraFrame(
          pstArgs->tdl_handle,
          pstArgs->vi_chn);  // if channel_size > 1, image should be taken from
                             // different vi_chn
      if (image == NULL) {
        printf("TDL_GetViFrame falied\n");
        continue;
      }

      channel_frame_id[i] += 1;

      int ret = TDL_APP_SetFrame(pstArgs->tdl_handle, pstArgs->channel_names[i],
                                 image, channel_frame_id[i], 3);
      if (ret != 0) {
        printf("TDL_APP_SetFrame failed with %d\n", ret);
        continue;
      }

      TDL_ReleaseCameraFrame(pstArgs->tdl_handle, pstArgs->vi_chn);
      TDL_DestroyImage(image);
    }
  }

  free(channel_frame_id);
}

void *run_tdl_thread(void *args) {
  RUN_TDL_THREAD_ARG_S *pstArgs = (RUN_TDL_THREAD_ARG_S *)args;

  uint64_t counter = 0;
  uint64_t last_counter = 0;
  uint32_t last_time_ms = get_time_in_ms();

  while (true) {
    // 检查键盘输入
    fd_set rfds;
    struct timeval tv = {0, 0};
    FD_ZERO(&rfds);
    FD_SET(STDIN_FILENO, &rfds);
    int key_pressed = select(STDIN_FILENO + 1, &rfds, NULL, NULL, &tv);
    if (key_pressed > 0 && FD_ISSET(STDIN_FILENO, &rfds)) {
      to_exit = true;
      break;  // 有键盘输入，退出循环
    }

    for (size_t i = 0; i < pstArgs->channel_size; i++) {
      TDLCaptureInfo capture_info = {0};

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

      int ret = TDL_APP_Capture(pstArgs->tdl_handle, pstArgs->channel_names[i],
                                &capture_info);

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
        for (uint32_t k = 0; k < pstArgs->gallery_feature->size; k++) {
          TDL_CaculateSimilarity(pstArgs->gallery_feature->feature[k],
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
      // TDL_ReleaseCameraFrame(pstArgs->tdl_handle, vi_chn);
      // TDL_DestroyImage(pstArgs->image);
    }

    if (to_exit) {
      break;
    }
  }

exit0:
  TDL_DestoryCamera(pstArgs->tdl_handle);
  TDL_DestroyHandle(pstArgs->tdl_handle);
}

int main(int argc, char *argv[]) {
  char *config_file = NULL;
  char *gallery_dir = NULL;
  char *output_dir = NULL;
  int vi_chn = 0;

  struct option long_options[] = {
      {"config_file", required_argument, 0, 'c'},
      {"gallery_dir", required_argument, 0, 'g'},
      {"output_dir", required_argument, 0, 'o'},
      {"vi_chn", no_argument, 0, 'v'},
      {"help", no_argument, 0, 'h'},
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "c:g:o:v:h", long_options, NULL)) !=
         -1) {
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
      case 'v':
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
  printf("  vi_chn:        %d\n", vi_chn);

  TDLFeatureInfo gallery_feature = {0};
  int ret = TDL_GetGalleryFeature(gallery_dir, &gallery_feature, FEATURE_SIZE);
  if (ret != 0) {
    printf("get gallery feature from %s failed with %#x!\n", gallery_dir, ret);
    goto exit0;
  }

  TDLImage image = NULL;
  TDLHandle tdl_handle = TDL_CreateHandle(0);

  char **channel_names = NULL;
  uint8_t channel_size = 0;
  ret = TDL_APP_Init(tdl_handle, "face_pet_capture", config_file,
                     &channel_names, &channel_size);
  if (ret != 0) {
    printf("TDL_APP_Init failed with %#x!\n", ret);
    goto exit1;
  }

  ret = TDL_InitCamera(tdl_handle);
  if (ret != 0) {
    printf("TDL_InitCamera %#x!\n", ret);
    return ret;
  }

  // 设置终端为非规范模式
  struct termios oldt, newt;
  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);

  printf("按任意键退出...\n");

  pthread_t stFrameThread, stTDLThread;

  SEND_FRAME_THREAD_ARG_S frame_args = {.tdl_handle = tdl_handle,
                                        .vi_chn = vi_chn,
                                        .channel_size = channel_size,
                                        .channel_names = channel_names};

  RUN_TDL_THREAD_ARG_S tdl_args = {.tdl_handle = tdl_handle,
                                   .channel_size = channel_size,
                                   .channel_names = channel_names,
                                   .gallery_feature = &gallery_feature};

  pthread_create(&stFrameThread, NULL, send_frame_thread, &frame_args);
  pthread_create(&stTDLThread, NULL, run_tdl_thread, &tdl_args);

  pthread_join(stFrameThread, NULL);
  pthread_join(stTDLThread, NULL);

  // 恢复终端设置
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);

exit1:
  for (int i = 0; i < channel_size; i++) {
    free(channel_names[i]);
  }
  free(channel_names);

exit0:

  for (int i = 0; i < gallery_feature.size; i++) {
    TDL_ReleaseFeatureMeta(&gallery_feature.feature[i]);
  }
  TDL_DestoryCamera(tdl_handle);
  TDL_DestroyHandle(tdl_handle);

  return ret;
}
