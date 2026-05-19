#include <getopt.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/time.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#include "cvi_comm_video.h"
#include "cvi_vi.h"
#include "meta_visualize.h"
#include "sample_utils.h"
#include "tdl_sdk.h"

// #ifndef __CV184X__
// #define ENABLE_RTSP
// #endif

#define VI_WIDTH 960
#define VI_HEIGHT 540
#define FEATURE_SIZE 256
#define APP_FRAME_BUFFER_SIZE 100
#define CAPTURE_WAIT_TIMEOUT_MS 30000

static volatile bool to_exit = false;
static const char *emotionStr[] = {"Anger",   "Disgust", "Fear",    "Happy",
                                   "Neutral", "Sad",     "Surprise"};

static uint32_t get_time_in_ms() {
  struct timeval tv;
  if (gettimeofday(&tv, NULL) < 0) {
    return 0;
  }
  return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

static bool has_keyboard_input(void) {
  fd_set rfds;
  struct timeval tv = {0, 0};
  FD_ZERO(&rfds);
  FD_SET(STDIN_FILENO, &rfds);
  int key_pressed = select(STDIN_FILENO + 1, &rfds, NULL, NULL, &tv);
  return key_pressed > 0 && FD_ISSET(STDIN_FILENO, &rfds);
}

static int wait_capture_result(TDLHandle tdl_handle, const char *channel_name,
                               TDLCaptureInfo *capture_info) {
  uint32_t wait_start_ms = get_time_in_ms();

  while (!to_exit) {
    int ret = TDL_APP_Capture(tdl_handle, channel_name, capture_info);
    if (ret == 0) {
      return 0;
    }
    if (ret == 2) {
      to_exit = true;
      return ret;
    }
    if (ret != 1) {
      return ret;
    }
    if (get_time_in_ms() - wait_start_ms > CAPTURE_WAIT_TIMEOUT_MS) {
      return -2;
    }
    if (has_keyboard_input()) {
      to_exit = true;
      return 2;
    }
    usleep(1000);
  }

  return 2;
}

static void process_capture_result(TDLHandle tdl_handle,
                                   const TDLCaptureInfo *capture_info,
                                   const TDLFeatureInfo *gallery_feature,
                                   const char *output_dir) {
#ifdef ENABLE_RTSP
  VIDEO_FRAME_INFO_S *frame = NULL;
  RtspContext rtsp_context = {0};
  rtsp_context.chn = 0;
  rtsp_context.pay_load_type = PT_H264;
  rtsp_context.frame_width = VI_WIDTH;
  rtsp_context.frame_height = VI_HEIGHT;
#endif

  for (uint32_t j = 0; j < capture_info->snapshot_size; j++) {
    const char *emotion = "Unknown";
    if (capture_info->snapshot_info[j].emotion >= 0 &&
        capture_info->snapshot_info[j].emotion <
            (int)(sizeof(emotionStr) / sizeof(emotionStr[0]))) {
      emotion = emotionStr[capture_info->snapshot_info[j].emotion];
    }
    printf("snapshot[%d]: male:%d,glass:%d,age:%d,emotion:%s\n", j,
           capture_info->snapshot_info[j].male,
           capture_info->snapshot_info[j].glass,
           capture_info->snapshot_info[j].age, emotion);

    if (capture_info->snapshot_info[j].object_image) {
      char filename[512];
      snprintf(filename, sizeof(filename),
               "%s/%" PRIu64 "_face_%" PRIu64
               "_qua_%.3f_male[%d]_glass[%d]_age[%d]_emotion[%s].jpg",
               output_dir ? output_dir : ".",
               capture_info->snapshot_info[j].snapshot_frame_id,
               capture_info->snapshot_info[j].track_id,
               capture_info->snapshot_info[j].quality,
               capture_info->snapshot_info[j].male,
               capture_info->snapshot_info[j].glass,
               capture_info->snapshot_info[j].age, emotion);

      int ret = TDL_EncodeFrame(
          tdl_handle, capture_info->snapshot_info[j].object_image, filename, 1);
      if (ret != 0) {
        printf("TDL_EncodeFrame failed with %#x!\n", ret);
        continue;
      }

      if (capture_info->snapshot_info[j].encoded_full_image) {
        snprintf(filename, sizeof(filename),
                 "%s/%" PRIu64 "_face_%" PRIu64
                 "_box[%.2f,%.2f,%.2f,%.2f]_full_image.jpg",
                 output_dir ? output_dir : ".",
                 capture_info->snapshot_info[j].snapshot_frame_id,
                 capture_info->snapshot_info[j].track_id,
                 capture_info->snapshot_info[j].ori_box.x1,
                 capture_info->snapshot_info[j].ori_box.y1,
                 capture_info->snapshot_info[j].ori_box.x2,
                 capture_info->snapshot_info[j].ori_box.y2);

        FILE *f = fopen(filename, "wb");
        if (!f) {
          printf("open file fail: %s\n", filename);
        } else {
          fwrite(capture_info->snapshot_info[j].encoded_full_image, 1,
                 capture_info->snapshot_info[j].full_length, f);
          fclose(f);
        }
      }
    }

    if (gallery_feature && gallery_feature->size > 0) {
      float max_similarity = 0.0f;
      float similarity = 0.0f;
      uint8_t top_index = 0;
      for (uint32_t k = 0; k < gallery_feature->size; k++) {
        TDL_CaculateSimilarity(gallery_feature->feature[k],
                               capture_info->features[j], &similarity);
        if (similarity > max_similarity) {
          max_similarity = similarity;
          top_index = (uint8_t)k;
        }
      }

      if (max_similarity > 0.4f) {
        printf("match feature %u.bin, track id: %" PRIu64
               ", similarity: %.2f\n",
               (unsigned)top_index, capture_info->snapshot_info[j].track_id,
               max_similarity);
      }
    }
  }

#ifdef ENABLE_RTSP
  TDLImage image = capture_info->image;
  if (image && TDL_WrapImage(image, &frame) == 0 && frame) {
    TDLBrush brush = {0};
    brush.size = 5;
    brush.color.r = 0;
    brush.color.g = 255;
    brush.color.b = 0;
    for (int i = 0; i < capture_info->person_meta.size; i++) {
      TDLObjectInfo *obj_info = &capture_info->person_meta.info[i];
      snprintf(obj_info->name, sizeof(obj_info->name), "id:%d",
               obj_info->track_id);
    }
    DrawObjRect(&capture_info->person_meta, frame, true, brush);

    brush.color.r = 255;
    brush.color.g = 0;
    brush.color.b = 0;
    for (int i = 0; i < capture_info->pet_meta.size; i++) {
      TDLObjectInfo *pet_info = &capture_info->pet_meta.info[i];
      snprintf(pet_info->name, sizeof(pet_info->name), "score:%.2f",
               pet_info->score);
    }
    DrawObjRect(&capture_info->pet_meta, frame, true, brush);

    brush.color.r = 0;
    brush.color.g = 0;
    brush.color.b = 255;
    for (int i = 0; i < capture_info->face_meta.size; i++) {
      TDLFaceInfo *face_info = &capture_info->face_meta.info[i];
      snprintf(face_info->name, sizeof(face_info->name), "id:%d",
               face_info->track_id);
    }
    DrawFaceRect(&capture_info->face_meta, frame, true, brush);

    brush.color.g = 255;
    brush.color.b = 0;
    {
      char text[128] = {0};
      snprintf(text, sizeof(text), "frame id:%d", capture_info->frame_id);
      ObjectWriteText(text, 50, 50, frame, brush);
    }

    if (SendFrameRTSP(frame, &rtsp_context) != 0) {
      printf("SendFrameRTSP failed\n");
    }
  }
#endif
}

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf("  %s -c <config_file> -g <gallery_dir> -o <output_dir> -v <vi_chn>\n",
         prog_name);
  printf(
      "  %s --config_file <path> --gallery_dir <dir> --output_dir <dir> "
      "--vi_chn <int>\n",
      prog_name);
  printf("Options:\n");
  printf(
      "  -c, --config_file : json config file\n"
      "  -g, --gallery_dir : the face feature directory contains feature files "
      "named 0.bin, 1.bin, 2.bin...(no more than 100)\n"
      "  -o, --output_dir : output dir to save snapshot\n"
      "  -v, --vi_chn : optional, default 0\n");
}

int main(int argc, char *argv[]) {
  char *config_file = NULL;
  char *gallery_dir = NULL;
  char *output_dir = NULL;
  char *vi_chn = NULL;
  int chn = 0;
  bool termios_changed = false;
  TDLHandle tdl_handle = NULL;
  char **channel_names = NULL;
  uint8_t channel_size = 0;
  int ret = 0;

  struct option long_options[] = {
      {"config_file", required_argument, 0, 'c'},
      {"gallery_dir", required_argument, 0, 'g'},
      {"output_dir", required_argument, 0, 'o'},
      {"vi_chn", required_argument, 0, 'v'},
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

  if (!config_file || !gallery_dir || !output_dir) {
    print_usage(argv[0]);
    return -1;
  }

  if (vi_chn) {
    chn = atoi(vi_chn);
  }

  printf("Running single-thread mode with:\n");
  printf("  config_file:  %s\n", config_file);
  printf("  gallery_dir:  %s\n", gallery_dir);
  printf("  output_dir:   %s\n", output_dir);
  printf("  vi_chn:       %d\n", chn);

  tdl_handle = TDL_CreateHandle(0);
  if (tdl_handle == NULL) {
    printf("TDL_CreateHandle failed\n");
    return -1;
  }

  TDLFeatureInfo gallery_feature = {0};
  ret = TDL_GetGalleryFeature(gallery_dir, &gallery_feature, FEATURE_SIZE);
  if (ret != 0) {
    printf("get gallery feature from %s failed with %#x!\n", gallery_dir, ret);
    goto exit0;
  }

  ret = InitCamera(tdl_handle, VI_WIDTH, VI_HEIGHT, IMAGE_YUV420SP_UV, 3);
  if (ret != 0) {
    printf("InitCamera %#x!\n", ret);
    goto exit0;
  }

  ret = TDL_APP_Init(tdl_handle, "face_pet_capture", config_file,
                     &channel_names, &channel_size, false);
  if (ret != 0) {
    printf("TDL_APP_Init failed with %#x!\n", ret);
    goto exit1;
  }

  struct termios oldt, newt;
  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  termios_changed = true;

  printf("单线程模式，按任意键退出...\n");

  uint64_t *channel_frame_id = calloc(channel_size, sizeof(uint64_t));
  if (channel_size > 0 && channel_frame_id == NULL) {
    printf("malloc channel_frame_id failed\n");
    ret = -1;
    goto exit2;
  }

  uint64_t counter = 0;
  uint64_t last_counter = 0;
  uint32_t last_time_ms = get_time_in_ms();

  while (!to_exit) {
    if (has_keyboard_input()) {
      to_exit = true;
      break;
    }

    for (size_t i = 0; i < channel_size && !to_exit; i++) {
      TDLImage image = GetCameraFrame(tdl_handle, chn);
      TDLCaptureInfo capture_info = {0};

      if (image == NULL) {
        printf("GetCameraFrame failed\n");
        usleep(1000);
        continue;
      }

      channel_frame_id[i] += 1;
      ret = TDL_APP_SetFrame(tdl_handle, channel_names[i], image,
                             channel_frame_id[i], APP_FRAME_BUFFER_SIZE);
      if (ret != 0) {
        printf("TDL_APP_SetFrame failed with %d\n", ret);
        ReleaseCameraFrame(tdl_handle, chn);
        TDL_DestroyImage(image);
        continue;
      }

      ret = wait_capture_result(tdl_handle, channel_names[i], &capture_info);
      if (ret == -2) {
        printf("wait capture timeout after %d ms, frame id: %" PRIu64 "\n",
               CAPTURE_WAIT_TIMEOUT_MS, channel_frame_id[i]);
        ReleaseCameraFrame(tdl_handle, chn);
        TDL_DestroyImage(image);
        continue;
      }
      if (ret == 2) {
        ReleaseCameraFrame(tdl_handle, chn);
        TDL_DestroyImage(image);
        to_exit = true;
        break;
      }
      if (ret != 0) {
        printf("TDL_APP_Capture failed with %#x!\n", ret);
        TDL_ReleaseCaptureInfo(&capture_info);
        ReleaseCameraFrame(tdl_handle, chn);
        TDL_DestroyImage(image);
        to_exit = true;
        break;
      }

      counter++;
      {
        int frm_diff = (int)(counter - last_counter);
        if (frm_diff > 30) {
          uint32_t cur_ts_ms = get_time_in_ms();
          float infer_time = (float)(cur_ts_ms - last_time_ms) / frm_diff;
          float fps = infer_time > 0.0f ? 1000.0f / infer_time : 0.0f;
          last_time_ms = cur_ts_ms;
          last_counter = counter;
          printf("detect person size: %d, pet size: %d\n",
                 capture_info.person_meta.size, capture_info.pet_meta.size);
          printf(
              "+++++++++++++++++++++++++++++++++++ frame:%d, infer time:%.2f, "
              "fps:%.2f\n",
              (int)counter, infer_time, fps);
        }
      }

      process_capture_result(tdl_handle, &capture_info, &gallery_feature,
                             output_dir);

      TDL_ReleaseCaptureInfo(&capture_info);
      ReleaseCameraFrame(tdl_handle, chn);
      TDL_DestroyImage(image);
    }
  }

  free(channel_frame_id);

exit2:
  if (termios_changed) {
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  }

  for (int i = 0; i < channel_size; i++) {
    free(channel_names[i]);
  }
  free(channel_names);

exit1:
  DestoryCamera(tdl_handle);

exit0:
  for (int i = 0; i < gallery_feature.size; i++) {
    TDL_ReleaseFeatureMeta(&gallery_feature.feature[i]);
  }
  TDL_DestroyHandle(tdl_handle);

  return ret;
}
