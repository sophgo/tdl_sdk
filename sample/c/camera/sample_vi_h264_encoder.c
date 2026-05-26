#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/time.h>
#include <termios.h>
#include <unistd.h>

#include "pthread_utils.h"
#include "sample_utils.h"
#include "tdl_sdk.h"

#define WIDTH 1280
#define HEIGHT 720
#define FPS 25
#define BITRATE 4096
#define GOP 60
#define VENC_CHN 1

static volatile bool to_exit = false;
static ImageQueue image_queue;
MUTEXAUTOLOCK_INIT(ResultMutex);

typedef struct {
  TDLHandle tdl_handle;
  int vi_chn;
  FILE *h264_file;
} EncodeArgs;

void *capture_thread(void *arg) {
  EncodeArgs *args = (EncodeArgs *)arg;
  TDLImage image = NULL;

  while (to_exit == false) {
    image = GetCameraFrame(args->tdl_handle, args->vi_chn);
    if (image == NULL) {
      fprintf(stderr, "Error: GetCameraFrame failed!\n");
      usleep(10000);
      continue;
    }

    if (Image_Enqueue(&image_queue, image) == -1) {
      if (to_exit) {
        ReleaseCameraFrame(args->tdl_handle, args->vi_chn);
        TDL_DestroyImage(image);
        break;
      }
      fprintf(stderr, "Error: Image_Enqueue failed!\n");
      ReleaseCameraFrame(args->tdl_handle, args->vi_chn);
      TDL_DestroyImage(image);
      continue;
    }
  }

  return NULL;
}

void *encode_thread(void *arg) {
  EncodeArgs *args = (EncodeArgs *)arg;
  TDLImage image = NULL;
  int ret;
  static unsigned int frame_count = 0;

  while (to_exit == false) {
    {
      MutexAutoLock(ResultMutex, lock);
      image = Image_Dequeue(&image_queue);
      if (image == NULL) {
        if (to_exit) {
          break;
        }
        continue;
      }
    }

    uint8_t *encoded_data = NULL;
    uint32_t encoded_size = 0;

    ret =
        TDL_EncodeH264FrameRaw(args->tdl_handle, image, VENC_CHN, WIDTH, HEIGHT,
                               FPS, BITRATE, GOP, &encoded_data, &encoded_size);
    if (ret != 0) {
      fprintf(stderr, "Error: TDL_EncodeH264FrameRaw failed! ret=%d\n", ret);
    } else if (encoded_data && encoded_size > 0) {
      fwrite(encoded_data, 1, encoded_size, args->h264_file);
      fflush(args->h264_file);
      free(encoded_data);

      frame_count++;
      if (frame_count % 30 == 0) {
        printf("[H264 Encoder] encoded %u frames\n", frame_count);
      }
    }

    TDL_DestroyImage(image);
    ReleaseCameraFrame(args->tdl_handle, args->vi_chn);
  }
  return NULL;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <output.h264>\n", argv[0]);
    return -1;
  }
  const char *out_path = argv[1];

  TDLHandle tdl_handle = TDL_CreateHandle(0);

  int ret = InitCamera(tdl_handle, WIDTH, HEIGHT, IMAGE_YUV420SP_UV, 3);
  if (ret != 0) {
    fprintf(stderr, "Error: InitCamera failed!\n");
    TDL_DestroyHandle(tdl_handle);
    return ret;
  }

  EncodeArgs encode_args;
  encode_args.tdl_handle = tdl_handle;
  encode_args.vi_chn = 0;
  encode_args.h264_file = fopen(out_path, "wb");
  if (encode_args.h264_file == NULL) {
    fprintf(stderr, "Error: cannot open output file %s\n", out_path);
    DestoryCamera(tdl_handle);
    TDL_DestroyHandle(tdl_handle);
    return -1;
  }

  InitQueue(&image_queue);

  pthread_t capture_tid, encode_tid;

  printf("[H264 Encoder] output: %s, %dx%d@%dfps %dkbps\n", out_path, WIDTH,
         HEIGHT, FPS, BITRATE);
  printf("Starting threads...\n");

  if (pthread_create(&capture_tid, NULL, capture_thread, &encode_args) != 0) {
    fprintf(stderr, "Error: Failed to create capture thread!\n");
    fclose(encode_args.h264_file);
    DestroyQueue(&image_queue);
    DestoryCamera(tdl_handle);
    TDL_DestroyHandle(tdl_handle);
    return -1;
  }
  if (pthread_create(&encode_tid, NULL, encode_thread, &encode_args) != 0) {
    fprintf(stderr, "Error: Failed to create encode thread!\n");
    to_exit = true;
    ExitQueue(&image_queue);
    fclose(encode_args.h264_file);
    DestroyQueue(&image_queue);
    DestoryCamera(tdl_handle);
    TDL_DestroyHandle(tdl_handle);
    return -1;
  }

  // Set terminal to non-canonical mode
  struct termios oldt, newt;
  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);

  printf("Press any key to exit...\n");

  while (1) {
    fd_set rfds;
    struct timeval tv = {0, 0};
    FD_ZERO(&rfds);
    FD_SET(STDIN_FILENO, &rfds);
    int key_pressed = select(STDIN_FILENO + 1, &rfds, NULL, NULL, &tv);
    if (key_pressed > 0 && FD_ISSET(STDIN_FILENO, &rfds)) {
      break;
    }
    usleep(100000);
  }

  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);

  printf("Stopping threads...\n");
  to_exit = true;
  ExitQueue(&image_queue);

  pthread_join(capture_tid, NULL);
  pthread_join(encode_tid, NULL);

  fclose(encode_args.h264_file);
  printf("H264 stream saved to %s\n", out_path);

  DestroyQueue(&image_queue);
  DestoryCamera(tdl_handle);
  TDL_DestroyHandle(tdl_handle);
  return ret;
}