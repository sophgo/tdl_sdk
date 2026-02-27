#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/time.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#include "meta_visualize.h"
#include "pthread_utils.h"
#include "sample_utils.h"
#include "tdl_sdk.h"

#define WIDTH 1280
#define HEIGHT 720

static uint64_t g_frame_id = 0;

static volatile bool to_exit = false;
static ImageQueue image_queue;
MUTEXAUTOLOCK_INIT(ResultMutex);

// --- Threading Structures ---
RtspContext rtsp_context = {.chn = 0,
                            .pay_load_type = PT_H264,
                            .frame_width = WIDTH,
                            .frame_height = HEIGHT};

typedef struct {
  TDLHandle tdl_handle;
  int vi_chn;
  RtspContext *rtsp_context;
} ProcessArgs;

// --- Worker Threads ---
void *capture_thread(void *arg) {
  ProcessArgs *args = (ProcessArgs *)arg;
  TDLImage image = NULL;

  while (to_exit == false) {
    // Capture frame (this might block waiting for VSYNC)
    image = GetCameraFrame(args->tdl_handle, args->vi_chn);
    if (image == NULL) {
      fprintf(stderr, "Error: GetCameraFrame failed!\n");
      // Avoid busy loop if camera fails
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

void *process_thread(void *arg) {
  ProcessArgs *args = (ProcessArgs *)arg;
  TDLImage image = NULL;
  VIDEO_FRAME_INFO_S *frame = NULL;
  int ret;
  struct timeval start_time, end_time;
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

    TDL_WrapImage(image, &frame);

    TDLBrush brush = {0};
    brush.size = 5;
    brush.color.r = 0;
    brush.color.g = 255;
    brush.color.b = 0;

    char text[128] = {0};
    snprintf(text, sizeof(text), "frame id:%d", g_frame_id);
    ObjectWriteText(text, 50, 50, frame, brush);

    g_frame_id++;

    // Send RTSP Frame
    ret = SendFrameRTSP(frame, args->rtsp_context);
    if (ret != 0) {
      fprintf(stderr, "Error: SendFrameRTSP failed!\n");
    }

    // Cleanup
    TDL_DestroyImage(image);
    ReleaseCameraFrame(args->tdl_handle, args->vi_chn);
  }
  return NULL;
}

int main(int argc, char *argv[]) {
  TDLHandle tdl_handle = TDL_CreateHandle(0);

  int ret = InitCamera(tdl_handle, WIDTH, HEIGHT, IMAGE_YUV420SP_UV, 3);
  if (ret != 0) {
    fprintf(stderr, "Error: InitCamera failed!\n");
    TDL_DestroyHandle(tdl_handle);
    return ret;
  }

  ProcessArgs process_args;
  process_args.tdl_handle = tdl_handle;
  process_args.vi_chn = 0;
  process_args.rtsp_context = &rtsp_context;

  InitQueue(&image_queue);

  pthread_t capture_tid, process_tid;

  printf("Starting threads...\n");
  if (pthread_create(&capture_tid, NULL, capture_thread, &process_args) != 0) {
    fprintf(stderr, "Error: Failed to create capture thread!\n");
    DestroyQueue(&image_queue);
    DestoryCamera(tdl_handle);
    TDL_DestroyHandle(tdl_handle);
    return -1;
  }
  if (pthread_create(&process_tid, NULL, process_thread, &process_args) != 0) {
    fprintf(stderr, "Error: Failed to create process thread!\n");
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

  printf("按任意键退出...\n");

  while (1) {
    fd_set rfds;
    struct timeval tv = {0, 0};
    FD_ZERO(&rfds);
    FD_SET(STDIN_FILENO, &rfds);
    int key_pressed = select(STDIN_FILENO + 1, &rfds, NULL, NULL, &tv);
    if (key_pressed > 0 && FD_ISSET(STDIN_FILENO, &rfds)) {
      break;
    }
    usleep(100000);  // Check input every 100ms
  }

  // Restore terminal
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);

  printf("Stopping threads...\n");
  // Signal threads to stop
  to_exit = true;
  ExitQueue(&image_queue);

  pthread_join(capture_tid, NULL);
  pthread_join(process_tid, NULL);

  DestroyQueue(&image_queue);
  DestoryCamera(tdl_handle);
  TDL_DestroyHandle(tdl_handle);
  return ret;
}
