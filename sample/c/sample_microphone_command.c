#include <cvi_comm_vpss.h>
#include <pthread.h>
#include <signal.h>
#include "cvi_audio.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tdl_sdk.h"
#include "tdl_utils.h"

#define AUDIOFORMATSIZE 2
#define CVI_AUDIO_BLOCK_MODE -1
#define PERIOD_SIZE 640

bool gRun = true;
static int g_sample_rate = 16000;
static float g_seconds = 2;

typedef struct {
  TDLHandle tdl_handle;
  TDLModel model_id;
} RUN_TDL_THREAD_ARG_S;

typedef struct {
  uint8_t *buf[2];
  int frame_size;
  bool ready[2];
  int write_idx;
  uint64_t seq[2];
  pthread_mutex_t mtx;
  pthread_cond_t cv;
} PingPong;

static PingPong pp;

int get_model_info(char *model_path, TDLModel *model_index) {
  if (strstr(model_path, "cls_sound_babay_cry") != NULL &&
      strstr(model_path, "cls_sound_babay_cry_8k") == NULL) {
    *model_index = TDL_MODEL_CLS_SOUND_BABAY_CRY;
  } else if (strstr(model_path, "cls_sound") != NULL) {
    *model_index = TDL_MODEL_CLS_SOUND_COMMAND;
  } else {
    return -1;
  }
  return 0;
}

static void SampleHandleSig(CVI_S32 signo) {
  signal(SIGINT, SIG_IGN);
  signal(SIGTERM, SIG_IGN);

  if (SIGINT == signo || SIGTERM == signo) {
    gRun = false;
  }
}

void *capture(void *_) {
  uint64_t seq = 0;
  CVI_S32 s32Ret;
  AUDIO_FRAME_S stFrame;
  AEC_FRAME_S stAecFrm;
  int loop = g_sample_rate / PERIOD_SIZE * g_seconds;
  int size = PERIOD_SIZE * AUDIOFORMATSIZE;

  uint8_t *buffer = (uint8_t *)malloc(pp.frame_size);
  memset(buffer, 0, pp.frame_size);

  while (gRun) {
    for (int i = 0; i < loop; ++i) {
      s32Ret = CVI_AI_GetFrame(0, 0, &stFrame, &stAecFrm, CVI_AUDIO_BLOCK_MODE);
      if (s32Ret != 0) {
        printf("CVI_AI_GetFrame --> none!!\n");
        continue;
      } else {
        memcpy(buffer + i * size, (uint8_t *)stFrame.u64VirAddr[0], size);
      }
    }

    pthread_mutex_lock(&pp.mtx);
    int idx = pp.write_idx;
    memcpy(pp.buf[idx], buffer, pp.frame_size);
    pp.seq[idx] = seq++;
    pp.ready[idx] = true;
    pp.write_idx = 1 - idx;
    pthread_cond_signal(&pp.cv);
    pthread_mutex_unlock(&pp.mtx);
  }
  s32Ret = CVI_AI_ReleaseFrame(0, 0, &stFrame, &stAecFrm);
  if (s32Ret != 0) printf("CVI_AI_ReleaseFrame Failed!!\n");
  free(buffer);
  return NULL;
}

void *infer(void *args) {
  RUN_TDL_THREAD_ARG_S *pstArgs = (RUN_TDL_THREAD_ARG_S *)args;
  TDLClassInfo obj_info = {0};

  while (gRun) {
    pthread_mutex_lock(&pp.mtx);
    while (!pp.ready[0] && !pp.ready[1]) {
      pthread_cond_wait(&pp.cv, &pp.mtx);
    }
    if (!gRun) {
      pthread_mutex_unlock(&pp.mtx);
      break;
    }
    int i = pp.ready[0] ? 0 : 1;

    TDLImage image = TDL_ReadAudioFrame(pp.buf[i], pp.frame_size);
    if (image == NULL) {
      printf("read audio failed\n");
      gRun = false;
    }

    pp.ready[i] = false;
    pthread_mutex_unlock(&pp.mtx);

    memset(&obj_info, 0, sizeof(TDLClassInfo));
    int ret = TDL_Classification(pstArgs->tdl_handle, pstArgs->model_id, image,
                                 &obj_info);
    if (ret != 0) {
      printf("TDL_Classification failed with %#x!\n", ret);
    } else {
      printf("pred_label: %d, score: %.2f\n", obj_info.class_id,
             obj_info.score);
    }
    TDL_DestroyImage(image);
  }

  return NULL;
}

int32_t SET_AUDIO_ATTR(int sample_rate) {
  AIO_ATTR_S AudinAttr;
  AudinAttr.enSamplerate = (AUDIO_SAMPLE_RATE_E)sample_rate;
  AudinAttr.u32ChnCnt = 1;
  AudinAttr.enSoundmode = AUDIO_SOUND_MODE_MONO;
  AudinAttr.enBitwidth = AUDIO_BIT_WIDTH_16;
  AudinAttr.enWorkmode = AIO_MODE_I2S_MASTER;
  AudinAttr.u32EXFlag = 0;
  AudinAttr.u32FrmNum = 10;
  AudinAttr.u32PtNumPerFrm = PERIOD_SIZE;
  AudinAttr.u32ClkSel = 0;
  AudinAttr.enI2sType = AIO_I2STYPE_INNERCODEC;
  int ret;

  ret = CVI_AI_SetPubAttr(0, &AudinAttr);
  if (ret != 0) printf("CVI_AI_SetPubAttr failed with %#x!\n", ret);

  ret = CVI_AI_Enable(0);
  if (ret != 0) printf("CVI_AI_Enable failed with %#x!\n", ret);

  ret = CVI_AI_EnableChn(0, 0);
  if (ret != 0) printf("CVI_AI_EnableChn failed with %#x!\n", ret);

  ret = CVI_AI_SetVolume(0, 12);
  if (ret != 0) printf("CVI_AI_SetVolume failed with %#x!\n", ret);

  printf("SET_AUDIO_ATTR success!!\n");
  return 0;
}

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf(
      "  %s -m <model_path> -j <json_config> [-r <sample_rate>] [-s "
      "<seconds>]\n",
      prog_name);
  printf(
      "  %s --model_path <model_path> --json_config <json_config> "
      "[--sample-rate <rate>] [--seconds <time>]\n\n",
      prog_name);
  printf("Options:\n");
  printf("  -m, --model_path   Path to voice command model\n");
  printf("  -j, --json_config  Path to json config file\n");
  printf("  -r, --sample-rate  Sample rate in Hz (default: 16000)\n");
  printf(
      "  -s, --seconds      Audio duration per inference in seconds (default: "
      "1.5)\n");
  printf("  -h, --help         Show this help message\n");
}

int main(int argc, char **argv) {
  char *model_path = NULL;
  char *json_config = NULL;

  struct option long_options[] = {{"model_path", required_argument, 0, 'm'},
                                  {"json_config", required_argument, 0, 'j'},
                                  {"sample-rate", required_argument, 0, 'r'},
                                  {"seconds", required_argument, 0, 's'},
                                  {"help", no_argument, 0, 'h'},
                                  {NULL, 0, NULL, 0}};

  int opt;
  while ((opt = getopt_long(argc, argv, "m:j:r:s:h", long_options, NULL)) !=
         -1) {
    switch (opt) {
      case 'm':
        model_path = optarg;
        break;
      case 'j':
        json_config = optarg;
        break;
      case 'r':
        g_sample_rate = atoi(optarg);
        break;
      case 's':
        g_seconds = atof(optarg);
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

  if (!model_path || !json_config) {
    fprintf(stderr, "Error: Model path and json config are required\n");
    print_usage(argv[0]);
    return -1;
  }

  printf("Running with:\n");
  printf("  Model path:  %s\n", model_path);
  printf("  Json config: %s\n", json_config);
  printf("  Sample rate: %d\n", g_sample_rate);
  printf("  Seconds:     %.1f\n", g_seconds);

  TDLModel model_id;
  if (get_model_info(model_path, &model_id) != 0) {
    printf("unsupported model: %s\n", model_path);
    return -1;
  }

  int frame_size = g_sample_rate * AUDIOFORMATSIZE * g_seconds;

  memset(&pp, 0, sizeof(pp));
  pp.frame_size = frame_size;
  pp.buf[0] = (uint8_t *)malloc(frame_size);
  pp.buf[1] = (uint8_t *)malloc(frame_size);
  if (!pp.buf[0] || !pp.buf[1]) {
    printf("malloc buffer failed\n");
    free(pp.buf[0]);
    free(pp.buf[1]);
    return -1;
  }

  pthread_mutex_init(&pp.mtx, NULL);
  pthread_cond_init(&pp.cv, NULL);

  signal(SIGINT, SampleHandleSig);
  signal(SIGTERM, SampleHandleSig);

  int ret = 0;

  TDLHandle tdl_handle = TDL_CreateHandle(0);
  ret = TDL_OpenModel(tdl_handle, model_id, model_path, json_config, 0);
  if (ret != 0) {
    printf("open model failed with %#x!\n", ret);
    goto exit0;
  }

  ret = TDL_SetModelThreshold(tdl_handle, model_id, 0.5);
  if (ret != 0) {
    printf("TDL_SetModelThreshold failed with %#x!\n", ret);
    goto exit1;
  }

  RUN_TDL_THREAD_ARG_S tdl_args = {.tdl_handle = tdl_handle,
                                   .model_id = model_id};

  if (CVI_AUDIO_INIT() == 0) {
    printf("CVI_AUDIO_INIT success!!\n");
  } else {
    printf("CVI_AUDIO_INIT failure!!\n");
    goto exit1;
  }
  SET_AUDIO_ATTR(g_sample_rate);

  printf("Initialization completed, waiting for voice commands...\n");

  pthread_t stFrameThread, stTDLThread;
  pthread_create(&stFrameThread, NULL, capture, NULL);
  pthread_create(&stTDLThread, NULL, infer, &tdl_args);

  pthread_join(stFrameThread, NULL);
  pthread_join(stTDLThread, NULL);

exit1:
  TDL_CloseModel(tdl_handle, model_id);

exit0:
  TDL_DestroyHandle(tdl_handle);
  free(pp.buf[0]);
  free(pp.buf[1]);
  return ret;
}
