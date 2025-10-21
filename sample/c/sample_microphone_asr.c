
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
#define SECOND 1
#define CVI_AUDIO_BLOCK_MODE -1
#define PERIOD_SIZE 640
#define SAMPLE_RATE 16000
#define FRAME_SIZE \
  SAMPLE_RATE *AUDIOFORMATSIZE *SECOND  // PCM_FORMAT_S16_LE (2bytes) 3 seconds

bool gRun = true;  // signal

typedef struct {
  TDLHandle tdl_handle;
  TDLModel model_id_encoder;
} RUN_TDL_THREAD_ARG_S;

typedef struct {
  uint8_t buf[2][FRAME_SIZE];
  bool ready[2];  // 标记该槽是否有新数据
  int write_idx;  // 0/1
  uint64_t seq[2];
  pthread_mutex_t mtx;
  pthread_cond_t cv;
} PingPong;

static PingPong pp;

int get_model_info(char *model_path, TDLModel *model_index) {
  int ret = 0;
  if (strstr(model_path, "speech_zipformer_encoder") != NULL) {
    *model_index = TDL_MODEL_RECOGNITION_SPEECH_ZIPFORMER_ENCODER;
  } else if (strstr(model_path, "speech_zipformer_decoder") != NULL) {
    *model_index = TDL_MODEL_RECOGNITION_SPEECH_ZIPFORMER_DECODER;
  } else if (strstr(model_path, "speech_zipformer_joiner") != NULL) {
    *model_index = TDL_MODEL_RECOGNITION_SPEECH_ZIPFORMER_JOINER;
  } else {
    ret = -1;
  }
  return ret;
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
  int loop = SAMPLE_RATE / PERIOD_SIZE * SECOND;  // 3 seconds
  int size = PERIOD_SIZE * AUDIOFORMATSIZE;       // PCM_FORMAT_S16_LE (2bytes)

  // Set video frame interface
  uint8_t buffer[FRAME_SIZE];  // 1 seconds
  memset(buffer, 0, FRAME_SIZE);

  while (gRun) {
    for (int i = 0; i < loop; ++i) {
      s32Ret = CVI_AI_GetFrame(0, 0, &stFrame, &stAecFrm,
                               CVI_AUDIO_BLOCK_MODE);  // Get audio frame
      if (s32Ret != 0) {
        printf("CVI_AI_GetFrame --> none!!\n");
        continue;
      } else {
        memcpy(buffer + i * size, (uint8_t *)stFrame.u64VirAddr[0],
               size);  // Set the period size date to global buffer
      }
    }

    pthread_mutex_lock(&pp.mtx);
    int idx = pp.write_idx;
    memcpy(pp.buf[idx], buffer, FRAME_SIZE);
    pp.seq[idx] = seq++;
    pp.ready[idx] = true;
    pp.write_idx = 1 - idx;  // 翻转
    pthread_cond_signal(&pp.cv);
    pthread_mutex_unlock(&pp.mtx);
  }
  s32Ret = CVI_AI_ReleaseFrame(0, 0, &stFrame, &stAecFrm);
  if (s32Ret != 0) printf("CVI_AI_ReleaseFrame Failed!!\n");
  return NULL;
}

void *infer(void *args) {
  RUN_TDL_THREAD_ARG_S *pstArgs = (RUN_TDL_THREAD_ARG_S *)args;
  TDLText text_meta = {0};

  while (gRun) {
    pthread_mutex_lock(&pp.mtx);
    // 等待任一槽 ready
    while (!pp.ready[0] && !pp.ready[1]) {
      pthread_cond_wait(&pp.cv, &pp.mtx);
    }
    if (!gRun) {
      pthread_mutex_unlock(&pp.mtx);
      break;
    }
    if (!gRun && !pp.ready[0] && !pp.ready[1]) {
      pthread_mutex_unlock(&pp.mtx);
      break;
    }
    int i = pp.ready[0] ? 0 : 1;

    TDLImage image = TDL_ReadAudioFrame(pp.buf[i], FRAME_SIZE);
    if (image == NULL) {
      printf("read audio failed\n");
      gRun = false;
    }

    pp.ready[i] = false;
    pthread_mutex_unlock(&pp.mtx);

    int ret = TDL_SpeechRecognition(
        pstArgs->tdl_handle, pstArgs->model_id_encoder, image, &text_meta);
    if (ret != 0) {
      printf("TDL_SpeechRecognition failed with %#x!\n", ret);
      gRun = false;
    } else {
      if (text_meta.text_info) {
        printf("%s", text_meta.text_info);
        fflush(stdout);
      }
    }
    TDL_ReleaseCharacterMeta(&text_meta);
  }

  return NULL;
}

int32_t SET_AUDIO_ATTR() {
  // STEP 1: cvitek_audin_set
  //_update_audin_config
  AIO_ATTR_S AudinAttr;
  AudinAttr.enSamplerate = (AUDIO_SAMPLE_RATE_E)SAMPLE_RATE;
  AudinAttr.u32ChnCnt = 1;
  AudinAttr.enSoundmode = AUDIO_SOUND_MODE_MONO;
  AudinAttr.enBitwidth = AUDIO_BIT_WIDTH_16;
  AudinAttr.enWorkmode = AIO_MODE_I2S_MASTER;
  AudinAttr.u32EXFlag = 0;
  AudinAttr.u32FrmNum = 10;                /* only use in bind mode */
  AudinAttr.u32PtNumPerFrm = PERIOD_SIZE;  // sample rate / fps
  AudinAttr.u32ClkSel = 0;
  AudinAttr.enI2sType = AIO_I2STYPE_INNERCODEC;
  int ret;
  // STEP 2:cvitek_audin_uplink_start
  //_set_audin_config
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
      "  %s -m "
      "<model_path_encoder>,<model_path_decoder>,<model_path_joiner> -t "
      "<tokens_path>\n",
      prog_name);
  printf(
      "  %s --model_path "
      "<model_path_encoder>,<model_path_decoder>,<model_path_joiner> "
      "--tokens_path <tokens_path>\n\n",
      prog_name);
  printf("Options:\n");
  printf("  -m, --model_path  Path to encoder, decoder and joiner model\n");
  printf("  -t, --tokens_path Path to tokens file\n");
  printf("  -h, --help        Show this help message\n");
}

int main(int argc, char **argv) {
  char *model_path_encoder = NULL;
  char *model_path_decoder = NULL;
  char *model_path_joiner = NULL;
  char *model_path = NULL;
  char *tokens_path = NULL;

  struct option long_options[] = {{"model_path", required_argument, 0, 'm'},
                                  {"tokens_path", required_argument, 0, 't'},
                                  {"help", no_argument, 0, 'h'},
                                  {NULL, 0, NULL, 0}};

  int opt;
  while ((opt = getopt_long(argc, argv, "m:t:h", long_options, NULL)) != -1) {
    switch (opt) {
      case 'm':
        model_path = optarg;
        break;
      case 't':
        tokens_path = optarg;
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

  if (!model_path || !tokens_path) {
    fprintf(stderr, "Error: All arguments are required\n");
    print_usage(argv[0]);
    return -1;
  }

  char *comm = strchr(model_path, ',');
  if (!comm || comm == model_path || !*(comm + 1)) {
    fprintf(stderr,
            "Error: Models must be in format "
            "'model_path_detect,model_path_keypoint,model_path_recognition\n");
    return -1;
  }

  const char *first_comma = strchr(model_path, ',');
  if (!first_comma || first_comma == model_path || first_comma[1] == '\0') {
    fprintf(stderr,
            "Error: Models must be in format "
            "'model_path_detect,model_path_keypoint,model_path_recognition'\n");
    return -1;
  }
  const char *second_comma = strchr(first_comma + 1, ',');
  if (!second_comma || second_comma == first_comma + 1 ||
      second_comma[1] == '\0') {
    fprintf(stderr,
            "Error: Models must be in format "
            "'model_path_detect,model_path_keypoint,model_path_recognition'\n");
    return -1;
  }

  if (strchr(second_comma + 1, ',')) {
    fprintf(stderr, "Error: Exactly three model paths are required\n");
    return -1;
  }

  char *comm1 = (char *)first_comma;
  char *comm2 = (char *)second_comma;

  model_path_encoder = model_path;
  *comm1 = '\0';
  model_path_decoder = comm1 + 1;
  *comm2 = '\0';
  model_path_joiner = comm2 + 1;

  printf("Running with:\n");
  printf("  Model path_encoder:     %s\n", model_path_encoder);
  printf("  Model path_decoder:     %s\n", model_path_decoder);
  printf("  Model path_joiner:      %s\n", model_path_joiner);
  printf("  Tokens path:            %s\n", tokens_path);

  memset(&pp, 0, sizeof(pp));
  pthread_mutex_init(&pp.mtx, NULL);
  pthread_cond_init(&pp.cv, NULL);

  // Set signal catch
  signal(SIGINT, SampleHandleSig);
  signal(SIGTERM, SampleHandleSig);

  int ret = 0;

  TDLModel model_id_encoder;
  ret = get_model_info(model_path_encoder, &model_id_encoder);
  if (ret != 0) {
    printf("None encoder model name to support\n");
    return -1;
  }

  TDLModel model_id_decoder;
  ret = get_model_info(model_path_decoder, &model_id_decoder);
  if (ret != 0) {
    printf("None decoder model name to support\n");
    return -1;
  }

  TDLModel model_id_joiner;
  ret = get_model_info(model_path_joiner, &model_id_joiner);
  if (ret != 0) {
    printf("None joiner model name to support\n");
    return -1;
  }

  TDLHandle tdl_handle = TDL_CreateHandle(0);
  ret = TDL_OpenModel(tdl_handle, model_id_encoder, model_path_encoder, NULL);
  if (ret != 0) {
    printf("open encoder model failed with %#x!\n", ret);
    goto exit1;
  }

  ret = TDL_OpenModel(tdl_handle, model_id_decoder, model_path_decoder, NULL);
  if (ret != 0) {
    printf("open decoder model failed with %#x!\n", ret);
    goto exit2;
  }

  ret = TDL_OpenModel(tdl_handle, model_id_joiner, model_path_joiner, NULL);
  if (ret != 0) {
    printf("open joiner model failed with %#x!\n", ret);
    goto exit3;
  }

  ret =
      TDL_SpeechRecognition_Init(tdl_handle, model_id_encoder, model_id_decoder,
                                 model_id_joiner, tokens_path);
  if (ret != 0) {
    printf("init joiner model failed with %#x!\n", ret);
    goto exit3;
  }

  RUN_TDL_THREAD_ARG_S tdl_args = {.tdl_handle = tdl_handle,
                                   .model_id_encoder = model_id_encoder};

  if (CVI_AUDIO_INIT() == 0) {
    printf("CVI_AUDIO_INIT success!!\n");
  } else {
    printf("CVI_AUDIO_INIT failure!!\n");
    return 0;
  }
  SET_AUDIO_ATTR();

  printf("Initialization completed, please start speaking:\n");

  pthread_t stFrameThread, stTDLThread;
  pthread_create(&stFrameThread, NULL, capture, NULL);
  pthread_create(&stTDLThread, NULL, infer, &tdl_args);

  pthread_join(stFrameThread, NULL);
  pthread_join(stTDLThread, NULL);

exit3:
  TDL_CloseModel(tdl_handle, model_id_joiner);

exit2:
  TDL_CloseModel(tdl_handle, model_id_decoder);

exit1:
  TDL_CloseModel(tdl_handle, model_id_encoder);

exit0:
  TDL_DestroyHandle(tdl_handle);
  return ret;

  return 0;
}
