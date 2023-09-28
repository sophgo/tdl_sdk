#define _GNU_SOURCE
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#include <iostream>
#include <map>
#include <queue>
#include <sstream>
#include <string>
#include <vector>
#include "cvi_audio.h"
#include "cviai.h"
#include "sample_utils.h"
#define AUDIOFORMATSIZE 2
// #define SECOND 3
#define CVI_AUDIO_BLOCK_MODE -1
#define PERIOD_SIZE 640
#define SIZE PERIOD_SIZE *AUDIOFORMATSIZE

// #define SAMPLE_RATE 8000
// #define FRAME_SIZE SAMPLE_RATE *AUDIOFORMATSIZE *SECOND  // PCM_FORMAT_S16_LE (2bytes) 3 seconds

// ESC class name
enum Classes { NO_BABY_CRY, BABY_CRY };
// This model has 6 classes, including Sneezing, Coughing, Clapping, Baby Cry, Glass breaking,
// Office. There will be a Normal option in the end, becuase the score is lower than threshold, we
// will set it to Normal.
static const char *enumStr[] = {"无指令", "小晶小晶", "拨打电话", "关闭屏幕",
                                "无指令", "我要拍照", "我要录像"};

static std::queue<std::array<int, SIZE>> que;
bool gRun = true;     // signal
bool record = false;  // record to output
char *outpath;        // output file path
static int start_index;
static int SAMPLE_RATE;
static int SECOND;
static int FRAME_SIZE;

// Init cviai handle.
cviai_handle_t ai_handle = NULL;
MUTEXAUTOLOCK_INIT(ResultMutex);
static void SampleHandleSig(CVI_S32 signo) {
  signal(SIGINT, SIG_IGN);
  signal(SIGTERM, SIG_IGN);

  if (SIGINT == signo || SIGTERM == signo) {
    gRun = false;
  }
}

void *thread_audio(void *arg) {
  CVI_S32 s32Ret;
  AUDIO_FRAME_S stFrame;
  AEC_FRAME_S stAecFrm;

  std::array<int, SIZE> buffer_temp;
  while (gRun) {
    s32Ret = CVI_AI_GetFrame(0, 0, &stFrame, &stAecFrm, CVI_AUDIO_BLOCK_MODE);  // Get audio frame
    if (s32Ret != CVIAI_SUCCESS) {
      printf("CVI_AI_GetFrame --> none!!\n");
      continue;
    } else {
      // 3 seconds
      // memcpy(buffer_temp, (CVI_U8 *)stFrame.u64VirAddr[0], size);
      {
        MutexAutoLock(ResultMutex, lock);
        std::copy(stFrame.u64VirAddr[0], stFrame.u64VirAddr[0] + SIZE, buffer_temp.begin());
        que.push(buffer_temp);
        que.pop();
      }
    }
  }

  s32Ret = CVI_AI_ReleaseFrame(0, 0, &stFrame, &stAecFrm);
  if (s32Ret != CVIAI_SUCCESS) printf("CVI_AI_ReleaseFrame Failed!!\n");

  pthread_exit(NULL);
}

// Get frame and set it to global buffer
void *thread_infer(void *arg) {
  uint32_t loop = SAMPLE_RATE / PERIOD_SIZE * SECOND;  // 3 seconds
  uint32_t size = 640 * 2;  // PERIOD_SIZE * AUDIOFORMATSIZE;       // PCM_FORMAT_S16_LE (2bytes)

  // Set video frame interface
  CVI_U8 buffer[FRAME_SIZE];  // 3 seconds
  memset(buffer, 0, FRAME_SIZE);
  VIDEO_FRAME_INFO_S Frame;
  Frame.stVFrame.pu8VirAddr[0] = buffer;  // Global buffer
  Frame.stVFrame.u32Height = 1;
  Frame.stVFrame.u32Width = FRAME_SIZE;

  printf("SAMPLE_RATE %d, SIZE:%d, FRAME_SIZE:%d \n", SAMPLE_RATE, SIZE, FRAME_SIZE);

  // classify the sound result
  int index = 0;
  int pre_label = -1;
  int maxval_sound = 0;

  std::array<int, SIZE> buffer_temp;

  while (gRun) {
    {
      MutexAutoLock(ResultMutex, lock);
      for (uint32_t i = 0; i < loop; i++) {
        // memcpy(buffer + i * size, (CVI_U8 *)que[i],
        //          size);  // Set the period size date to global buffer
        buffer_temp = que.front();
        que.pop();
        std::copy(std::begin(buffer_temp), std::end(buffer_temp), buffer + i * size);
        que.push(buffer_temp);
      }
    }

    int16_t *psound = (int16_t *)buffer;
    float meanval = 0;
    for (int i = 0; i < SAMPLE_RATE * SECOND; i++) {
      meanval += psound[i];
      if (psound[i] > maxval_sound) {
        maxval_sound = psound[i];
      }
    }
    // printf("maxvalsound:%d,meanv:%f\n", maxval_sound, meanval / (SAMPLE_RATE * SECOND));
    if (!record) {
      int ret = CVI_AI_SoundClassification_V2(ai_handle, &Frame, &pre_label);  // Detect the audio
      if (ret == CVIAI_SUCCESS) {
        printf("esc class: %s\n", enumStr[pre_label + start_index]);
        if (pre_label != 0) {
          usleep(2500 * 1000);

        } else {
          usleep(250 * 1000);
        }

      } else {
        printf("detect failed\n");
        gRun = false;
      }
    } else {
      char output_cur[128];
      sprintf(output_cur, "%s%d%s", outpath, index, ".raw");
      // sprintf(output_cur, "%s%s", output_cur, ".raw");
      FILE *fp = fopen(output_cur, "wb");
      printf("to write:%s\n", output_cur);
      fwrite((char *)buffer, 1, FRAME_SIZE, fp);
      fclose(fp);
      usleep(1000 * 1000);
      // gRun = false;
    }

    index++;
  }

  pthread_exit(NULL);
}

CVI_S32 SET_AUDIO_ATTR(CVI_VOID) {
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
  CVI_S32 s32Ret;
  // STEP 2:cvitek_audin_uplink_start
  //_set_audin_config
  s32Ret = CVI_AI_SetPubAttr(0, &AudinAttr);
  if (s32Ret != CVIAI_SUCCESS) printf("CVI_AI_SetPubAttr failed with %#x!\n", s32Ret);

  s32Ret = CVI_AI_Enable(0);
  if (s32Ret != CVIAI_SUCCESS) printf("CVI_AI_Enable failed with %#x!\n", s32Ret);

  s32Ret = CVI_AI_EnableChn(0, 0);
  if (s32Ret != CVIAI_SUCCESS) printf("CVI_AI_EnableChn failed with %#x!\n", s32Ret);

  s32Ret = CVI_AI_SetVolume(0, 4);
  if (s32Ret != CVIAI_SUCCESS) printf("CVI_AI_SetVolume failed with %#x!\n", s32Ret);

  printf("SET_AUDIO_ATTR success!!\n");
  return CVIAI_SUCCESS;
}

int main(int argc, char **argv) {
  if (argc != 5 && argc != 7) {
    printf(
        "Usage: %s ESC_MODEL_PATH SAMPLE_RATE ORDER_TYPE SECOND RECORD OUTPUT\n"
        "\t\tESC_MODEL_PATH, esc model path.\n"
        "\t\tSAMPLE_RATE, sample rate.\n"
        "\t\tSECOND, input time (s).\n"
        "\t\tORDER_TYPE, 0 for ipc, 1 for car.\n"
        "\t\tRECORD, record input to file, 0: disable 1. enable.\n"
        "\t\tOUTPUT, output file path: {output file path}.raw\n",
        argv[0]);
    return CVIAI_FAILURE;
  }

  SAMPLE_RATE = atoi(argv[2]);
  SECOND = atoi(argv[3]);
  int ORDER_TYPE = atoi(argv[4]);
  start_index = ORDER_TYPE == 0 ? 0 : 4;

  FRAME_SIZE = SAMPLE_RATE * AUDIOFORMATSIZE * SECOND;

  // Set signal catch
  signal(SIGINT, SampleHandleSig);
  signal(SIGTERM, SampleHandleSig);

  uint32_t loop = SAMPLE_RATE / PERIOD_SIZE * SECOND;
  for (uint32_t i = 0; i < loop; i++) {
    std::array<int, SIZE> tmp_arr = {0};
    que.push(tmp_arr);
  }

  CVI_S32 ret = CVIAI_SUCCESS;
  if (CVI_AUDIO_INIT() == CVIAI_SUCCESS) {
    printf("CVI_AUDIO_INIT success!!\n");
  } else {
    printf("CVI_AUDIO_INIT failure!!\n");
    return 0;
  }

  SET_AUDIO_ATTR();

  ret = CVI_AI_CreateHandle(&ai_handle);

  if (ret != CVIAI_SUCCESS) {
    printf("Create ai handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_SOUNDCLASSIFICATION_V2, argv[1]);
  if (ret != CVIAI_SUCCESS) {
    CVI_AI_DestroyHandle(ai_handle);
    return ret;
  }

  // CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_SOUNDCLASSIFICATION_V2, true);
  CVI_AI_SetPerfEvalInterval(ai_handle, CVI_AI_SUPPORTED_MODEL_SOUNDCLASSIFICATION_V2,
                             10);  // only used to performance evaluation
  if (argc == 7) {
    record = atoi(argv[5]) ? true : false;
    outpath = (char *)malloc(strlen(argv[3]));
    strcpy(outpath, argv[6]);
  }

  pthread_t pcm_output_thread, get_sound;
  pthread_create(&pcm_output_thread, NULL, thread_infer, NULL);
  pthread_create(&get_sound, NULL, thread_audio, NULL);

  pthread_join(pcm_output_thread, NULL);
  pthread_join(get_sound, NULL);
  if (argc == 7) {
    free(outpath);
  }

  CVI_AI_DestroyHandle(ai_handle);
  return 0;
}
