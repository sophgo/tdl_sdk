#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cviai.h"
#include "ive/ive.h"

#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#include "acodec.h"
#include "cvi_audio.h"
#include "sample_comm.h"

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define PERIOD_SIZE 640
#define SAMPLE_RATE 16000
#define FRAME_SIZE SAMPLE_RATE * 2 * 3  // PCM_FORMAT_S16_LE (2bytes) 3 seconds

char ES_Classes[8][32] = {"Sneezing/Coughing", "Sneezong/Coughing", "Clapping",    "Laughing",
                          "Baby Cry",          "Glass breaking",    "Clock_alarm", "Office"};

CVI_U8 buffer[FRAME_SIZE * 2];  // 6 seconds
int w_idx = 0;                  // write frame idx
int r_idx = 0;                  // read frame idx
bool mtx = false;
bool gRun = true;

static void SampleHandleSig(CVI_S32 signo) {
  signal(SIGINT, SIG_IGN);
  signal(SIGTERM, SIG_IGN);

  if (SIGINT == signo || SIGTERM == signo) {
    gRun = false;
  }
}
void *thread_uplink_audio(void *arg) {
  CVI_S32 s32Ret;
  AUDIO_FRAME_S stFrame;
  AEC_FRAME_S stAecFrm;
  int size = PERIOD_SIZE * 2;  // PCM_FORMAT_S16_LE (2bytes)
  // cycle buffer
  while (gRun) {
    memset(&stAecFrm, 0, sizeof(AEC_FRAME_S));
    memset(&stFrame, 0, sizeof(AUDIO_FRAME_S));
    s32Ret = CVI_AI_GetFrame(0, 0, &stFrame, &stAecFrm, CVI_FALSE);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_AI_GetFrame --> none!!\n");
      continue;
    } else {
      memcpy(buffer + w_idx, (CVI_U8 *)stFrame.u64VirAddr[0], size);
      if (w_idx == FRAME_SIZE * 2 - size)
        w_idx = 0;
      else
        w_idx = w_idx + size;
    }
    if (w_idx == r_idx) {
      r_idx = (r_idx + 32000) % 192000;
    }
    if (w_idx - r_idx > 96000 && !mtx)  // 16000 * 2 * 3
      mtx = true;
    else if (w_idx - r_idx < 0 && !mtx) {
      if (w_idx + 96000 - r_idx > 0) mtx = true;
    }
  }
  pthread_exit(NULL);
}

CVI_S32 SAMPLE_AUDIO_GET_AUDIO_FRAME_BY_FRAME(CVI_VOID) {
  printf("This section is treated as sample code flow for porting\n");

  // STEP 1: cvitek_audin_set
  //_update_audin_config
  AIO_ATTR_S AudinAttr;

  AudinAttr.enSamplerate = (AUDIO_SAMPLE_RATE_E)16000;
  AudinAttr.u32ChnCnt = 1;
  AudinAttr.enSoundmode = AUDIO_SOUND_MODE_MONO;
  AudinAttr.enBitwidth = AUDIO_BIT_WIDTH_16;
  AudinAttr.enWorkmode = AIO_MODE_I2S_MASTER;
  AudinAttr.u32EXFlag = 0;
  AudinAttr.u32FrmNum = 10;        /* only use in bind mode */
  AudinAttr.u32PtNumPerFrm = 640;  // sample rate / fps
  AudinAttr.u32ClkSel = 0;
  AudinAttr.enI2sType = AIO_I2STYPE_INNERCODEC;
  CVI_S32 s32Ret;
  // STEP 2:cvitek_audin_uplink_start
  //_set_audin_config
  s32Ret = CVI_AI_SetPubAttr(0, &AudinAttr);
  if (s32Ret != CVI_SUCCESS) printf("CVI_AI_SetPubAttr failed with %#x!\n", s32Ret);
  s32Ret = CVI_AI_Enable(0);
  if (s32Ret != CVI_SUCCESS) printf("CVI_AI_Enable failed with %#x!\n", s32Ret);
  s32Ret = CVI_AI_EnableChn(0, 0);
  if (s32Ret != CVI_SUCCESS) printf("CVI_AI_EnableChn failed with %#x!\n", s32Ret);
  return CVI_SUCCESS;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: %s <esc_model_path> \n", argv[0]);
    return CVI_FAILURE;
  }

  // Set signal catch
  signal(SIGINT, SampleHandleSig);
  signal(SIGTERM, SampleHandleSig);

  CVI_S32 ret = CVI_SUCCESS;
  if (CVI_AUDIO_INIT() == CVI_SUCCESS) {
    printf("CVI_AUDIO_INIT success!!\n");
  } else {
    printf("CVI_AUDIO_INIT failure!!\n");
    return 0;
  }

  SAMPLE_AUDIO_GET_AUDIO_FRAME_BY_FRAME();
  pthread_t pcm_output_thread;
  pthread_create(&pcm_output_thread, NULL, thread_uplink_audio, NULL);

  CVI_U8 fbuffer[FRAME_SIZE];
  VIDEO_FRAME_INFO_S Frame;
  Frame.stVFrame.pu8VirAddr[0] = fbuffer;
  Frame.stVFrame.u32Height = 1;
  Frame.stVFrame.u32Width = FRAME_SIZE;
  // Init cviai handle.
  cviai_handle_t ai_handle = NULL;
  ret = CVI_AI_CreateHandle(&ai_handle);

  if (ret != CVI_SUCCESS) {
    printf("Create ai handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_ESCLASSIFICATION, argv[1]);
  if (ret != CVI_SUCCESS) {
    printf("Set model esc failed with %#x!\n", ret);
    return ret;
  }

  // use 2 result to classify the sound
  int index = -1;
  int prev_index = -1;
  int prev2_index = -1;
  while (gRun) {
    if (!mtx) {
      usleep(300 * 1000);
      continue;
    } else {
      int diff = MIN(96000, 192000 - r_idx);
      memcpy(fbuffer, buffer + r_idx, diff);
      if (diff != 96000) memcpy(fbuffer + diff, buffer, 96000 - diff);
      if (r_idx != 160000)
        r_idx += 32000;
      else
        r_idx = 0;
      mtx = false;
    }
    CVI_AI_ESClassification(ai_handle, &Frame, &index);
    if (prev_index > -1 && prev2_index > -1 && index > -1) {
      if ((prev2_index == 0 || prev2_index == 1) && (prev_index == 0 || prev_index == 1))
        printf("esc class: %s  \n", ES_Classes[0]);
      else if (prev_index == prev2_index) {
        printf("esc class: %s  \n", ES_Classes[prev_index]);
      } else
        printf("esc class: detecting  \n");
    }
    prev2_index = prev_index;
    prev_index = index;
  }
  CVI_AI_DestroyHandle(ai_handle);
  pthread_join(pcm_output_thread, NULL);

  return 0;
}
