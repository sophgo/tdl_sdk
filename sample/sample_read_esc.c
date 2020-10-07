#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cviai.h"
#include "ive/ive.h"

#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include "acodec.h"
#include "cvi_audio.h"
#include "sample_comm.h"

#define UPDATE_INTERVAL 10
#define PERIOD_SIZE 640
#define SAMPLE_RATE 16000
#define FRAME_SIZE SAMPLE_RATE * 2  // PCM_FORMAT_S16_LE (2bytes)

char ES_Classes[50][32] = {"Dog",
                           "Rooster",
                           "Pig",
                           "Cow",
                           "Frog",
                           "Cat",
                           "Hen",
                           "Insects flying",
                           "Sheep",
                           "Crow",
                           "Rain",
                           "Sea waves",
                           "Crackling fire",
                           "Crickets",
                           "Chirping birds",
                           "Water drops",
                           "Wind",
                           "Pouring water",
                           "Toilet flush",
                           "Thunderstorm",
                           "Crying baby",
                           "Sneezing",
                           "Clapping",
                           "Breathing",
                           "Coughing",
                           "Footsteps",
                           "Laughing",
                           "Brushing teeth",
                           "Snoring",
                           "Drinking sipping",
                           "Door knock",
                           "Mouse click",
                           "Keyboard typing",
                           "Door wood creaks",
                           "Can opening",
                           "Washing machine",
                           "Vacuum cleaner",
                           "Clock alarm",
                           "Clock tick",
                           "Glass breaking",
                           "Helicopter",
                           "Chainsaw",
                           "Siren",
                           "Car horn",
                           "Engine",
                           "Train",
                           "Church_bells",
                           "Airplane",
                           "Fireworks",
                           "Hand saw"};

CVI_U8 buffer[FRAME_SIZE];
void *thread_uplink_audio(void *arg) {
  CVI_S32 s32Ret;
  AUDIO_FRAME_S stFrame;
  AEC_FRAME_S stAecFrm;
  memset(&stAecFrm, 0, sizeof(AEC_FRAME_S));
  memset(&stFrame, 0, sizeof(AUDIO_FRAME_S));
  int loop = SAMPLE_RATE / PERIOD_SIZE;
  int size = PERIOD_SIZE * 2;  // PCM_FORMAT_S16_LE (2bytes)
  for (int i = 0; i < loop; ++i) {
    s32Ret = CVI_AI_GetFrame(0, 0, &stFrame, &stAecFrm, CVI_FALSE);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_AI_GetFrame --> none!!\n");
      continue;
    } else
      memcpy(buffer + i * size, (CVI_U8 *)stFrame.u64VirAddr[0], size);
  }
  pthread_exit(NULL);
}

CVI_S32 SAMPLE_AUDIO_GET_AUDIO_FRAME_BY_FRAME(CVI_VOID) {
  printf("This section is treated as sample code flow for porting\n");
  printf("Do not execute as function internally\n");

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
  s32Ret = CVI_AI_Enable(0);
  s32Ret = CVI_AI_EnableChn(0, 0);
  // STEP 3: create a thread to get frame and put to APP buffer
  pthread_t pcm_output_thread;
  if (s32Ret != CVI_SUCCESS) printf("CVI_AO_SendFrame failed with %#x!\n", s32Ret);
  pthread_create(&pcm_output_thread, NULL, thread_uplink_audio, NULL);
  pthread_join(pcm_output_thread, NULL);
  return CVI_SUCCESS;
}

int main(int argc, char **argv) {
  CVI_S32 ret = CVI_SUCCESS;
  if (CVI_AUDIO_INIT() == CVI_SUCCESS) {
    printf("CVI_AUDIO_INIT success!!\n");
  } else {
    printf("CVI_AUDIO_INIT failure!!\n");
    return 0;
  }

  // IVE_HANDLE handle = CVI_IVE_CreateHandle();
  SAMPLE_AUDIO_GET_AUDIO_FRAME_BY_FRAME();
  VIDEO_FRAME_INFO_S Frame;
  Frame.stVFrame.pu8VirAddr[0] = buffer;
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
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }

  int index = -1;
  CVI_AI_ESClassification(ai_handle, &Frame, &index);
  printf("index:%s\n", ES_Classes[index]);

  CVI_AI_DestroyHandle(ai_handle);

  return 0;
}
