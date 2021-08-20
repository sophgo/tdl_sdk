#define _GNU_SOURCE
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include "cvi_audio.h"
#include "cviai.h"

#define PERIOD_SIZE 640
#define SAMPLE_RATE 16000
#define FRAME_SIZE SAMPLE_RATE * 2 * 3  // PCM_FORMAT_S16_LE (2bytes) 3 seconds

bool gRun = true;

#define VOLUMEMAX 32768.0  // db [-80 0] <---> [0 120]
float SAMPLE_AUDIO_Calculate_DB(CVI_U8 *buffer, int frames) {
  int framesShort = frames / 2;
  float fVal = 0;
  float fDB = 0;
  float ret = 0;
  if (framesShort > 0) {
    int sum = 0;
    short *pos = (short *)buffer;  // frm cvi_u8 to short
    for (int i = 0; i < framesShort; i++) {
      sum += abs(*pos);
      pos++;
    }
    fVal = (float)sum / (framesShort * VOLUMEMAX);
    fDB = 20 * log10(fVal);
    ret = fDB * 1.5 + 120.0;
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

// Get frame and set it to global buffer
void *thread_uplink_audio(void *arg) {
  CVI_S32 s32Ret;
  AUDIO_FRAME_S stFrame;
  AEC_FRAME_S stAecFrm;
  int loop = SAMPLE_RATE / PERIOD_SIZE * 3;  // 3 seconds
  int size = PERIOD_SIZE * 2;                // PCM_FORMAT_S16_LE (2bytes)

  // Set video frame interface
  CVI_U8 buffer[FRAME_SIZE];  // 3 seconds

  while (gRun) {
    for (int i = 0; i < loop; ++i) {
      s32Ret = CVI_AI_GetFrame(0, 0, &stFrame, &stAecFrm, CVI_FALSE);  // Get audio frame
      if (s32Ret != CVI_SUCCESS) {
        printf("CVI_AI_GetFrame --> none!!\n");
        continue;
      } else {
        memcpy(buffer + i * size, (CVI_U8 *)stFrame.u64VirAddr[0],
               size);  // Set the period size date to global buffer
      }
      s32Ret = CVI_AI_ReleaseFrame(0, 0, &stFrame, &stAecFrm);
    }
    float dB = SAMPLE_AUDIO_Calculate_DB(buffer, FRAME_SIZE);
    printf("3 seconds Average dB value: %f\n", dB);
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
  if (s32Ret != CVI_SUCCESS) printf("CVI_AI_SetPubAttr failed with %#x!\n", s32Ret);

  s32Ret = CVI_AI_Enable(0);
  if (s32Ret != CVI_SUCCESS) printf("CVI_AI_Enable failed with %#x!\n", s32Ret);

  s32Ret = CVI_AI_EnableChn(0, 0);
  if (s32Ret != CVI_SUCCESS) printf("CVI_AI_EnableChn failed with %#x!\n", s32Ret);

  s32Ret = CVI_AI_SetVolume(0, 4);
  if (s32Ret != CVI_SUCCESS) printf("CVI_AI_SetVolume failed with %#x!\n", s32Ret);

  printf("SET_AUDIO_ATTR success!!\n");
  return CVI_SUCCESS;
}

int main(int argc, char **argv) {
  // Set signal catch
  signal(SIGINT, SampleHandleSig);
  signal(SIGTERM, SampleHandleSig);

  if (CVI_AUDIO_INIT() == CVI_SUCCESS) {
    printf("CVI_AUDIO_INIT success!!\n");
  } else {
    printf("CVI_AUDIO_INIT failure!!\n");
    return 0;
  }

  SET_AUDIO_ATTR();

  pthread_t pcm_output_thread;
  pthread_create(&pcm_output_thread, NULL, thread_uplink_audio, NULL);
  pthread_join(pcm_output_thread, NULL);
  return 0;
}
