#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cvi_tdl.h"

#define AUDIOFORMATSIZE 2
#define SECOND 3
#define CVI_AUDIO_BLOCK_MODE -1
#define PERIOD_SIZE 640
#define SAMPLE_RATE 16000
#define FRAME_SIZE SAMPLE_RATE *AUDIOFORMATSIZE *SECOND  // PCM_FORMAT_S16_LE (2bytes) 3 seconds

static bool read_binary_file(const char *szfile, void *p_buffer, int buffer_len) {
  FILE *fp = fopen(szfile, "rb");
  if (fp == NULL) {
    printf("read file failed,%s\n", szfile);
    return false;
  }
  fseek(fp, 0, SEEK_END);
  int len = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  if (len != buffer_len) {
    printf("size not equal,expect:%d,has %d\n", buffer_len, len);
    return false;
  }
  fread(p_buffer, len, 1, fp);
  fclose(fp);
  return true;
}
static int test_binary_short_audio_data(const char *szfile, CVI_U8 *p_buffer,
                                        cvitdl_handle_t tdl_handle) {
  VIDEO_FRAME_INFO_S Frame;
  Frame.stVFrame.pu8VirAddr[0] = p_buffer;  // Global buffer
  Frame.stVFrame.u32Height = 1;
  Frame.stVFrame.u32Width = FRAME_SIZE;
  if (!read_binary_file(szfile, p_buffer, FRAME_SIZE)) {
    printf("read file failed\n");
    return -1;
  }
  int index = -1;
  int ret = CVI_TDL_SoundClassification_V2(tdl_handle, &Frame, &index);
  if (ret != 0) {
    printf("sound classification failed\n");
    return -1;
  }
  return index;
}

int main(int argc, char *argv[]) {
  CVI_U8 buffer[FRAME_SIZE];  // 3 seconds

  cvitdl_handle_t tdl_handle = NULL;
  CVI_S32 ret = CVI_TDL_CreateHandle3(&tdl_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create tdl handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_SOUNDCLASSIFICATION_V2, argv[1]);
  if (ret != CVI_SUCCESS) {
    printf("open modelfile failed %#x!\n", ret);
    return ret;
  } else {
    printf("model opened\n");
  }

  int cls = test_binary_short_audio_data(argv[2], buffer, tdl_handle);
  printf("result is:%d\n", cls);
  CVI_TDL_DestroyHandle(tdl_handle);

  return ret;
}
