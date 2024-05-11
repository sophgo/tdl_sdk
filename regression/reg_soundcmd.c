#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"

cvitdl_handle_t tdl_handle = NULL;

static int run(const char *img_dir, float *Acc) {
  DIR *dirp;
  struct dirent *entry;
  dirp = opendir(img_dir);
  int total = 0, true_total = 0;
  int _true[22] = {0};
  int _total[22] = {0};
  while ((entry = readdir(dirp)) != NULL) {
    if (entry->d_type != 8 && entry->d_type != 0) continue;
    char line[500] = "\0";
    strcat(line, img_dir);
    strcat(line, "/");
    strcat(line, entry->d_name);

    char *delim = "_";
    char *pch;
    pch = strtok(entry->d_name, delim);
    FILE *fp = fopen(line, "rb");
    fseek(fp, 0, SEEK_END);
    int size = (int)ftell(fp) * sizeof(char);
    CVI_U8 *temp = (CVI_U8 *)malloc(size);
    fseek(fp, 0, SEEK_SET);
    fread(temp, 1, size, fp);
    fclose(fp);
    VIDEO_FRAME_INFO_S frame;
    frame.stVFrame.pu8VirAddr[0] = temp;
    frame.stVFrame.u32Height = 1;
    frame.stVFrame.u32Width = size;
    int index = -1;
    CVI_TDL_SoundClassification(tdl_handle, &frame, &index);
    free(temp);

    if (index == atoi(pch)) {
      true_total++;
      _true[index]++;
    }
    _total[atoi(pch)]++;
    total++;
  }
  *Acc = (float)true_total / (float)total;
  closedir(dirp);

  for (int i = 0; i < 22; ++i) {
    printf("index: %d Acc: %f\n", i, (float)_true[i] / (float)_total[i]);
  }
  return CVI_TDL_SUCCESS;
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <soundcmd model path> <data dir>.\n", argv[0]);
    return CVI_TDL_FAILURE;
  }

  CVI_S32 ret = CVI_TDL_SUCCESS;

  ret = CVI_TDL_CreateHandle(&tdl_handle);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Create handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_SOUNDCLASSIFICATION, argv[1]);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Set model esc failed with %#x!\n", ret);
    return ret;
  }

  float Acc = 0.0;
  run(argv[2], &Acc);
  printf("Num of soundcmd -> Acc: %f\n", Acc);

  CVI_TDL_DestroyHandle(tdl_handle);
}
