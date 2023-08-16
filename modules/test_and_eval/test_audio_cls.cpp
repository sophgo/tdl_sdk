#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "core/cviai_types_mem_internal.h"
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "evaluation/cviai_media.h"
#include "sys_utils.hpp"
#define AUDIOFORMATSIZE 2
#define SECOND 3
#define CVI_AUDIO_BLOCK_MODE -1
#define PERIOD_SIZE 640
// #define SAMPLE_RATE 16000
// #define FRAME_SIZE SAMPLE_RATE *AUDIOFORMATSIZE *SECOND  // PCM_FORMAT_S16_LE (2bytes) 3 seconds

int test_binary_short_audio_data(const std::string &strf, CVI_U8 *p_buffer,
                                 cviai_handle_t ai_handle, int sample_rate) {
  VIDEO_FRAME_INFO_S Frame;
  int frame_size = sample_rate * AUDIOFORMATSIZE * SECOND;
  Frame.stVFrame.pu8VirAddr[0] = p_buffer;  // Global buffer
  Frame.stVFrame.u32Height = 1;
  Frame.stVFrame.u32Width = frame_size;
  if (!read_binary_file(strf, p_buffer, frame_size)) {
    printf("read file failed\n");
    return -1;
  }
  int index = -1;
  int ret = CVI_AI_SoundClassification_V2(ai_handle, &Frame, &index);
  if (ret != 0) {
    printf("sound classification failed\n");
    return -1;
  }
  return index;
}

int main(int argc, char *argv[]) {
  int sample_rate = atoi(argv[5]);
  int frame_size = sample_rate * AUDIOFORMATSIZE * SECOND;
  CVI_U8 buffer[frame_size];  // 3 seconds

  cviai_handle_t ai_handle = NULL;
  CVI_S32 ret = CVI_AI_CreateHandle(&ai_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create ai handle failed with %#x!\n", ret);
    return ret;
  }
  std::string modelf(argv[1]);

  ret = CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_SOUNDCLASSIFICATION_V2, modelf.c_str());
  if (ret != CVI_SUCCESS) {
    printf("open modelfile failed %#x!\n", ret);
    return ret;
  }
  std::cout << "model opened:" << modelf << std::endl;
  if (argc == 6) {
    std::string str_root_dir = argv[2];
    std::string str_list_file = argv[3];
    std::string str_res_file = argv[4];
    std::vector<std::string> strfiles = read_file_lines(str_list_file);
    FILE *fp = fopen(str_res_file.c_str(), "w");
    int num_correct = 0;
    size_t num_total = strfiles.size();
    for (size_t i = 0; i < num_total; i++) {
      std::cout << "process:" << i << "/" << num_total << ",file:" << strfiles[i] << "\t";
      std::string strf = str_root_dir + std::string("/") + strfiles[i];
      int cls = test_binary_short_audio_data(strf, buffer, ai_handle, sample_rate);
      std::string str_res =
          strfiles[i] + std::string(",") + std::to_string(cls) + std::string("\t");
      fwrite(str_res.c_str(), str_res.length(), 1, fp);
      if (cls == -1) continue;
      std::string strlabel = std::string("/") + std::to_string(cls) + std::string("/");
      if (strf.find(strlabel) != strf.npos) {
        num_correct++;
      }
      std::cout << str_res << "correct num:" << num_correct << std::endl;
    }
    fclose(fp);
    std::cout << "total:" << strfiles.size() << ", correct:" << num_correct
              << ", false:" << strfiles.size() - num_correct << std::endl;
  } else {
    int cls = test_binary_short_audio_data(argv[2], buffer, ai_handle, sample_rate);
    std::cout << "result:" << cls << std::endl;
  }

  CVI_AI_DestroyHandle(ai_handle);

  return ret;
}

/*
how to use:

./test_audio_cls \                                                                         // cpp
name /mnt/data/nfsuser_xq/models/sound_cls/c10_lightv2_mel40_mix_sr8k_cv181x.cvimodel \         //
model path /mnt/data/nfsuser_xq/models/sound_cls/audio_data_new_sr8k_eval_bin/0 \ // file dir
/mnt/data/nfsuser_xq/models/sound_cls/audio_data_new_sr8k_eval_bin/0/files_list.txt \      // file
list.txt /mnt/data/nfsuser_xq/models/sound_cls/audio_data_new_sr8k_eval_bin/new_8k_0_res.txt \ //
save path 8000 // sample_rate [8000 | 16000]
*/