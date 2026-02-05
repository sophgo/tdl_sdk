#include <signal.h>
#include <iostream>
#include <string>
#include <vector>
#include "audio/audio_player.hpp"
#include "cvi_sys.h"

#define PERIOD_SIZE 80

bool gRun = true;

static void SampleHandleSig(int signo) {
  if (SIGINT == signo || SIGTERM == signo) {
    gRun = false;
  }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("Usage: %s <pcm_file_with_1_channel> <sample_rate>\n", argv[0]);
    return -1;
  }

  std::string sound_file = argv[1];
  int sample_rate = std::stoi(argv[2]);

  FILE *fpAo = fopen(sound_file.c_str(), "rb");
  if (!fpAo) {
    printf("fpAo open fail\n");
    return -1;
  }

  // Set signal catch
  signal(SIGINT, SampleHandleSig);
  signal(SIGTERM, SampleHandleSig);

  // Initialize Audio Player
  AudioPlayer player;
  AudioPlayer::Config config;
  config.sample_rate = sample_rate;
  config.period_size = PERIOD_SIZE;
  config.volume = 20;

  // Note: AudioPlayer uses AudioSystem which handles CVI_AUDIO_INIT safely
  if (player.Init(config) != 0) {
    printf("AudioPlayer Init failed\n");
    CVI_SYS_Exit();
    fclose(fpAo);
    return -1;
  }

  int frame_bytes = config.period_size * 2;
  std::vector<uint8_t> buffer(frame_bytes);

  printf("Start playing audio file: %s at %d Hz\n", sound_file.c_str(),
         sample_rate);

  while (gRun) {
    size_t read_bytes = fread(buffer.data(), 1, frame_bytes, fpAo);
    if (read_bytes > 0) {
      if (player.SendFrame(buffer.data(), read_bytes) != 0) {
        printf("SendFrame failed\n");
        break;
      }
    } else {
      // End of file
      break;
    }
  }

  printf("Playback finished.\n");

  player.Deinit();
  CVI_SYS_Exit();
  fclose(fpAo);

  return 0;
}
