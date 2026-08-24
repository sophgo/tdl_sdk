#include <cstring>
#include <string>
#include <vector>
#include "tdl_model_factory.hpp"
#include "utils/common_utils.hpp"

// 离线流式声音分类:加载一次模型,对一整段裸 PCM 按固定 2s 窗口滑动逐窗推理,
// 等价于麦克风 sample(sample_microphone_command.c) 的"实时滑窗检测",
// 但数据来源是文件而非麦克风。
//
// 声音分类模型(CLS_SOUND_*) 输入张量固定为 ~2s,一次推理只能吃一个窗口,
// 因此禁止把整段 PCM 一次性喂进去(会导致 mel 帧数远超张量而越界)。
//
// 输入必须是裸 PCM(无 wav 头): 单声道 / 16-bit signed LE / 采样率与训练一致。
//   例如无人机模型: 8000 Hz, 2s -> 每窗 8000*2 = 16000 采样 = 32000 字节。

#define AUDIOFORMATSIZE 2  // 16-bit = 2 bytes/sample

static void print_usage(const char *prog) {
  printf(
      "Usage: %s <model_id_name> <model_dir> <pcm_path> [sample_rate] "
      "[win_sec] [step_sec]\n",
      prog);
  printf("  model_id_name : e.g. CLS_SOUND_DRONE / CLS_SOUND_BABAY_CRY\n");
  printf("  model_dir     : dir containing the .cvimodel\n");
  printf("  pcm_path      : raw PCM, mono/16-bit LE, no wav header\n");
  printf("  sample_rate   : default 8000\n");
  printf("  win_sec       : window length in seconds, default 2\n");
  printf(
      "  step_sec      : hop between windows in seconds, default = win_sec "
      "(non-overlap)\n");
  printf("\nExample (drone, 8k/2s, non-overlap):\n");
  printf("  %s CLS_SOUND_DRONE ./models ./drone_8k.pcm 8000 2\n", prog);
  printf("Example (50%% overlap):\n");
  printf("  %s CLS_SOUND_DRONE ./models ./drone_8k.pcm 8000 2 1\n", prog);
}

int main(int argc, char **argv) {
  if (argc < 4) {
    print_usage(argv[0]);
    return -1;
  }

  std::string model_id_name = argv[1];
  std::string model_dir = argv[2];
  std::string pcm_path = argv[3];
  int sample_rate = (argc > 4) ? atoi(argv[4]) : 8000;
  float win_sec = (argc > 5) ? atof(argv[5]) : 2.0f;
  float step_sec = (argc > 6) ? atof(argv[6]) : win_sec;  // 默认不重叠

  if (sample_rate <= 0 || win_sec <= 0 || step_sec <= 0) {
    printf("invalid sample_rate/win_sec/step_sec\n");
    return -1;
  }

  // 读入整段裸 PCM
  std::vector<uint8_t> buffer;
  if (!CommonUtils::readBinaryFile(pcm_path, buffer)) {
    printf("read file failed: %s\n", pcm_path.c_str());
    return -1;
  }

  const int win_bytes = (int)(sample_rate * win_sec) * AUDIOFORMATSIZE;
  const int step_bytes = (int)(sample_rate * step_sec) * AUDIOFORMATSIZE;
  const int total_bytes = (int)buffer.size();

  if (total_bytes < win_bytes) {
    printf(
        "input too short: %d bytes < one window %d bytes (%.2fs @ %dHz). "
        "need at least one full window.\n",
        total_bytes, win_bytes, win_sec, sample_rate);
    return -1;
  }

  printf(
      "audio: %d bytes, %.2f s @ %dHz | window %.2fs (%d bytes) step %.2fs\n",
      total_bytes, total_bytes / (float)(sample_rate * AUDIOFORMATSIZE),
      sample_rate, win_sec, win_bytes, step_sec);

  // 模型只加载一次
  TDLModelFactory &model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();
  model_factory.setModelDir(model_dir);

  std::shared_ptr<BaseModel> model_sound =
      model_factory.getModel(model_id_name);
  if (!model_sound) {
    printf("Failed to create model: %s\n", model_id_name.c_str());
    return -1;
  }

  int win_count = 0;
  int hit_count = 0;  // class_id != 0 (非 background) 的窗口数
  float hit_score_sum = 0.0f;

  // 滑动窗口逐窗推理,每窗即时打印
  for (int off = 0; off + win_bytes <= total_bytes; off += step_bytes) {
    // 每窗构造一个独立输入(尺寸恒为一个窗口,避免越界)
    std::vector<std::shared_ptr<BaseImage>> input_datas;
    std::shared_ptr<BaseImage> bin_data = ImageFactory::createImage(
        win_bytes, 1, ImageFormat::GRAY, TDLDataType::UINT8, true);
    uint8_t *data_buffer = bin_data->getVirtualAddress()[0];
    memcpy(data_buffer, buffer.data() + off, win_bytes);
    input_datas.push_back(bin_data);

    std::vector<std::shared_ptr<ModelOutputInfo>> output_info;
    model_sound->inference(input_datas, output_info);

    float t_start = off / (float)(sample_rate * AUDIOFORMATSIZE);
    for (size_t i = 0; i < output_info.size(); i++) {
      std::shared_ptr<ModelClassificationInfo> out =
          std::static_pointer_cast<ModelClassificationInfo>(output_info[i]);
      int class_id = out->topk_class_ids[0];
      float score = out->topk_scores[0];
      printf("[win %3d | %6.2fs] class_id: %d  score: %.4f\n", win_count,
             t_start, class_id, score);
      if (class_id != 0) {
        hit_count++;
        hit_score_sum += score;
      }
    }
    win_count++;
  }

  printf(
      "\n==== summary ====\n"
      "windows: %d | hits(class!=0): %d (%.1f%%) | avg hit score: %.4f\n",
      win_count, hit_count, win_count ? 100.0f * hit_count / win_count : 0.0f,
      hit_count ? hit_score_sum / hit_count : 0.0f);

  return 0;
}
