#include "sound_classification_v2.hpp"
#include <iostream>
#include "cviai_log.hpp"
using namespace melspec;
using namespace cviai;

SoundClassificationV2::SoundClassificationV2() : Core(CVI_MEM_SYSTEM) {
  int num_frames = time_len_ * sample_rate_;
  bool htk = false;
  mp_extractor_ = new MelFeatureExtract(num_frames, sample_rate_, num_fft_, hop_len_, num_mel_,
                                        fmin_, fmax_, "reflect", htk);
  m_skip_preprocess_ = true;
}

SoundClassificationV2::~SoundClassificationV2() { delete mp_extractor_; }

int SoundClassificationV2::inference(VIDEO_FRAME_INFO_S *stOutFrame, int *index) {
  int img_width = stOutFrame->stVFrame.u32Width / 2;  // unit: 16 bits
  int img_height = stOutFrame->stVFrame.u32Height;
  // Mat *image = new Mat(img_height, img_width);

  // save audio to image array
  short *temp_buffer = reinterpret_cast<short *>(stOutFrame->stVFrame.pu8VirAddr[0]);
  mp_extractor_->update_data(temp_buffer, img_width * img_height);
  // std::cout<<"update data done\n";
  const TensorInfo &tinfo = getInputTensorInfo(0);
  int8_t *input_ptr = tinfo.get<int8_t>();
  mp_extractor_->melspectrogram_impl(input_ptr, int(tinfo.tensor_elem), tinfo.qscale);
  // FILE *fp = fopen("/mnt/data/admin1_data/alios_test/feat.bin","wb");
  // fwrite(input_ptr,tinfo.tensor_elem,1,fp);
  // fclose(fp);
  // std::cout<<"melspectrogram_impl data done\n";

  std::vector<VIDEO_FRAME_INFO_S *> frames = {stOutFrame};
  run(frames);
  // std::cout<<"run data done\n";
  const TensorInfo &info = getOutputTensorInfo(0);

  // get top k
  *index = get_top_k(info.get<float>(), info.tensor_elem);
  // std::cout<<"output index:"<<*index<<std::endl;
  return CVI_SUCCESS;
}
int SoundClassificationV2::get_top_k(float *result, size_t count) {
  // int TOP_K = 1;
  float *data = result;  // reinterpret_cast<float *>(malloc(count * sizeof(float)));
  // memcpy(data, result, count * sizeof(float));
  int idx = -1;
  float max = -10000;
  for (size_t i = 0; i < count; i++) {
    // std::cout<<"i:"<<i<<",val:"<<result[i]<<std::endl;
    if (result[i] > max) {
      max = data[i];
      idx = i;
    }
  }
  // if (max < threshold_) return count;  // pct lower than threshold, so return classes + 1
  return idx;
}
