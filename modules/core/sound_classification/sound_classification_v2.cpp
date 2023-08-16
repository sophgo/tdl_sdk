#include "sound_classification_v2.hpp"
#include <iostream>
#include "cviai_log.hpp"
#include "cviruntime.h"
using namespace melspec;
using namespace cviai;
using namespace std;

SoundClassificationV2::SoundClassificationV2() : Core(CVI_MEM_SYSTEM) {
  int num_frames = time_len_ * sample_rate_;
  bool htk = false;
  mp_extractor_ = new MelFeatureExtract(num_frames, sample_rate_, num_fft_, hop_len_, num_mel_,
                                        fmin_, fmax_, "reflect", htk);
}

SoundClassificationV2::~SoundClassificationV2() { delete mp_extractor_; }

int SoundClassificationV2::onModelOpened() {
  CVI_SHAPE input_shape = getInputShape(0);
  std::cout << "input_shape = " << input_shape.dim[2] << std::endl;
  int32_t image_width = input_shape.dim[3];

  if (image_width == 188) {
    sample_rate_ = 16000;
  } else if (image_width == 94) {
    sample_rate_ = 8000;
  } else {
    return false;
  }
  return true;
}

int SoundClassificationV2::inference(VIDEO_FRAME_INFO_S *stOutFrame, int *index) {
  int img_width = stOutFrame->stVFrame.u32Width / 2;  // unit: 16 bits
  int img_height = stOutFrame->stVFrame.u32Height;
  // Mat *image = new Mat(img_height, img_width);

  // save audio to image array
  short *temp_buffer = reinterpret_cast<short *>(stOutFrame->stVFrame.pu8VirAddr[0]);
  mp_extractor_->update_data(temp_buffer, img_width * img_height);

  model_timer_.TicToc("start");
  // mp_extractor_->update_data(temp_buffer,img_width*img_height);

  // std::cout<<"update data done\n";
  const TensorInfo &tinfo = getInputTensorInfo(0);
  int8_t *input_ptr = tinfo.get<int8_t>();
  // mp_extractor_->melspectrogram_impl(input_ptr, int(tinfo.tensor_elem), tinfo.qscale);
  // int8_t *optimize_ptr = new int8_t[int(tinfo.tensor_elem)];
  mp_extractor_->melspectrogram_optimze(temp_buffer, img_width * img_height, input_ptr,
                                        int(tinfo.tensor_elem), tinfo.qscale);
  // int iseq = 1;
  // for(int i = 0; i < int(tinfo.tensor_elem);i++){
  //   int diff = input_ptr[i] - optimize_ptr[i];
  //   if(diff != 0){
  //     std::cout<<"not
  //     equal:"<<i<<",src:"<<int(input_ptr[i])<<",new:"<<int(optimize_ptr[i])<<std::endl; break;
  //   }
  // }

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
  model_timer_.TicToc("post");
  // std::cout<<"output index:"<<*index<<std::endl;
  return CVI_SUCCESS;
}
int SoundClassificationV2::get_top_k(float *result, size_t count) {
  // int TOP_K = 1;
  float *data = result;
  float conf_fg = 1.0 / (1 + std::exp(-result[1]));
  if (conf_fg > m_model_threshold) {
    return 1;
  } else {
    return 0;
  }
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
