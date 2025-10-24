#ifndef ZIPFORMER_ENCODER_HPP
#define ZIPFORMER_ENCODER_HPP
#include "feature-fbank.h"
#include "model/base_model.hpp"
#include "online-feature.h"

class ZipformerEncoder : public BaseModel {
 public:
  ZipformerEncoder();
  virtual ~ZipformerEncoder();
  int32_t setupNetwork(NetParam& net_param) override;
  virtual int32_t inference(
      const std::shared_ptr<BaseImage>& image,
      std::shared_ptr<ModelOutputInfo>& out_data,
      const std::map<std::string, float>& parameters = {}) override;
  virtual int32_t outputParse(
      const std::shared_ptr<BaseImage>& image,
      std::shared_ptr<ModelOutputInfo>& out_data) override;
  virtual int32_t onModelOpened() override;
  int32_t setModel(std::shared_ptr<BaseModel> decoder_model,
                   std::shared_ptr<BaseModel> joiner_model);

  int32_t setTokensPath(std::string tokens_path);

 private:
  int32_t prepareInput();
  int32_t greedy_search(float* data_ptr,
                        std::shared_ptr<ModelOutputInfo>& out_data);

  knf::OnlineFbank* fbank_extractor_ = nullptr;
  float* float_chached_inputs_ = nullptr;
  int32_t* int32_chached_inputs_ = nullptr;
  std::vector<int> float_cached_offset_;
  std::vector<int> int32_cached_offset_;
  bool init_decoder_output_ = false;
  int32_t num_processed_frames_ = 0;
  int frame_offset_;
  int segment_size_;
  int num_mel_;
  int feature_num_;
  std::vector<int32_t> hyp_ = {0};
  std::vector<std::string> tokens_;

  std::shared_ptr<BaseModel> decoder_model_;
  std::shared_ptr<BaseModel> joiner_model_;

  std::shared_ptr<BaseImage> decoder_input_data_;
  std::shared_ptr<BaseImage> joiner_input_data_;
};

#endif
