#include "net/bm_llm_net.hpp"
#include <assert.h>
#include "utils/tdl_log.hpp"

#define SOC_TARGET 1
static const uint16_t ATTENTION_MASK = 0xC61C;
void empty(bm_handle_t &bm_handle, bm_device_mem_t &mem) {
  int value = 0;
  auto ret = bm_memset_device_ext(bm_handle, &value, 1, mem);
  assert(BM_SUCCESS == ret);
}

BMLLMNet::BMLLMNet() {
  sgen_ = std::mt19937(std::random_device()());
  infer_param_.max_new_tokens = 2048;
  infer_param_.top_p = 1.0f;
  infer_param_.temperature = 1.0f;
  infer_param_.repetition_penalty = 1.0f;
  infer_param_.repetition_last_n = 64;
  infer_param_.generation_mode = "greedy";
  infer_param_.prompt_mode = "prompted";
}

void BMLLMNet::d2d(bm_device_mem_t &dst, bm_device_mem_t &src) {
  bm_memcpy_d2d_byte(bm_handle_, dst, 0, src, 0, bm_mem_get_device_size(src));
}

void BMLLMNet::d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset) {
  bm_memcpy_d2d_byte(bm_handle_, dst, offset, src, 0,
                     bm_mem_get_device_size(src));
}

void BMLLMNet::d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset,
                   int size) {
  bm_memcpy_d2d_byte(bm_handle_, dst, offset, src, 0, size);
}

int32_t BMLLMNet::init(const std::vector<int> &devices,
                       const std::string &model_path) {
  // request bm_handle
  LOGI("Device [ ");
  for (auto d : devices) {
    LOGI("%d ", d);
  }
  LOGI("] loading ....\n");
  for (auto d : devices) {
    bm_handle_t h;
    bm_status_t status = bm_dev_request(&h, d);
    assert(BM_SUCCESS == status);
    handles_.push_back(h);
  }
  bm_handle_ = handles_[0];

  // create bmruntime
#ifdef SOC_TARGET
  p_bmrt_ = bmrt_create(handles_[0]);
#else
  p_bmrt_ = bmrt_create_ex(handles_.data(), handles_.size());
#endif
  assert(NULL != p_bmrt_);

  // load bmodel by file
  LOGI("Model[%s] loading ....\n", model_path.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt_, model_path.c_str());
  assert(true == ret);
  LOGI("Done!\n");

  // net embed and lm_head
  net_embed_ = bmrt_get_network_info(p_bmrt_, "embedding");
  net_embed_cache_ = bmrt_get_network_info(p_bmrt_, "embedding_cache");
  net_lm_ = bmrt_get_network_info(p_bmrt_, "lm_head");
  net_greedy_head_ = bmrt_get_network_info(p_bmrt_, "greedy_head");
  net_penalty_sample_head_ =
      bmrt_get_network_info(p_bmrt_, "penalty_sample_head");
  SEQLEN_ = net_embed_->stages[0].input_shapes[0].dims[1];  // real seqlen
  auto num_nets = bmrt_get_network_number(p_bmrt_);
  NUM_LAYERS_ = (num_nets - 5) / 2;

  // resize
  visited_tokens_.resize(SEQLEN_);

  // net blocks
  for (int i = 0; i < NUM_LAYERS_; i++) {
    auto block_name = "block_" + std::to_string(i);
    auto cache_name = "block_cache_" + std::to_string(i);
    net_blocks_.emplace_back(
        bmrt_get_network_info(p_bmrt_, block_name.c_str()));
    net_blocks_cache_.emplace_back(
        bmrt_get_network_info(p_bmrt_, cache_name.c_str()));
  }
  // output hidden_states [1, 1, hidden_size],bf16
  hidden_bytes_ =
      bm_mem_get_device_size(net_blocks_cache_[0]->stages[0].output_mems[0]);
  // key cache [1, 1, head_num, head_size],bf16
  kv_bytes_ =
      bm_mem_get_device_size(net_blocks_cache_[0]->stages[0].output_mems[1]);

  // kv cache
  past_key_.resize(NUM_LAYERS_);
  past_value_.resize(NUM_LAYERS_);
  is_dynamic_ = net_blocks_[0]->is_dynamic;
  auto addr_mode = net_blocks_cache_[0]->addr_mode;
  io_alone_ = addr_mode == 1;
  for (int i = 0; i < NUM_LAYERS_; i++) {
    assert(addr_mode == net_blocks_cache_[i]->addr_mode);
    if (io_alone_) {
      past_key_[i] = net_blocks_cache_[i]
                         ->stages[0]
                         .input_mems[3];  // history key cache
                                          // [1,seq_len,head_num,head_size],bf16
      past_value_[i] =
          net_blocks_cache_[i]->stages[0].input_mems
              [4];  // history value cache [1,seq_len,head_num,head_size],bf16
    } else {
      auto ret = bm_malloc_device_byte(
          bm_handle_, &past_key_[i], net_blocks_cache_[i]->max_input_bytes[3]);
      assert(BM_SUCCESS == ret);
      ret = bm_malloc_device_byte(bm_handle_, &past_value_[i],
                                  net_blocks_cache_[i]->max_input_bytes[4]);
      assert(BM_SUCCESS == ret);
    }
  }
  return 0;
}

void BMLLMNet::deinit() {
  if (false == io_alone_) {
    for (int i = 0; i < NUM_LAYERS_; i++) {
      bm_free_device(bm_handle_, past_key_[i]);
      bm_free_device(bm_handle_, past_value_[i]);
    }
  }
  bmrt_destroy(p_bmrt_);
  for (auto h : handles_) {
    bm_dev_free(h);
  }
}

void BMLLMNet::netLaunch(const bm_net_info_t *net, int stage_idx) {
  std::vector<bm_tensor_t> in_tensors(net->input_num);
  std::vector<bm_tensor_t> out_tensors(net->output_num);

  for (int i = 0; i < net->input_num; i++) {
    bmrt_tensor_with_device(
        &in_tensors[i], net->stages[stage_idx].input_mems[i],
        net->input_dtypes[i], net->stages[stage_idx].input_shapes[i]);
  }
  for (int i = 0; i < net->output_num; i++) {
    bmrt_tensor_with_device(
        &out_tensors[i], net->stages[stage_idx].output_mems[i],
        net->output_dtypes[i], net->stages[stage_idx].output_shapes[i]);
  }
  auto ret = bmrt_launch_tensor_ex(p_bmrt_, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
  bm_thread_sync(bm_handle_);
}

void BMLLMNet::netLaunchDyn(const bm_net_info_t *net, int stage_idx) {
  std::vector<bm_tensor_t> in_tensors(net->input_num);
  std::vector<bm_tensor_t> out_tensors(net->output_num);

  for (int i = 0; i < net->input_num; i++) {
    bmrt_tensor_with_device(
        &in_tensors[i], net->stages[stage_idx].input_mems[i],
        net->input_dtypes[i], net->stages[stage_idx].input_shapes[i]);
  }
  for (int i = 0; i < net->output_num; i++) {
    bmrt_tensor_with_device(
        &out_tensors[i], net->stages[stage_idx].output_mems[i],
        net->output_dtypes[i], net->stages[stage_idx].output_shapes[i]);
  }

  int h_bytes = bm_mem_get_device_size(in_tensors[0].device_mem) / SEQLEN_;
  bm_set_device_mem(&in_tensors[0].device_mem, h_bytes * TOKEN_LEN_,
                    bm_mem_get_device_addr(in_tensors[0].device_mem));
  int pid_bytes = bm_mem_get_device_size(in_tensors[1].device_mem) / SEQLEN_;
  bm_set_device_mem(&in_tensors[1].device_mem, pid_bytes * TOKEN_LEN_,
                    bm_mem_get_device_addr(in_tensors[1].device_mem));
  int mask_bytes =
      bm_mem_get_device_size(in_tensors[2].device_mem) / SEQLEN_ / SEQLEN_;
  bm_set_device_mem(&in_tensors[2].device_mem,
                    mask_bytes * TOKEN_LEN_ * TOKEN_LEN_,
                    bm_mem_get_device_addr(in_tensors[2].device_mem));

  in_tensors[0].shape.dims[1] = TOKEN_LEN_;
  in_tensors[1].shape.dims[1] = TOKEN_LEN_;
  in_tensors[2].shape.dims[2] = TOKEN_LEN_;
  in_tensors[2].shape.dims[3] = TOKEN_LEN_;

  auto ret = bmrt_launch_tensor_ex(p_bmrt_, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
  bm_thread_sync(bm_handle_);
}

// logits_mem: [1, 1, hidden_size],bf16,come from
// net_blocks_[NUM_LAYERS_-1]->stages[0].output_mems[0] output: [1, 1,
// vocab_size],fp32
void BMLLMNet::headLaunch(const bm_net_info_t *net,
                          bm_device_mem_t &logits_mem) {
  std::vector<bm_tensor_t> in_tensors(net->input_num);
  std::vector<bm_tensor_t> out_tensors(net->output_num);

  bmrt_tensor_with_device(&in_tensors[0], logits_mem, net->input_dtypes[0],
                          net->stages[0].input_shapes[0]);

  for (int i = 1; i < net->input_num; i++) {
    bmrt_tensor_with_device(&in_tensors[i], net->stages[0].input_mems[i],
                            net->input_dtypes[i],
                            net->stages[0].input_shapes[i]);
  }
  for (int i = 0; i < net->output_num; i++) {
    bmrt_tensor_with_device(&out_tensors[i], net->stages[0].output_mems[i],
                            net->output_dtypes[i],
                            net->stages[0].output_shapes[i]);
  }
  auto ret = bmrt_launch_tensor_ex(p_bmrt_, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
  bm_thread_sync(bm_handle_);
}

int BMLLMNet::greedySearch(const bm_net_info_t *net,
                           bm_device_mem_t &logits_mem) {
  auto &out_mem = net->stages[0].output_mems[0];
  headLaunch(net, logits_mem);
  int token = 0;
  bm_memcpy_d2s(bm_handle_, (void *)&token, out_mem);
  return token;
}

int BMLLMNet::penaltySample(const bm_net_info_t *net,
                            bm_device_mem_t &logits_mem) {
  auto &in1_mem = net->stages[0].input_mems[1];    // input_ids
  auto &in2_mem = net->stages[0].input_mems[2];    // top_p
  auto &in3_mem = net->stages[0].input_mems[3];    // temperature
  auto &in4_mem = net->stages[0].input_mems[4];    // repetition_penalty
  auto &out0_mem = net->stages[0].output_mems[0];  // probs
  auto &out1_mem = net->stages[0].output_mems[1];  // topken_topk

  // repeat_penalty + top_p + top_k + temperature
  std::vector<int> generated_tokens(SEQLEN_,
                                    visited_tokens_[token_length_ - 1]);
  int repeat_last_n = std::min(infer_param_.repetition_last_n, token_length_);
  std::copy(visited_tokens_.begin() + token_length_ - repeat_last_n,
            visited_tokens_.begin() + token_length_, generated_tokens.begin());
  bm_memcpy_s2d(bm_handle_, in1_mem, (void *)generated_tokens.data());
  bm_memcpy_s2d(bm_handle_, in2_mem, (void *)&infer_param_.top_p);
  bm_memcpy_s2d(bm_handle_, in3_mem, (void *)&infer_param_.temperature);
  bm_memcpy_s2d(bm_handle_, in4_mem, (void *)&infer_param_.repetition_penalty);

  // inference
  headLaunch(net, logits_mem);

  // get logit & token
  int candidate_num = net->stages[0].output_shapes[0].dims[1];
  std::vector<float> probs(candidate_num);
  bm_memcpy_d2s(bm_handle_, probs.data(), out0_mem);
  std::vector<int> tokens(candidate_num);
  bm_memcpy_d2s(bm_handle_, tokens.data(), out1_mem);

  // penalty_sample
  std::discrete_distribution<> dist(probs.begin(), probs.end());
  return tokens[dist(sgen_)];
}

int BMLLMNet::forwardFirst(const std::vector<int> &tokens) {
  std::vector<int> position_id(SEQLEN_, 0);
  std::vector<uint16_t> attention_mask(SEQLEN_ * SEQLEN_, ATTENTION_MASK);
  std::fill(visited_tokens_.begin(), visited_tokens_.end(), 0);
  std::copy(tokens.begin(), tokens.end(), visited_tokens_.data());

  token_length_ = tokens.size();
  TOKEN_LEN_ = tokens.size();

  for (int i = 0; i < token_length_; i++) {
    position_id[i] = i;
  }
  if (is_dynamic_) {
    for (int i = 0; i < token_length_; i++) {
      for (int j = 0; j < TOKEN_LEN_; j++) {
        if (j <= i) {
          attention_mask[i * TOKEN_LEN_ + j] = 0;
        }
      }
    }
  } else {
    for (int i = 0; i < token_length_; i++) {
      for (int j = 0; j < SEQLEN_; j++) {
        if (j <= i) {
          attention_mask[i * SEQLEN_ + j] = 0;
        }
      }
    }
  }
  // empty
  for (int i = 0; i < NUM_LAYERS_; i++) {
    // hidden_states  [1, seq_len, hidden_size],bf16
    empty(bm_handle_, net_blocks_[i]->stages[0].input_mems[0]);
    // cached hidden states [1, 1, hidden_size],bf16
    empty(bm_handle_, net_blocks_cache_[i]->stages[0].input_mems[0]);
  }

  // forward embeding
  auto &in_mem = net_embed_->stages[0].input_mems[0];
  // output embedding [1, seq_len, hidden_size],bf16
  auto &out_mem = net_embed_->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle_, in_mem, (void *)visited_tokens_.data());
  netLaunch(net_embed_);  // prefil embedding

  // forward blocks
  for (int idx = 0; idx < NUM_LAYERS_; idx++) {
    // hidden_states  [1, seq_len, 3584]
    auto &in0_mem = net_blocks_[idx]->stages[0].input_mems[0];
    // position_ids [1, seq_len]
    auto &in1_mem = net_blocks_[idx]->stages[0].input_mems[1];
    // attention_mask [1, seq_len, seq_len]
    auto &in2_mem = net_blocks_[idx]->stages[0].input_mems[2];
    empty(bm_handle_, net_blocks_[idx]->stages[0].input_mems[0]);
    d2d(in0_mem, out_mem, 0, token_length_ * hidden_bytes_);
    if (idx == 0) {
      // only first time need copy
      bm_memcpy_s2d(bm_handle_, in1_mem, (void *)position_id.data());
      bm_memcpy_s2d(bm_handle_, in2_mem, (void *)attention_mask.data());
    }
    if (is_dynamic_)
      netLaunchDyn(net_blocks_[idx]);
    else
      netLaunch(net_blocks_[idx]);
    // update output hidden_states [1, seq_len, 3584]
    out_mem = net_blocks_[idx]->stages[0].output_mems[0];
    // update history key cache [1,seq_len,head_num,head_size],bf16
    d2d(past_key_[idx], net_blocks_[idx]->stages[0].output_mems[1], 0,
        token_length_ * kv_bytes_);
    // update history value cache [1,seq_len,head_num,head_size],bf16
    d2d(past_value_[idx], net_blocks_[idx]->stages[0].output_mems[2], 0,
        token_length_ * kv_bytes_);
  }

  // forward lmhead
  auto &lm_in_mem = net_lm_->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm_->stages[0].output_mems[0];
  bm_memcpy_d2d_byte(bm_handle_, lm_in_mem, 0, out_mem,
                     (token_length_ - 1) * hidden_bytes_, hidden_bytes_);
  netLaunch(net_lm_);

  int token = 0;
  if (infer_param_.generation_mode == "greedy") {
    token = greedySearch(net_greedy_head_, lm_out_mem);
  } else if (infer_param_.generation_mode == "penalty_sample") {
    token = penaltySample(net_penalty_sample_head_, lm_out_mem);
  }

  visited_tokens_[token_length_] = token;
  token_length_ += 1;
  return token;
}

int BMLLMNet::forwardNext() {
  int cur_token = visited_tokens_[token_length_ - 1];

  std::vector<uint16_t> attention_mask(SEQLEN_ + 1, 0);
  for (int i = token_length_ - 1; i < SEQLEN_; i++) {
    attention_mask[i] = ATTENTION_MASK;
  }
  int32_t position_id = token_length_ - 1;
  // input token [1, 1]
  auto &in_mem = net_embed_cache_->stages[0].input_mems[0];
  // output embedding [1, 1, hidden_size],bf16
  auto &out_mem = net_embed_cache_->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle_, in_mem, (void *)&cur_token);
  netLaunch(net_embed_cache_);

  // blocks
  int token_offset = (token_length_ - 1) * kv_bytes_;
  for (int idx = 0; idx < NUM_LAYERS_; idx++) {
    // cached hidden states [1, 1, hidden_size],bf16
    auto &in0_mem = net_blocks_cache_[idx]->stages[0].input_mems[0];
    // position_ids [1, 1]
    auto &in1_mem = net_blocks_cache_[idx]->stages[0].input_mems[1];
    // attention_mask [1, 1,1, seq_len+1]
    auto &in2_mem = net_blocks_cache_[idx]->stages[0].input_mems[2];
    // history key cache [1,seq_len,head_num,head_size],bf16
    auto &in3_mem = net_blocks_cache_[idx]->stages[0].input_mems[3];
    // history value cache [1,seq_len,head_num,head_size],bf16
    auto &in4_mem = net_blocks_cache_[idx]->stages[0].input_mems[4];
    // output hidden_states [1, 1, hidden_size],bf16
    auto &out0_mem = net_blocks_cache_[idx]->stages[0].output_mems[0];
    // output key cache [1,1,head_num,head_size],bf16
    auto &out1_mem = net_blocks_cache_[idx]->stages[0].output_mems[1];
    // output value cache [1,1,head_num,head_size],bf16
    auto &out2_mem = net_blocks_cache_[idx]->stages[0].output_mems[2];
    d2d(in0_mem, out_mem);
    if (io_alone_) {
      if (idx == 0) {
        // copy position_ids and attention_mask for first block
        bm_memcpy_s2d(bm_handle_, in1_mem, (void *)&position_id);
        bm_memcpy_s2d(bm_handle_, in2_mem, (void *)attention_mask.data());
      } else {
        // copy position_ids and attention_mask from the first block ,use d2d
        // instead of s2d
        d2d(in1_mem, net_blocks_cache_[0]->stages[0].input_mems[1]);
        d2d(in2_mem, net_blocks_cache_[0]->stages[0].input_mems[2]);
      }
    } else {
      if (idx == 0) {
        bm_memcpy_s2d(bm_handle_, in1_mem, (void *)&position_id);
        bm_memcpy_s2d(bm_handle_, in2_mem, (void *)attention_mask.data());
      }
      d2d(in3_mem, past_key_[idx]);
      d2d(in4_mem, past_value_[idx]);
    }
    netLaunch(net_blocks_cache_[idx]);
    // update output hidden_states [1, 1, hidden_size],bf16
    out_mem = out0_mem;
    // update output key cache [1,1,head_num,head_size],bf16
    bm_memcpy_d2d_byte(bm_handle_, past_key_[idx], token_offset, out1_mem, 0,
                       kv_bytes_);
    // update output value cache [1,1,head_num,head_size],bf16
    bm_memcpy_d2d_byte(bm_handle_, past_value_[idx], token_offset, out2_mem, 0,
                       kv_bytes_);
  }

  // forward lmhead
  auto &lm_in_mem = net_lm_->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm_->stages[0].output_mems[0];
  d2d(lm_in_mem, out_mem);
  netLaunch(net_lm_);

  int token = 0;
  if (infer_param_.generation_mode == "greedy") {
    token = greedySearch(net_greedy_head_, lm_out_mem);
  } else if (infer_param_.generation_mode == "penalty_sample") {
    token = penaltySample(net_penalty_sample_head_, lm_out_mem);
  }

  visited_tokens_[token_length_] = token;
  token_length_ += 1;
  return token;
}
