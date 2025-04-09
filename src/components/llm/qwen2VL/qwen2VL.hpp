#ifndef TDL_SDK_QWEN2VL_HPP
#define TDL_SDK_QWEN2VL_HPP

#include <assert.h>
#include <getopt.h>
#include <inttypes.h>
#include <stdio.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include "../../../include/framework/net/bm_llm_net.hpp"
#include "bmruntime_interface.h"
#include "memory.h"

// 声明外部函数empty
extern void empty(bm_handle_t &bm_handle, bm_device_mem_t &mem);

// static const uint16_t ATTENTION_MASK = 0xC61C; // -9984 by bfloat16
static const float QWEN2VL_ATTENTION_MASK = -10000.;

class Qwen2VL {
 public:
  void init(int devid, std::string model_path);
  void deinit();
  int forward_first(std::vector<int> &tokens, std::vector<int> &position_id,
                    std::vector<float> &pixel_values, std::vector<int> &posids,
                    std::vector<float> &attnmask, int img_offset,
                    int pixel_num);
  int forward_next();

  std::mt19937 sgen;
  Qwen2VL() : sgen(std::random_device()()) {};

 private:
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);
  void head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int greedy_search(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int penalty_sample(const bm_net_info_t *net, bm_device_mem_t &logits_mem,
                     std::vector<int> &input_tokens, int &token_length);
  uint16_t mask_value;

 public:
  int token_length;
  int SEQLEN;  // read from bmodel
  int HIDDEN_SIZE;
  int NUM_LAYERS;  // read from bmodel
  uint64_t IMAGE_BYTES;
  std::vector<std::vector<int>> POSITION_IDS;
  int MAX_POS = 0;
  std::string generation_mode;
  int MAX_PIXELS;
  uint64_t VIT_DIMS;

 private:
  bm_handle_t bm_handle;
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_lm, *net_greedy_head, *net_penalty_sample_head;
  const bm_net_info_t *net_vit;
  bm_device_mem_t dev_buffer;
  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;
};

#endif  // TDL_SDK_QWEN2VL_HPP
