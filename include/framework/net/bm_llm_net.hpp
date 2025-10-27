#ifndef TDL_SDK_NET_BM_LLM_NET_HPP
#define TDL_SDK_NET_BM_LLM_NET_HPP

#include <random>
#include "bmruntime_interface.h"

#include "common/common_types.hpp"
class BMLLMNet {
 public:
  BMLLMNet();
  int32_t init(const std::vector<int> &devices, const std::string &model_path);
  int32_t setInferParam(const LLMInferParam &infer_param);
  LLMInferParam getInferParam() { return infer_param_; }
  int getMaxSeqLen() { return SEQLEN_; }

  // TODO(fuquan.ke):move this two interfaces to llm_model.hpp
  int forwardFirst(const std::vector<int> &tokens);
  int forwardNext();

 private:
  void deinit();
  void netLaunch(const bm_net_info_t *net, int stage_idx = 0);
  void netLaunchDyn(const bm_net_info_t *net, int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset,
                  int size);

  void headLaunch(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int greedySearch(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int penaltySample(const bm_net_info_t *net, bm_device_mem_t &logits_mem);

 public:
  // 生成参数 - 这些需要保留为公共成员，因为它们可能需要从外部设置
  LLMInferParam infer_param_;

 private:
  // 保留必要的状态变量
  std::mt19937 sgen_;
  int token_length_;  // 当前token长度，需要在多次调用之间保持
  int SEQLEN_;        // 从bmodel读取的序列长度
  int NUM_LAYERS_;    // 从bmodel读取的层数
  bool io_alone_;     // IO模式标志
  bool is_dynamic_;   // 动态shape标志
  std::vector<int> visited_tokens_;  // 已访问的token历史

  // 运行时环境
  std::vector<bm_handle_t> handles_;
  bm_handle_t bm_handle_;
  void *p_bmrt_;

  // 网络信息
  std::vector<const bm_net_info_t *> net_blocks_;
  std::vector<const bm_net_info_t *> net_blocks_cache_;
  const bm_net_info_t *net_embed_;
  const bm_net_info_t *net_embed_cache_;
  const bm_net_info_t *net_lm_;
  const bm_net_info_t *net_greedy_head_;
  const bm_net_info_t *net_penalty_sample_head_;

  // KV缓存
  std::vector<bm_device_mem_t> past_key_;
  std::vector<bm_device_mem_t> past_value_;

  // 缓存大小信息 - 可以在init中计算并保存
  int hidden_bytes_;
  int kv_bytes_;

  // 当前处理的token长度 - 可以在每次forward时设置
  int TOKEN_LEN_;
};

#endif
