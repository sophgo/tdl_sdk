#pragma once
#include "base_process.hpp"
#include "cvi_tdl.h"

class SocProcessor : public BaseProcessor {
 public:
  SocProcessor() = default;
  ;
  virtual ~SocProcessor() = default;
  ;
  void process_init(int vpssgrp_width, int vpssgrp_height, int BlkCount = 1, int BlkCount = 1);
};