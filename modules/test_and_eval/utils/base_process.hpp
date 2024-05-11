#pragma once
#include "cvi_tdl.h"

class BaseProcessor {
 public:
  BaseProcessor();
  virtual ~BaseProcessor();
}

protected : cvitdl_handle_t tdl_handle;
imgprocess_t img_handle;
}
;
