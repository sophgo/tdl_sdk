#include "base_process.hpp"
#include <functional>
#include <string>

BaseProcessor::BaseProcessor() {
  int ret = CVI_TDL_CreateHandle(&tdl_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create tdl handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_TDL_Create_ImageProcessor(&img_handle);
  if (ret != CVI_SUCCESS) {
    printf("CVI_TDL_Create_ImageProcessor failed with %d!\n", ret);
    return ret;
  }
}

BaseProcessor::~BaseProcessor() {
  if (img_handle) {
    CVI_TDL_Destroy_ImageProcessor(img_handle);
  }
  if (tdl_handle) {
    CVI_TDL_DestroyHandle(tdl_handle);
  }
}
