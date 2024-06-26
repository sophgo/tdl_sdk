#pragma once
#include "core/face/cvtdl_face_types.h"
//  @attention 人脸检测的预处理参数和算法参数是否公用
#include "core/object/cvtdl_object_types.h"
#include "core_internel.hpp"

namespace cvitdl {

class FaceDetectionBase : public Core {
 public:
  FaceDetectionBase() : Core(CVI_MEM_DEVICE){};
  virtual ~FaceDetectionBase(){};
  virtual int inference(VIDEO_FRAME_INFO_S *stOutFrame, cvtdl_face_t *meta) {
    LOGE("inference function not implement!\n");
    return 0;
  }

 private:
  cvtdl_pre_param_t preprocess_param_;
  cvtdl_det_algo_param_t alg_param_;
};
}  // namespace cvitdl