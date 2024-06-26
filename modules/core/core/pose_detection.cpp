#include "pose_detection.hpp"

namespace cvitdl {
int PoseDetectionBase::setupInputPreprocess(std::vector<InputPreprecessSetup> *data) {
  if (data->size() != 1) {
    LOGE("Detection model only has 1 input.\n");
    return CVI_TDL_ERR_INVALID_ARGS;
  }

  for (int i = 0; i < 3; i++) {
    (*data)[0].factor[i] = preprocess_param_.factor[i];
    (*data)[0].mean[i] = preprocess_param_.mean[i];
  }

  (*data)[0].format = preprocess_param_.format;
  (*data)[0].use_crop = use_crop_;
  (*data)[0].rescale_type = preprocess_param_.rescale_type;
  return CVI_TDL_SUCCESS;
}
}  // namespace cvitdl