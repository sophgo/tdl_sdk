#pragma once
#include <bitset>
#include "core.hpp"
#include "core/object/cvai_object_types.h"
#include "object_utils.hpp"

namespace cviai {

class YoloV8Pose final : public Core {
 public:
  YoloV8Pose();

  virtual ~YoloV8Pose();
  int inference(VIDEO_FRAME_INFO_S *srcFrame, cvai_object_t *obj_meta);

 private:
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;

  void outputParser(const int num_boxes, const int feature_length, const int frame_width,
                    const int frame_height, cvai_object_t *obj_meta);

  void postProcess(Detections &dets, int frame_width, int frame_height, cvai_object_t *obj,
                   std::vector<int> &valild_ids, float *data);
};
}  // namespace cviai
