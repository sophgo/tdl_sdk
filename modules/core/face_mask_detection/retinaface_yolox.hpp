#pragma once
#include "core.hpp"
#include "core/face/cvai_face_types.h"

#include "opencv2/opencv.hpp"

namespace cviai {

class RetinafaceYolox final : public Core {
 public:
  RetinafaceYolox();
  virtual ~RetinafaceYolox();
  int inference(VIDEO_FRAME_INFO_S *srcFrame, cvai_face_t *face_meta);
  virtual bool allowExportChannelAttribute() const override { return true; }

 private:
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
  void outputParser(const int image_width, const int image_height, const int frame_width,
                    const int frame_height, cvai_face_t *face_meta);
};
}  // namespace cviai