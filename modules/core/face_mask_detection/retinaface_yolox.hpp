#pragma once
#ifdef ATHENA2
#include "core_a2.hpp"
#else
#include "core.hpp"
#endif
#include "core/face/cvtdl_face_types.h"

#include "opencv2/opencv.hpp"

namespace cvitdl {

class RetinafaceYolox final : public Core {
 public:
  RetinafaceYolox();
  virtual ~RetinafaceYolox();
  int inference(VIDEO_FRAME_INFO_S *srcFrame, cvtdl_face_t *face_meta);
  virtual bool allowExportChannelAttribute() const override { return true; }

 private:
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
  void outputParser(const int image_width, const int image_height, const int frame_width,
                    const int frame_height, cvtdl_face_t *face_meta);
};
}  // namespace cvitdl