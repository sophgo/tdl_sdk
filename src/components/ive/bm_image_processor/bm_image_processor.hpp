#ifndef __BM_IMAGE_PROCESSOR_HPP__
#define __BM_IMAGE_PROCESSOR_HPP__

#include <memory>
#include "cvi_tpu.h"
#include "image/base_image.hpp"
#include "ive/image_processor.hpp"

class BmImageProcessor : public ImageProcessor {
 public:
  BmImageProcessor();
  virtual ~BmImageProcessor();

  virtual int32_t subads(std::shared_ptr<BaseImage> &src1,
                         std::shared_ptr<BaseImage> &src2,
                         std::shared_ptr<BaseImage> &dst) override;

  virtual int32_t thresholdProcess(std::shared_ptr<BaseImage> &input,
                                   std::shared_ptr<BaseImage> &output,
                                   CVI_U32 threshold_type, CVI_U32 threshold,
                                   CVI_U32 max_value) override;

  virtual int32_t twoWayBlending(std::shared_ptr<BaseImage> &left,
                                 std::shared_ptr<BaseImage> &right,
                                 std::shared_ptr<BaseImage> &output,
                                 CVI_S32 overlay_lx, CVI_S32 overlay_rx,
                                 CVI_U8 *wgt) override;
  int32_t compareResult(CVI_U8 *tpu_result, CVI_U8 *cpu_result, CVI_S32 size);

 private:
  bm_handle_t handle_;
};

#endif  // __BM_IMAGE_PROCESSOR_HPP__
