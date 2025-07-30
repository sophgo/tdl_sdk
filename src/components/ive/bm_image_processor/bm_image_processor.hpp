#ifndef __BM_IMAGE_PROCESSOR_HPP__
#define __BM_IMAGE_PROCESSOR_HPP__

#include <memory>
#include "cvi_tpu.hpp"
#include "image/base_image.hpp"
#include "ive/image_processor.hpp"

class BmImageProcessor : public ImageProcessor {
 public:
  BmImageProcessor(const std::string &tpu_kernel_module_path);
  virtual ~BmImageProcessor();

  virtual int32_t subads(std::shared_ptr<BaseImage> &src1,
                         std::shared_ptr<BaseImage> &src2,
                         std::shared_ptr<BaseImage> &dst) override;

  virtual int32_t thresholdProcess(std::shared_ptr<BaseImage> &input,
                                   CVI_U32 threshold_type, CVI_U32 threshold,
                                   CVI_U32 max_value,
                                   std::shared_ptr<BaseImage> &output) override;

  virtual int32_t twoWayBlending(std::shared_ptr<BaseImage> &left,
                                 std::shared_ptr<BaseImage> &right,
                                 std::shared_ptr<BaseImage> &wgt,
                                 std::shared_ptr<BaseImage> &output) override;
  virtual int32_t erode(std::shared_ptr<BaseImage> &input, CVI_U32 kernal_w,
                        CVI_U32 kernal_h,
                        std::shared_ptr<BaseImage> &output) override;
  virtual int32_t dilate(std::shared_ptr<BaseImage> &input, CVI_U32 kernal_w,
                         CVI_U32 kernal_h,
                         std::shared_ptr<BaseImage> &output) override;
  int32_t morphProcess(std::shared_ptr<BaseImage> &input,
                       std::string morph_type, CVI_U32 kernal_w,
                       CVI_U32 kernal_h, std::shared_ptr<BaseImage> &output);
  int32_t compareResult(CVI_U8 *tpu_result, CVI_U8 *cpu_result, CVI_S32 size);

 private:
  bm_handle_t handle_;
  tpu_kernel_module_t tpu_module_;
  tpu_kernel_function_t func_id_subads_;
  tpu_kernel_function_t func_id_threshold_;
  tpu_kernel_function_t func_id_blend_2way_;
  tpu_kernel_function_t func_id_morph_;
};

#endif  // __BM_IMAGE_PROCESSOR_HPP__
