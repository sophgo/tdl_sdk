#ifndef INCLUDE_OPENCV_PREPROCESSOR_H_
#define INCLUDE_OPENCV_PREPROCESSOR_H_

#include "preprocess/base_preprocessor.hpp"

class OpenCVPreprocessor : public BasePreprocessor {
 public:
  OpenCVPreprocessor();
  virtual ~OpenCVPreprocessor() override = default;

  virtual std::shared_ptr<BaseImage> resize(
      const std::shared_ptr<BaseImage>& image, int newWidth,
      int newHeight) override;

  // 裁剪图像
  virtual std::shared_ptr<BaseImage> crop(
      const std::shared_ptr<BaseImage>& image, int x, int y, int width,
      int height) override;

  virtual std::shared_ptr<BaseImage> preprocess(
      const std::shared_ptr<BaseImage>& src_image,
      const PreprocessParams& params,
      std::shared_ptr<BaseMemoryPool> memory_pool) override;
  virtual int32_t preprocessToImage(
      const std::shared_ptr<BaseImage>& src_image,
      const PreprocessParams& params,
      std::shared_ptr<BaseImage> dst_image) override;
  virtual int32_t preprocessToTensor(
      const std::shared_ptr<BaseImage>& src_image,
      const PreprocessParams& params, const int batch_idx,
      std::shared_ptr<BaseTensor> tensor) override;
  virtual std::vector<float> getRescaleConfig(const PreprocessParams& params,
                                              const int image_width,
                                              const int image_height) override;
};

#endif  // INCLUDE_OPENCV_PREPROCESSOR_H_