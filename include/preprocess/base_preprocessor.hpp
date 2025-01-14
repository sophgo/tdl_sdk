#ifndef BASE_PREPROCESSOR_H
#define BASE_PREPROCESSOR_H

#include <memory>
#include <string>
#include <vector>

#include "cvi_comm.h"
#include "image/base_image.hpp"
#include "memory/base_memory_pool.hpp"
#include "net/base_tensor.hpp"
class BasePreprocessor {
 public:
  virtual ~BasePreprocessor() = default;

  // 图像尺寸调整
  virtual std::shared_ptr<BaseImage> resize(
      const std::shared_ptr<BaseImage>& image, int newWidth, int newHeight) = 0;

  // 裁剪图像
  virtual std::shared_ptr<BaseImage> crop(
      const std::shared_ptr<BaseImage>& image, int x, int y, int width,
      int height) = 0;

  virtual std::shared_ptr<BaseImage> preprocess(
      const std::shared_ptr<BaseImage>& src_image,
      const PreprocessParams& params,
      std::shared_ptr<BaseMemoryPool> memory_pool) = 0;
  virtual int32_t preprocessToImage(const std::shared_ptr<BaseImage>& src_image,
                                    const PreprocessParams& params,
                                    std::shared_ptr<BaseImage> dst_image) = 0;
  //[scalex,scaley,offsetx,offsety]
  // use as : new_x = (x - offsetx) * scalex
  //          new_y = (y - offsety) * scaley
  virtual std::vector<float> getRescaleConfig(const PreprocessParams& params,
                                              const int image_width,
                                              const int image_height) = 0;
  virtual int32_t preprocessToTensor(
      const std::shared_ptr<BaseImage>& src_image,
      const PreprocessParams& params, const int batch_idx,
      std::shared_ptr<BaseTensor> tensor) = 0;
};

#endif  // BASE_PREPROCESSOR_H