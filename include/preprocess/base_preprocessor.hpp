#ifndef BASE_PREPROCESSOR_H
#define BASE_PREPROCESSOR_H

#include <memory>
#include <string>
#include <vector>

#include "cvi_comm.h"
#include "image/base_image.hpp"
#include "memory/base_memory_pool.hpp"
class BasePreprocessor {
 public:
  virtual ~BasePreprocessor() = default;

  // 图像尺寸调整
  virtual std::shared_ptr<BaseImage> resize(const std::shared_ptr<BaseImage>& image, int newWidth,
                                            int newHeight) = 0;

  // 裁剪图像
  virtual std::shared_ptr<BaseImage> crop(const std::shared_ptr<BaseImage>& image, int x, int y,
                                          int width, int height) = 0;

  // 新增的统一预处理接口
  // 根据给定的 PreprocessParams，将图像依次进行
  // resize、crop、format转换、normalize等操作
  virtual std::shared_ptr<BaseImage> preprocess(const std::shared_ptr<BaseImage>& src_image,
                                                const PreprocessParams& params,
                                                std::shared_ptr<BaseMemoryPool> memory_pool) = 0;
};

#endif  // BASE_PREPROCESSOR_H