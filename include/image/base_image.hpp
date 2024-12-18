#ifndef BaseImage_H
#define BaseImage_H

#include <memory>
#include <string>
#include <vector>

#include "common/common_types.hpp"
#include "memory/base_memory_pool.hpp"

class BaseImage {
 public:
  virtual ~BaseImage() = default;
  BaseImage(uint32_t width, uint32_t height, ImageFormat imageFormat,
            ImagePixDataType pix_data_type, std::unique_ptr<MemoryBlock> memory_block = nullptr);
  BaseImage() {}
  // 获取图像宽度
  virtual uint32_t getWidth() const = 0;

  // 获取图像高度
  virtual uint32_t getHeight() const = 0;

  // 获取图像通道数
  virtual int getChannels() const = 0;

  // 获取图像数据指针
  virtual void* getData() const = 0;

  // 物理地址个数与plane个数一致
  virtual std::vector<uint64_t> getPhysicalAddress() const = 0;

  // 虚拟地址个数与plane个数一致
  virtual std::vector<uint8_t*> getVirtualAddress() const = 0;

  virtual uint32_t getPlaneNum() const = 0;

  // 获取图像数据大小 (字节数)
  virtual uint32_t getImageByteSize() const = 0;

  // 获取设备类型 (如 CPU, GPU)
  virtual std::string getDeviceType() const = 0;

  // 获取硬件相关的元信息（如 GPU buffer handle 等）
  virtual void* getPlatformMetadata() const = 0;

  // 获取图像类型
  virtual ImageType getImageType() { return image_type_; }

  // 获取图像像素数据类型
  virtual ImagePixDataType getPixDataType() { return pix_data_type_; }

  ImageFormat getImageFormat() const { return image_format_; }

  virtual uint32_t getInternalType() = 0;
  std::unique_ptr<MemoryBlock> getMemoryBlock() { return std::move(memory_block_); }
  virtual int32_t invalidateCache() = 0;
  virtual int32_t flushCache() = 0;

  virtual int32_t randomFill();

  virtual int32_t readImage(const std::string& file_path) = 0;
  virtual int32_t writeImage(const std::string& file_path) = 0;

 protected:
  ImageType image_type_ = ImageType::UNKOWN;

  ImagePixDataType pix_data_type_ = ImagePixDataType::UINT8;

  std::unique_ptr<MemoryBlock> memory_block_;

  ImageFormat image_format_ = ImageFormat::UNKOWN;
  bool is_from_pool_ = false;
};

#endif  // BaseImage_H