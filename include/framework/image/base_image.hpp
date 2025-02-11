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
  // BaseImage(uint32_t width, uint32_t height, ImageFormat imageFormat,
  //           ImagePixDataType pix_data_type, bool alloc_memory = false,
  //           std::shared_ptr<BaseMemoryPool> memory_pool = nullptr);
  BaseImage();

  virtual int32_t prepareImageInfo(uint32_t width, uint32_t height,
                                   ImageFormat imageFormat,
                                   ImagePixDataType pix_data_type) = 0;
  virtual int32_t allocateMemory();
  virtual int32_t freeMemory();
  virtual bool isInitialized() const;
  virtual bool isAligned() const;

  virtual uint32_t getWidth() const = 0;
  virtual uint32_t getHeight() const = 0;
  virtual std::vector<uint32_t> getStrides() const = 0;
  virtual std::vector<uint64_t> getPhysicalAddress() const = 0;
  virtual std::vector<uint8_t*> getVirtualAddress() const = 0;
  virtual uint32_t getPlaneNum() const = 0;

  virtual uint32_t getImageByteSize() const { return 0; };
  virtual std::string getDeviceType() const { return ""; };
  virtual ImageFormat getImageFormat() const { return image_format_; }

  virtual ImageImplType getImageType() const { return image_type_; }
  virtual ImagePixDataType getPixDataType() const { return pix_data_type_; }
  virtual bool isPlanar() const;

  virtual uint32_t getInternalType() = 0;
  virtual void* getInternalData() const = 0;

  // 当引用模型输入作为预处理输出结果时，需要进行内存同步
  virtual int32_t invalidateCache();
  virtual int32_t flushCache();

  virtual int32_t randomFill();

  virtual int32_t setupMemoryBlock(std::unique_ptr<MemoryBlock>& memory_block);

  virtual int32_t setupMemory(uint64_t phy_addr, uint8_t* vir_addr,
                              uint32_t length) = 0;
  std::shared_ptr<BaseMemoryPool> getMemoryPool() { return memory_pool_; }

 private:
  MemoryPoolType getMemoryPoolType();

 protected:
  ImageImplType image_type_ = ImageImplType::UNKOWN;
  ImagePixDataType pix_data_type_ = ImagePixDataType::UINT8;
  ImageFormat image_format_ = ImageFormat::UNKOWN;

  std::unique_ptr<MemoryBlock> memory_block_ = nullptr;
  std::shared_ptr<BaseMemoryPool> memory_pool_ = nullptr;

  bool is_local_mempool_ =
      false;  // True:本地创建的内存池，False:外部传入的内存池
};

class ImageFactory {
 public:
  static std::shared_ptr<BaseImage> createImage(
      uint32_t width, uint32_t height, ImageFormat imageFormat,
      ImagePixDataType pixDataType, bool alloc_memory,
      InferencePlatform platform = InferencePlatform::AUTOMATIC);

  static std::shared_ptr<BaseImage> constructImage(void* custom_frame,
                                                   ImageImplType frame_type);

  static std::shared_ptr<BaseImage> readImage(
      const std::string& file_path, bool use_rgb = false,
      InferencePlatform platform = InferencePlatform::AUTOMATIC);
  static int32_t writeImage(const std::string& file_path,
                            const std::shared_ptr<BaseImage>& image);
};
#endif  // BaseImage_H
