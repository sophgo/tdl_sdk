#ifndef BaseImage_H
#define BaseImage_H

#include <memory>
#include <string>
#include <vector>

#include "common/common_types.hpp"
#include "memory/base_memory_pool.hpp"

#ifndef NO_OPENCV
#include <opencv2/opencv.hpp>
#endif

class BaseImage {
 public:
  virtual ~BaseImage();
  // BaseImage(uint32_t width, uint32_t height, ImageFormat imageFormat,
  //           TDLDataType pix_data_type, bool alloc_memory = false,
  //           std::shared_ptr<BaseMemoryPool> memory_pool = nullptr);
  BaseImage(ImageType image_type = ImageType::RAW_FRAME);

  virtual int32_t prepareImageInfo(uint32_t width,
                                   uint32_t height,
                                   ImageFormat imageFormat,
                                   TDLDataType pix_data_type,
                                   uint32_t align_size = 0);
  virtual int32_t allocateMemory();
  virtual int32_t freeMemory();
  virtual bool isInitialized() const;
  virtual bool isAligned() const;

  virtual uint32_t getWidth() const { return width_; }
  virtual uint32_t getHeight() const { return height_; }
  virtual std::vector<uint32_t> getStrides() const { return strides_; }
  virtual std::vector<uint64_t> getPhysicalAddress() const;
  virtual std::vector<uint8_t*> getVirtualAddress() const;
  virtual uint32_t getPlaneNum() const { return plane_num_; }

  virtual uint32_t getImageByteSize() const { return img_bytes_; }
  virtual ImageFormat getImageFormat() const { return image_format_; }

  virtual ImageType getImageType() const { return image_type_; }
  virtual TDLDataType getPixDataType() const { return pix_data_type_; }
  virtual std::unique_ptr<MemoryBlock>& getMemoryBlock() {
    return memory_block_;
  }

  virtual bool isPlanar() const;

  virtual uint32_t getInternalType() const { return 0; }
  virtual void* getInternalData() const { return nullptr; }

  // 当引用模型输入作为预处理输出结果时，需要进行内存同步
  virtual int32_t invalidateCache();
  virtual int32_t flushCache();

  virtual int32_t randomFill();

  virtual int32_t setupMemoryBlock(std::unique_ptr<MemoryBlock>& memory_block);

  std::shared_ptr<BaseMemoryPool> getMemoryPool() { return memory_pool_; }
  virtual int32_t setMemoryPool(std::shared_ptr<BaseMemoryPool> memory_pool);
  int32_t copyFromBuffer(const uint8_t* buffer, uint32_t size);

 protected:
  virtual int32_t setupMemory(uint64_t phy_addr,
                              uint8_t* vir_addr,
                              uint32_t length);

 private:
  MemoryPoolType getMemoryPoolType();
  int32_t initImageInfo();

 protected:
  ImageType image_type_ = ImageType::UNKOWN;
  TDLDataType pix_data_type_ = TDLDataType::UINT8;
  ImageFormat image_format_ = ImageFormat::UNKOWN;

  std::unique_ptr<MemoryBlock> memory_block_ = nullptr;
  std::shared_ptr<BaseMemoryPool> memory_pool_ = nullptr;

  bool is_local_mempool_ =
      false;  // True:本地创建的内存池，False:外部传入的内存池

  uint32_t width_ = 0;
  uint32_t height_ = 0;
  std::vector<uint32_t> strides_;
  uint32_t plane_num_ = 0;
  uint32_t img_bytes_ = 0;
  uint32_t align_size_ = 0;
};

class ImageFactory {
 public:
  static std::shared_ptr<BaseImage> createImage(
      uint32_t width,
      uint32_t height,
      ImageFormat imageFormat,
      TDLDataType pixDataType,
      bool alloc_memory,
      InferencePlatform platform = InferencePlatform::AUTOMATIC);

  static std::shared_ptr<BaseImage> constructImage(void* custom_frame,
                                                   ImageType frame_type);

  static std::shared_ptr<BaseImage> readImage(
      const std::string& file_path,
      bool use_rgb = false,
      InferencePlatform platform = InferencePlatform::AUTOMATIC);
  static int32_t writeImage(const std::string& file_path,
                            const std::shared_ptr<BaseImage>& image);

  static std::shared_ptr<BaseImage> alignFace(
      const std::shared_ptr<BaseImage>& image,
      const float* src_landmark_xy,
      const float* dst_landmark_xy,
      int num_points,
      std::shared_ptr<BaseMemoryPool> memory_pool);

  static std::shared_ptr<BaseImage> alignLicensePlate(
      const std::shared_ptr<BaseImage>& image,
      const float* src_landmark_xy,
      const float* dst_landmark_xy,
      int num_points,
      std::shared_ptr<BaseMemoryPool> memory_pool);

  static std::shared_ptr<BaseImage> wrapVPSSFrame(void* vpss_frame,
                                                  bool own_memory);
  static std::shared_ptr<BaseImage> wrapMat(void* mat_frame, bool is_rgb);
#ifndef NO_OPENCV
  static std::shared_ptr<BaseImage> convertFromMat(cv::Mat& mat,
                                                   bool is_rgb = false);

  // 返回的Mat引用image内的内存,如果后续对mat操作,将影响image的内容
  static int32_t convertToMat(std::shared_ptr<BaseImage>& image,
                              cv::Mat& mat,
                              bool& is_rgb);
#endif
};
#endif  // BaseImage_H
