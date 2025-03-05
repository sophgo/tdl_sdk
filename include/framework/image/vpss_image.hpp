#ifndef VPSS_IMAGE_H
#define VPSS_IMAGE_H

#include "cvi_comm.h"
#include "image/base_image.hpp"

class VPSSImage : public BaseImage {
 public:
  VPSSImage(uint32_t width, uint32_t height, ImageFormat imageFormat,
            TDLDataType pix_data_type, bool alloc_memory = false,
            std::shared_ptr<BaseMemoryPool> memory_pool = nullptr);
  VPSSImage(const VIDEO_FRAME_INFO_S& frame);
  // VPSSImage();
  ~VPSSImage();

  static std::unique_ptr<BaseImage> createImage(
      uint32_t width, uint32_t height, ImageImplType imageType,
      ImageFormat imageFormat, BaseMemoryPool* memory_pool = nullptr);

  int32_t prepareImageInfo(uint32_t width, uint32_t height,
                           ImageFormat imageFormat,
                           TDLDataType pix_data_type) override;

  int32_t setupMemory(uint64_t phy_addr, uint8_t* vir_addr, uint32_t length);

  bool isInitialized() const override;

  uint32_t getWidth() const override;
  uint32_t getHeight() const override;
  std::vector<uint32_t> getStrides() const override;

  uint32_t getInternalType() override;
  void* getInternalData() const override;

  std::vector<uint64_t> getPhysicalAddress() const override;
  std::vector<uint8_t*> getVirtualAddress() const override;
  uint32_t getPlaneNum() const override;
  uint32_t getImageByteSize() const override;
  std::string getDeviceType() const override;
  static PIXEL_FORMAT_E convertPixelFormat(ImageFormat imageFormat,
                                           TDLDataType pix_data_type);
  VIDEO_FRAME_INFO_S* getFrame() const;
  void setFrame(const VIDEO_FRAME_INFO_S& frame);

  PIXEL_FORMAT_E getPixelFormat() const;

  uint32_t getVbPoolId() const;

  // after vpss process,its virtual address would be cleared,
  // so we need to restore it
  int32_t restoreVirtualAddress(bool check_swap_rgb = false);
  int32_t checkToSwapRGB();

 private:
  int32_t extractImageInfo(const VIDEO_FRAME_INFO_S& frame);
  int32_t initFrameInfo(uint32_t width, uint32_t height,
                        ImageFormat imageFormat, TDLDataType pix_data_type,
                        VIDEO_FRAME_INFO_S* frame);

 private:
  VIDEO_FRAME_INFO_S frame_;
  bool is_from_pool_ = false;
};

#endif  // VPSS_IMAGE_H
