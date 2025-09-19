#include "base_image.hpp"

class BmCVImage : public BaseImage {
 public:
  // 构造与析构
  // BmCVImage();
  ~BmCVImage();

  BmCVImage(uint32_t width, uint32_t height, ImageFormat imageFormat,
            TDLDataType pix_data_type, bool alloc_memory = false,
            std::shared_ptr<BaseMemoryPool> memory_pool = nullptr);

  BmCVImage(const bm_image& bm_image);

  virtual int32_t prepareImageInfo(uint32_t width, uint32_t height,
                                   ImageFormat imageFormat,
                                   TDLDataType pix_data_type,
                                   uint32_t align_size = 0) override;

  virtual int32_t setupMemory(uint64_t phy_addr, uint8_t* vir_addr,
                              uint32_t length) override;

  virtual uint32_t getWidth() const override;
  virtual uint32_t getHeight() const override;
  virtual std::vector<uint32_t> getStrides() const override;
  virtual std::vector<uint64_t> getPhysicalAddress() const override;
  virtual std::vector<uint8_t*> getVirtualAddress() const override;
  virtual uint32_t getPlaneNum() const override;

  virtual uint32_t getImageByteSize() const override;

  virtual uint32_t getInternalType() const override;
  virtual void* getInternalData() const override;

 private:
  int32_t formatBase2Bm(ImageFormat& imageFormat,
                        bm_image_format_ext& bm_format);
  int32_t formatBm2Base(bm_image_format_ext& bm_format,
                        ImageFormat& imageFormat);
  int32_t dataTypeBase2Bm(TDLDataType& pix_data_type,
                          bm_image_data_format_ext& bm_data_format);
  int32_t dataTypeBm2Base(bm_image_data_format_ext& bm_data_format,
                          TDLDataType& pix_data_type);
  int32_t extractImageInfo(const bm_image& bm_image);

 private:
  bm_image bm_image_;
  bm_handle_t handle_;
  uint32_t img_width_;   // 图像宽度
  uint32_t img_height_;  // 图像高度
};
