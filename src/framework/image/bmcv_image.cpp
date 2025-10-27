#include "image/bmcv_image.hpp"

#include "memory/cpu_memory_pool.hpp"
#include "utils/common_utils.hpp"
#include "utils/tdl_log.hpp"

#include "memory/bm_memory_pool.hpp"

#define ALIGN(x, a) (((x) + ((a)-1)) & ~((a)-1))

// BmCVImage::BmCVImage() {}

BmCVImage::BmCVImage(uint32_t width, uint32_t height, ImageFormat imageFormat,
                     TDLDataType pix_data_type, bool alloc_memory,
                     std::shared_ptr<BaseMemoryPool> memory_pool) {
  // 初始化handle
  bm_status_t ret = bm_dev_request(&handle_, 0);
  if (ret != BM_SUCCESS) {
    LOGE("Failed to initialize BM handle, error code: %d\n", ret);
  }
  prepareImageInfo(width, height, imageFormat, pix_data_type);
  image_type_ = ImageType::BMCV_FRAME;
  memory_pool_ = memory_pool;
  if (memory_pool == nullptr) {
    memory_pool_ = std::make_shared<BmMemoryPool>(handle_);
    LOGI("use BM memory pool");
  }
  if (alloc_memory) {
    int32_t ret = allocateMemory();
    if (ret != 0) {
      LOGE("allocateMemory failed");
      throw std::runtime_error("allocateMemory failed");
    }
  }
}

BmCVImage::BmCVImage(const bm_image& bm_image) {
  bm_image_ = bm_image;
  memory_pool_ = BaseMemoryPoolFactory::createMemoryPool();

  int32_t ret = extractImageInfo(bm_image);
  if (ret != 0) {
    LOGE("extractImageInfo failed, ret: %d", ret);
    throw std::runtime_error("extractImageInfo failed");
  }
}

BmCVImage::~BmCVImage() {
  if (memory_block_ == nullptr) {
    LOGE("memory_block_ is nullptr");
    return;
  }
  LOGI(
      "to destroy BmCVImage,width: %d,height: %d,image_format: "
      "%d,pix_data_type: %d,own_memory: %d,virtual_address: "
      "%p,physical_address: "
      "%p,size: %llu",
      img_width_, img_height_, image_format_, pix_data_type_,
      memory_block_->own_memory, (void*)memory_block_->virtualAddress,
      (void*)memory_block_->physicalAddress, memory_block_->size);
  int32_t ret = bm_image_detach(bm_image_);
  if (ret != BM_SUCCESS) {
    LOGE("bm_image_detach failed, ret: %d", ret);
  }
#if defined(__BM1684X__) || defined(__BM1684__)
  ret = bm_image_destroy(bm_image_);
#else
  ret = bm_image_destroy(&bm_image_);

#endif
  if (ret != BM_SUCCESS) {
    LOGE("bm_image_destroy failed, ret: %d", ret);
  }

  if (memory_block_ && memory_block_->own_memory) {
    if (memory_pool_ != nullptr) {
      ret = memory_pool_->release(memory_block_);
      if (ret != 0) {
        LOGE("memory_pool_->release failed");
      }
    } else {
      LOGE("memory_pool_ is nullptr");
      assert(false);
    }
    memory_block_ = nullptr;
  }
  bm_dev_free(handle_);
}

int32_t BmCVImage::formatBase2Bm(ImageFormat& imageFormat,
                                 bm_image_format_ext& bm_format) {
  switch (imageFormat) {
    case ImageFormat::GRAY:
      bm_format = FORMAT_GRAY;
      break;
    case ImageFormat::RGB_PACKED:
      bm_format = FORMAT_RGB_PACKED;
      break;
    case ImageFormat::BGR_PACKED:
      bm_format = FORMAT_BGR_PACKED;
      break;
    case ImageFormat::RGB_PLANAR:
      bm_format = FORMAT_RGB_PLANAR;
      break;
    case ImageFormat::BGR_PLANAR:
      bm_format = FORMAT_BGR_PLANAR;
      break;
    default:
      LOGE("unsupported image format: %d", static_cast<int>(imageFormat));
      return -1;
  }
  return 0;
}

int32_t BmCVImage::formatBm2Base(bm_image_format_ext& bm_format,
                                 ImageFormat& imageFormat) {
  switch (bm_format) {
    case FORMAT_GRAY:
      imageFormat = ImageFormat::GRAY;
      break;
    case FORMAT_RGB_PACKED:
      imageFormat = ImageFormat::RGB_PACKED;
      break;
    case FORMAT_BGR_PACKED:
      imageFormat = ImageFormat::BGR_PACKED;
      break;
    case FORMAT_RGB_PLANAR:
      imageFormat = ImageFormat::RGB_PLANAR;
      break;
    case FORMAT_BGR_PLANAR:
      imageFormat = ImageFormat::BGR_PLANAR;
      break;
    default:
      LOGE("unsupported image format: %d", static_cast<int>(bm_format));
      return -1;
  }
  return 0;
}

int32_t BmCVImage::dataTypeBase2Bm(TDLDataType& pix_data_type,
                                   bm_image_data_format_ext& bm_data_format) {
  switch (pix_data_type) {
    case TDLDataType::UINT8:
      bm_data_format = DATA_TYPE_EXT_1N_BYTE;
      break;
    case TDLDataType::INT8:
      bm_data_format = DATA_TYPE_EXT_1N_BYTE_SIGNED;
      break;
    case TDLDataType::FP32:
      bm_data_format = DATA_TYPE_EXT_FLOAT32;
      break;
    default:
      return -1;
  }
  return 0;
}

int32_t BmCVImage::dataTypeBm2Base(bm_image_data_format_ext& bm_data_format,
                                   TDLDataType& pix_data_type) {
  switch (bm_data_format) {
    case DATA_TYPE_EXT_1N_BYTE:
      pix_data_type = TDLDataType::UINT8;
      break;
    case DATA_TYPE_EXT_1N_BYTE_SIGNED:
      pix_data_type = TDLDataType::INT8;
      break;
    case DATA_TYPE_EXT_FLOAT32:
      pix_data_type = TDLDataType::FP32;
      break;
    default:
      LOGE("unsupported data type: %d", static_cast<int>(bm_data_format));
      return -1;
  }
  return 0;
}

int32_t BmCVImage::prepareImageInfo(uint32_t width, uint32_t height,
                                    ImageFormat imageFormat,
                                    TDLDataType pix_data_type,
                                    uint32_t align_size) {
  (void)align_size;
  image_format_ = imageFormat;
  pix_data_type_ = pix_data_type;
  img_width_ = width;
  img_height_ = height;
  bm_image_format_ext bm_format;
  bm_image_data_format_ext bm_data_format;
  formatBase2Bm(imageFormat, bm_format);
  dataTypeBase2Bm(pix_data_type, bm_data_format);
  align_size = 64;
  int align_width = ALIGN(img_width_, align_size);
  int* stride;
  if (imageFormat == ImageFormat::GRAY) {
    stride = new int[1];
    stride[0] = align_width;
  } else if (imageFormat == ImageFormat::RGB_PLANAR ||
             imageFormat == ImageFormat::BGR_PLANAR) {
    stride = new int[3];
    stride[0] = align_width;
    stride[1] = align_width;
    stride[2] = align_width;
  } else if (imageFormat == ImageFormat::RGB_PACKED ||
             imageFormat == ImageFormat::BGR_PACKED) {
    stride = new int[1];
    stride[0] = ALIGN(img_width_ * 3, align_size);
  } else {
    LOGE("unsupported image format: %d", static_cast<int>(imageFormat));
    return -1;
  }
  int ret = bm_image_create(handle_, img_height_, img_width_, bm_format,
                            bm_data_format, &bm_image_, stride);
  if (ret != BM_SUCCESS) {
    LOGE("bm_image_create failed, ret: %d", ret);
    return -1;
  }
  LOGI(
      "image width: %d,height: %d,imgformat: %d,pixformat: "
      "%d",
      img_width_, img_height_, static_cast<int>(image_format_),
      static_cast<int>(pix_data_type_));
  delete[] stride;
  return 0;
}

uint32_t BmCVImage::getWidth() const { return img_width_; }
uint32_t BmCVImage::getHeight() const { return img_height_; }

std::vector<uint32_t> BmCVImage::getStrides() const {
  std::vector<uint32_t> strides;
  bm_status_t ret;
  int plane_count = getPlaneNum();
  int stride = 0;
  ret = bm_image_get_stride(bm_image_, &stride);
  if (ret == BM_SUCCESS) {
    for (int i = 0; i < plane_count; i++) {
      strides.push_back(stride);
    }
  } else {
    LOGE("bm_image_get_stride failed, ret: %d", ret);
  }
  return strides;
}

std::vector<uint64_t> BmCVImage::getPhysicalAddress() const {
  std::vector<uint64_t> physical_address;
  int plane_count = getPlaneNum();
  int* plane_size = new int[plane_count];
  bm_status_t ret = bm_image_get_byte_size(bm_image_, plane_size);
  if (image_format_ == ImageFormat::RGB_PLANAR ||
      image_format_ == ImageFormat::BGR_PLANAR) {
    plane_size[1] = plane_size[2] = plane_size[0] = plane_size[0] / 3;
  }
  int offset = 0;
  if (ret == BM_SUCCESS) {
    for (int i = 0; i < plane_count; i++) {
      physical_address.push_back(memory_block_->physicalAddress + offset);
      offset += plane_size[i];
    }
  } else {
    LOGE("bm_image_get_byte_size failed, ret: %d", ret);
  }
  delete[] plane_size;
  return physical_address;
}

std::vector<uint8_t*> BmCVImage::getVirtualAddress() const {
  std::vector<uint8_t*> virtual_address;
  int plane_count = getPlaneNum();
  int* plane_size = new int[plane_count];
  bm_image_get_byte_size(bm_image_, plane_size);
  if (image_format_ == ImageFormat::RGB_PLANAR ||
      image_format_ == ImageFormat::BGR_PLANAR) {
    plane_size[1] = plane_size[2] = plane_size[0] = plane_size[0] / 3;
  }
  int offset = 0;
  for (int i = 0; i < plane_count; i++) {
    virtual_address.push_back((uint8_t*)memory_block_->virtualAddress + offset);
    offset += plane_size[i];
  }
  delete[] plane_size;
  return virtual_address;
}

uint32_t BmCVImage::getPlaneNum() const {
  if (image_format_ == ImageFormat::RGB_PLANAR ||
      image_format_ == ImageFormat::BGR_PLANAR) {
    return 3;
  }
  return bm_image_get_plane_num(bm_image_);
}

uint32_t BmCVImage::getInternalType() const {
  return static_cast<uint32_t>(bm_image_.image_format);
}

void* BmCVImage::getInternalData() const { return (void*)&bm_image_; }

uint32_t BmCVImage::getImageByteSize() const {
  uint32_t size = 0;
  int plane_count = getPlaneNum();
  int* plane_size = new int[plane_count];
  bm_image_get_byte_size(bm_image_, plane_size);
  if (image_format_ == ImageFormat::RGB_PLANAR ||
      image_format_ == ImageFormat::BGR_PLANAR) {
    plane_size[1] = plane_size[2] = plane_size[0] = plane_size[0] / 3;
  }
  for (int i = 0; i < plane_count; i++) {
    size += plane_size[i];
  }
  delete[] plane_size;
  return size;
}

int32_t BmCVImage::setupMemory(uint64_t phy_addr, uint8_t* vir_addr,
                               uint32_t length) {
  LOGI("setupMemory, phy_addr: %p, vir_addr: %p, length: %d", (void*)phy_addr,
       (void*)vir_addr, length);

  bm_device_mem_t* device_memory;
  if (memory_block_ == nullptr) {
    memory_block_ = std::make_unique<MemoryBlock>();
    memory_block_->physicalAddress = phy_addr;
    memory_block_->virtualAddress = vir_addr;
    memory_block_->size = length;
    memory_block_->own_memory = false;
    memory_block_->handle = new bm_device_mem_t();
    device_memory = (bm_device_mem_t*)memory_block_->handle;
    device_memory->u.device.device_addr = phy_addr;
    device_memory->size = length;
  } else {
    device_memory = (bm_device_mem_t*)memory_block_->handle;
  }
  bm_status_t ret = bm_image_attach(bm_image_, device_memory);
  if (ret != BM_SUCCESS) {
    LOGE("bm_image_attach failed, ret: %d", ret);
    return -1;
  }

  return 0;
}

int32_t BmCVImage::extractImageInfo(const bm_image& bm_image) {
  bm_image_ = bm_image;
  bm_device_mem_t* device_memory;
  int32_t ret = bm_image_attach(bm_image_, device_memory);
  if (ret != BM_SUCCESS) {
    LOGE("bm_image_attach failed, ret: %d", ret);
    return -1;
  }

  bm_image_format_ext bm_format = bm_image_.image_format;
  bm_image_data_format_ext bm_data_format = bm_image_.data_type;

  image_type_ = ImageType::BMCV_FRAME;

  formatBm2Base(bm_format, image_format_);
  dataTypeBm2Base(bm_data_format, pix_data_type_);

  memory_block_ = std::make_unique<MemoryBlock>();
  memory_block_->id = 0;
  memory_block_->size = getImageByteSize();
  memory_block_->physicalAddress = device_memory->u.device.device_addr;
  memory_block_->virtualAddress = device_memory->u.system.system_addr;
  memory_block_->own_memory = false;
  memory_block_->handle = device_memory;

  return 0;
}