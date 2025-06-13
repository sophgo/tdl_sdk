#include "preprocess/bmcv_preprocessor.hpp"
#include <Eigen/Dense>
#include "image/bmcv_image.hpp"
#include "utils/common_utils.hpp"
#include "utils/tdl_log.hpp"
// 使用Eigen优化RGB图像处理
void processImageEigen(uint8_t* ptr, int height, int stride, float alpha,
                       float beta) {
  using namespace Eigen;

  Map<Matrix<uint8_t, Dynamic, Dynamic, RowMajor>> img(ptr, height, stride);

  Matrix<float, Dynamic, Dynamic, RowMajor> result = img.cast<float>();

  result = (result.array() * alpha + beta).matrix();

  result = result.array().min(255.0f).max(0.0f);

  img = result.cast<uint8_t>();
}

BmCVPreprocessor::BmCVPreprocessor() {
  bm_status_t ret = bm_dev_request(&handle_, 0);
  if (ret != BM_SUCCESS) {
    LOGE("bm_dev_request failed! ret: %d", ret);
    assert(0);
  }
}

BmCVPreprocessor::~BmCVPreprocessor() { bm_dev_free(handle_); }

std::shared_ptr<BaseImage> BmCVPreprocessor::preprocess(
    const std::shared_ptr<BaseImage>& src_image, const PreprocessParams& params,
    std::shared_ptr<BaseMemoryPool> memory_pool) {
  if (memory_pool == nullptr) {
    LOGW("input memory_pool is nullptr,use src image memory pool\n");
    memory_pool = src_image->getMemoryPool();
  }
  if (memory_pool == nullptr) {
    LOGE("memory_pool is nullptr!\n");
    return nullptr;
  }

  std::shared_ptr<BmCVImage> bmcv_image = std::make_shared<BmCVImage>(
      params.dst_width, params.dst_height, params.dst_image_format,
      params.dst_pixdata_type, false, memory_pool);
  std::unique_ptr<MemoryBlock> memory_block =
      memory_pool->allocate(bmcv_image->getImageByteSize());
  if (memory_block == nullptr) {
    LOGE("BmCVImage allocate memory failed!\n");
    return nullptr;
  }
  int32_t ret = bmcv_image->setupMemoryBlock(memory_block);
  if (ret != 0) {
    LOGE("BmCVImage setupMemoryBlock failed!\n");
    return nullptr;
  }
  LOGI("setup output image done");

  ret = preprocessToImage(src_image, params, bmcv_image);
  if (ret != 0) {
    LOGE("preprocessToImage failed!\n");
    return nullptr;
  }
  LOGI("preprocessToImage done");
  return bmcv_image;
}

int32_t BmCVPreprocessor::preprocessToImage(
    const std::shared_ptr<BaseImage>& src_image, const PreprocessParams& params,
    std::shared_ptr<BaseImage> dst_image) {
  if (!dst_image->isInitialized()) {
    LOGE("dst_image is not initialized!\n");
    return -1;
  }

  std::vector<float> rescale_params =
      getRescaleConfig(params, src_image->getWidth(), src_image->getHeight());

  int pad_x = (params.crop_x - rescale_params[2]) / rescale_params[0];
  int pad_y = (params.crop_y - rescale_params[3]) / rescale_params[1];

  int resized_w = params.dst_width - pad_x * 2;
  int resized_h = params.dst_height - pad_y * 2;
  LOGI("resized_w:%d,resized_h:%d,pad_x:%d,pad_y:%d,src_w:%d,src_h:%d",
       resized_w, resized_h, pad_x, pad_y, src_image->getWidth(),
       src_image->getHeight());
  // 获取输入bm_image
  bm_image* input = (bm_image*)src_image->getInternalData();
  bm_image* output = (bm_image*)dst_image->getInternalData();
  bmcv_padding_attr_t* padding_attr = new bmcv_padding_attr_t;
  padding_attr->dst_crop_stx = pad_x;
  padding_attr->dst_crop_sty = pad_y;
  padding_attr->dst_crop_w = resized_w;
  padding_attr->dst_crop_h = resized_h;
  padding_attr->padding_r = 0;
  padding_attr->padding_g = 0;
  padding_attr->padding_b = 0;
  padding_attr->if_memset = 1;
  bmcv_rect_t* crop_rect = new bmcv_rect_t;
  crop_rect->start_x = params.crop_x;
  crop_rect->start_y = params.crop_y;
  crop_rect->crop_w =
      params.crop_width > 0 ? params.crop_width : src_image->getWidth();
  crop_rect->crop_h =
      params.crop_height > 0 ? params.crop_height : src_image->getHeight();
  bm_status_t ret = bmcv_image_vpp_convert_padding(
      handle_, 1, *input, output, padding_attr, crop_rect, BMCV_INTER_LINEAR);
  if (ret != BM_SUCCESS) {
    LOGE("bmcv_image_vpp_convert_padding failed, ret:%d", ret);
    return -1;
  }
  if (dst_image->getImageFormat() == ImageFormat::RGB_PLANAR ||
      dst_image->getImageFormat() == ImageFormat::BGR_PLANAR ||
      dst_image->getImageFormat() == ImageFormat::GRAY) {
    bmcv_convert_to_attr convert_to_attr;
    convert_to_attr.alpha_0 = params.scale[0];
    convert_to_attr.alpha_1 = params.scale[1];
    convert_to_attr.alpha_2 = params.scale[2];
    convert_to_attr.beta_0 = -params.mean[0];
    convert_to_attr.beta_1 = -params.mean[1];
    convert_to_attr.beta_2 = -params.mean[2];
    if (output->data_type == DATA_TYPE_EXT_1N_BYTE) {
      bmcv_image_convert_to(handle_, 1, convert_to_attr, output, output);
    } else {
      // 处理其他格式 width != strides[0]的报错
      output->width = dst_image->getStrides()[0];
      bmcv_image_convert_to(handle_, 1, convert_to_attr, output, output);
      output->width = dst_image->getWidth();
    }
  } else if (dst_image->getImageFormat() == ImageFormat::RGB_PACKED ||
             dst_image->getImageFormat() == ImageFormat::BGR_PACKED) {
    // 获取数据指针和平面大小
    std::vector<uint8_t*> base_addr = dst_image->getVirtualAddress();
    std::vector<uint32_t> strides = dst_image->getStrides();
    int alpha = params.scale[0];
    int beta = -params.mean[0];

    // 直接操作内存进行像素变换
    uint8_t* ptr = dst_image->getVirtualAddress()[0];
    int stride = strides[0];
    int height = dst_image->getHeight();
    int width = dst_image->getWidth();
    int plane_num = dst_image->getPlaneNum();
    processImageEigen(ptr, height, stride, alpha, beta);
  } else {
    LOGE(
        "bmcv_image_convert_to only support RGB_PLANAR,BGR_PLANAR,GRAY, "
        "RGB_PACKED,BGR_PACKED, not support %d",
        static_cast<int>(dst_image->getImageFormat()));
    return -1;
  }

  return 0;
}

int32_t BmCVPreprocessor::preprocessToTensor(
    const std::shared_ptr<BaseImage>& src_image, const PreprocessParams& params,
    const int batch_idx, std::shared_ptr<BaseTensor> tensor) {
  LOGI("params.dst_image_format: %d,params.dst_pixdata_type: %d",
       (int)params.dst_image_format, (int)params.dst_pixdata_type);
  std::shared_ptr<BmCVImage> bmcv_image = std::make_shared<BmCVImage>(
      params.dst_width, params.dst_height, params.dst_image_format,
      params.dst_pixdata_type, false);
  std::vector<uint32_t> strides = bmcv_image->getStrides();
  int32_t ret = 0;
  uint32_t tensor_stride = tensor->getWidth() * tensor->getElementSize();
  if (strides[0] == tensor_stride) {
    LOGI("bmcv preprocessor, construct image from input tensor");
    ret = tensor->constructImage(bmcv_image, batch_idx);
    if (ret != 0) {
      LOGE("tensor constructImage failed, ret: %d\n", ret);
      return -1;
    }
  } else {
    LOGI("bmcv preprocessor, image stride:%d, tensor stride:%d", strides[0],
         tensor_stride);
    ret = bmcv_image->allocateMemory();
    if (ret != 0) {
      LOGE("bmcv_image allocateMemory failed, ret: %d\n", ret);
      return -1;
    }
  }
  LOGI(
      "to "
      "preprocessToImage,scale:%f,%f,%f,mean:%f,%f,%f,dst_height:%d,dst_width:%"
      "d,"
      "dst_pixdata_type:%d,dstStride:%d",
      params.scale[0], params.scale[1], params.scale[2], params.mean[0],
      params.mean[1], params.mean[2], params.dst_height, params.dst_width,
      (int)params.dst_pixdata_type, strides[0]);
  ret = preprocessToImage(src_image, params, bmcv_image);
  bm_image* bmcv_image_ptr = (bm_image*)bmcv_image->getInternalData();
  if (ret != 0) {
    LOGE("preprocessToImage failed, ret: %d\n", ret);
    return -1;
  }
  if (strides[0] != tensor->getWidth()) {
    // copy bmcv image to tensor
    LOGI("copy bmcv image to tensor");
    ret = tensor->copyFromImage(bmcv_image, batch_idx);
    if (ret != 0) {
      LOGE("tensor copyFromImage failed, ret: %d\n", ret);
      return -1;
    }
  }
  return ret;
}
