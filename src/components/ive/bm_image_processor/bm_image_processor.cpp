#include "bm_image_processor.hpp"
#include <memory>
#include <vector>
#include "image/base_image.hpp"

extern int cpu_2way_blend(int lwidth, int lheight, unsigned char *left_img,
                          int rwidth, int rheight, unsigned char *right_img,
                          int bwidth, int bheight, unsigned char *blend_img,
                          int overlay_lx, int overlay_rx, unsigned char *wgt,
                          int channel, int format, int wgt_mode);

int32_t ImageFormatToPixelFormat(ImageFormat image_format, CVI_S32 &format) {
  switch (image_format) {
    case ImageFormat::GRAY:
      format = PIXEL_FORMAT_YUV_400;
      break;
    case ImageFormat::RGB_PLANAR:
      format = PIXEL_FORMAT_RGB_888_PLANAR;
      break;
    case ImageFormat::BGR_PLANAR:
      format = PIXEL_FORMAT_BGR_888_PLANAR;
      break;
    case ImageFormat::RGB_PACKED:
      format = PIXEL_FORMAT_RGB_888;
      break;
    case ImageFormat::BGR_PACKED:
      format = PIXEL_FORMAT_BGR_888;
      break;
    case ImageFormat::YUV420P_UV:
      format = PIXEL_FORMAT_YUV_PLANAR_420;
      break;
    default:
      printf("Image format not supported: %d\n",
             static_cast<int>(image_format));
      return -1;
  }
  return 0;
}

int32_t getChannelSize(CVI_S32 format, CVI_S32 width, CVI_S32 height,
                       int src1_size[3]) {
  switch (format) {
    case PIXEL_FORMAT_YUV_PLANAR_420:
      src1_size[0] = height * width;
      src1_size[1] = BM_ALIGN(width / 2, 2) * BM_ALIGN(height / 2, 2);
      src1_size[2] = BM_ALIGN(width / 2, 2) * BM_ALIGN(height / 2, 2);
      break;
    case PIXEL_FORMAT_YUV_400:
      src1_size[0] = height * width;
      src1_size[1] = src1_size[2] = 0;
      break;
    case PIXEL_FORMAT_RGB_888_PLANAR:
    case PIXEL_FORMAT_BGR_888_PLANAR:
      src1_size[0] = src1_size[1] = src1_size[2] = width * height;
      break;
    case PIXEL_FORMAT_RGB_888:
    case PIXEL_FORMAT_BGR_888:
      src1_size[0] = src1_size[1] = src1_size[2] = width * height;
      break;
    default:
      printf("Pixel format(%d) not supported!\n", format);
      return -1;
      break;
  }
  return 0;
}

BmImageProcessor::BmImageProcessor() {
  // 初始化handle
  bm_status_t ret = bm_dev_request(&handle_, 0);
  if (ret != BM_SUCCESS) {
    printf("Failed to initialize BM handle, error code: %d\n", ret);
  }
  // 加载TPU模块
#ifdef USE_CV184X
  tpu_module_ = tpu_kernel_load_module_file(
      handle_, "/mnt/tpu_files/lib/libtpu_kernel_module.so");
#else
  tpu_module_ = tpu_kernel_load_module_file(handle_, "");
#endif
  if (!tpu_module_) {
    printf("Failed to load TPU kernel module\n");
    bm_dev_free(handle_);
    return;
  }
}

BmImageProcessor::~BmImageProcessor() {
  // 释放TPU模块
  tpu_kernel_unload_module(handle_, tpu_module_);
  // 释放handle
  bm_dev_free(handle_);
}

int32_t BmImageProcessor::subads(std::shared_ptr<BaseImage> &src1,
                                 std::shared_ptr<BaseImage> &src2,
                                 std::shared_ptr<BaseImage> &dst) {
  // 参数校验
  if (!src1 || !src2) {
    printf("src1 or src2 is nullptr\n");
    return -1;  // 参数无效
  }

  if (src1->getHeight() != src2->getHeight() ||
      src1->getWidth() != src2->getWidth() ||
      src1->getImageFormat() != src2->getImageFormat() ||
      src1->getPlaneNum() != src2->getPlaneNum()) {
    printf("src1 or src2 is nullptr\n");
    return -1;  // 参数无效
  }

  // 从BaseImage获取必要的参数
  CVI_S32 height = src1->getHeight();
  CVI_S32 width = src1->getWidth();
  std::vector<uint32_t> strides = src1->getStrides();
  CVI_S32 aligned_width = strides[0];
  ImageFormat image_format = src1->getImageFormat();
  TDLDataType pix_data_type = src1->getPixDataType();
  CVI_S32 channel = (image_format == ImageFormat::GRAY) ? 1 : 3;

  if (!dst || dst->getHeight() != height || dst->getWidth() != width ||
      dst->getImageFormat() != image_format ||
      dst->getPixDataType() != pix_data_type) {
    dst = ImageFactory::createImage(width, height, image_format, pix_data_type,
                                    true, InferencePlatform::AUTOMATIC);
  }

  // 将ImageFormat转换为PIXEL_FORMAT_E
  CVI_S32 format;
  ImageFormatToPixelFormat(image_format, format);
  // 设备内存
  bm_device_mem_t src1_img_mem[3];
  bm_device_mem_t src2_img_mem[3];
  bm_device_mem_t dst_img_mem[3];

#if defined(USE_CMODEL_CV184X)
  // 获取虚拟地址
  std::vector<uint8_t *> src1_vir_addrs_init = src1->getVirtualAddress();
  std::vector<uint8_t *> src2_vir_addrs_init = src2->getVirtualAddress();
  std::vector<uint8_t *> dst_vir_addrs_init = dst->getVirtualAddress();
  // 计算每个通道的大小
  int src1_size[3] = {0};
  int src2_size[3] = {0};
  int dst_size[3] = {0};
  // 根据格式获取各通道大小
  getChannelSize(format, width, height, src1_size);
  getChannelSize(format, width, height, src2_size);
  getChannelSize(format, width, height, dst_size);
  unsigned char *src1_vir_addrs[3] = {
      src1_vir_addrs_init[0], src1_vir_addrs_init[0] + src1_size[0],
      src1_vir_addrs_init[0] + src1_size[0] + src1_size[1]};
  unsigned char *src2_vir_addrs[3] = {
      src2_vir_addrs_init[0], src2_vir_addrs_init[0] + src2_size[0],
      src2_vir_addrs_init[0] + src2_size[0] + src2_size[1]};
  unsigned char *dst_vir_addrs[3] = {
      dst_vir_addrs_init[0], dst_vir_addrs_init[0] + dst_size[0],
      dst_vir_addrs_init[0] + dst_size[0] + dst_size[1]};
  // 为每个通道分配设备内存并拷贝数据
  for (int c = 0; c < channel; c++) {
    // 对每个通道分配设备内存
    bm_malloc_device_byte(handle_, &src1_img_mem[c],
                          sizeof(unsigned char) * src1_size[c]);
    bm_malloc_device_byte(handle_, &src2_img_mem[c],
                          sizeof(unsigned char) * src2_size[c]);
    bm_malloc_device_byte(handle_, &dst_img_mem[c],
                          sizeof(unsigned char) * dst_size[c]);
    // 将主机内存数据拷贝到设备内存
    bm_memcpy_s2d(handle_, src1_img_mem[c], src1_vir_addrs[c]);
    bm_memcpy_s2d(handle_, src2_img_mem[c], src2_vir_addrs[c]);
  }

#else
  // 获取基础设备内存
  // src1_img_mem[0] =
  //     *reinterpret_cast<bm_device_mem_t *>(src1->getMemoryBlock()->handle);
  // src2_img_mem[0] =
  //     *reinterpret_cast<bm_device_mem_t *>(src2->getMemoryBlock()->handle);
  // dst_img_mem[0] =
  //     *reinterpret_cast<bm_device_mem_t *>(dst->getMemoryBlock()->handle);
  src1_img_mem[0].u.device.device_addr =
      src1->getMemoryBlock()->physicalAddress;
  src2_img_mem[0].u.device.device_addr =
      src2->getMemoryBlock()->physicalAddress;
  dst_img_mem[0].u.device.device_addr = dst->getMemoryBlock()->physicalAddress;
  // 更新第一个通道的size
  src1_img_mem[0].size = strides[0] * height;
  src2_img_mem[0].size = strides[0] * height;
  dst_img_mem[0].size = strides[0] * height;

  // 计算其他通道的偏移地址
  for (int c = 1; c < channel; c++) {
    // 复制整个结构体，保持flags等信息不变
    memcpy(&src1_img_mem[c], &src1_img_mem[0], sizeof(bm_device_mem_t));
    memcpy(&src2_img_mem[c], &src2_img_mem[0], sizeof(bm_device_mem_t));
    memcpy(&dst_img_mem[c], &dst_img_mem[0], sizeof(bm_device_mem_t));

    // 只调整设备地址和大小
    src1_img_mem[c].u.device.device_addr =
        src1_img_mem[c - 1].u.device.device_addr + src1_img_mem[c - 1].size;
    src2_img_mem[c].u.device.device_addr =
        src2_img_mem[c - 1].u.device.device_addr + src2_img_mem[c - 1].size;
    dst_img_mem[c].u.device.device_addr =
        dst_img_mem[c - 1].u.device.device_addr + dst_img_mem[c - 1].size;

    src1_img_mem[c].size = strides[c] * height;
    src2_img_mem[c].size = strides[c] * height;
    dst_img_mem[c].size = strides[c] * height;
  }

#endif
  bm_status_t ret =
      tpu_cv_subads(handle_, height, aligned_width, format, channel,
                    src1_img_mem, src2_img_mem, dst_img_mem, tpu_module_);

  if (ret != BM_SUCCESS) {
    printf("tpu_cv_subads failed\n");
    // 释放设备内存
#if defined(USE_CMODEL_CV184X)
    for (int c = 0; c < channel; c++) {
      bm_free_device(handle_, src1_img_mem[c]);
      bm_free_device(handle_, src2_img_mem[c]);
      bm_free_device(handle_, dst_img_mem[c]);
    }
#endif
    return static_cast<int32_t>(ret);
  }

#if defined(USE_CMODEL_CV184X)
  // 将结果从设备内存拷贝回主机内存
  for (int c = 0; c < channel; c++) {
    bm_memcpy_d2s(handle_, dst_vir_addrs[c], dst_img_mem[c]);
  }

  // 释放设备内存
  for (int c = 0; c < channel; c++) {
    bm_free_device(handle_, src1_img_mem[c]);
    bm_free_device(handle_, src2_img_mem[c]);
    bm_free_device(handle_, dst_img_mem[c]);
  }
#endif
  return 0;
}

int32_t BmImageProcessor::thresholdProcess(std::shared_ptr<BaseImage> &input,
                                           std::shared_ptr<BaseImage> &output,
                                           CVI_U32 threshold_type,
                                           CVI_U32 threshold,
                                           CVI_U32 max_value) {
  // 参数校验
  if (!input) {
    printf("input is nullptr\n");
    return -1;  // 参数无效
  }

  // 从BaseImage获取必要的参数
  CVI_S32 height = input->getHeight();
  CVI_S32 width = input->getWidth();
  std::vector<uint32_t> strides = input->getStrides();
  CVI_S32 aligned_width = strides[0];
  ImageFormat image_format = input->getImageFormat();
  TDLDataType pix_data_type = input->getPixDataType();
  if (!output || output->getHeight() != height || output->getWidth() != width ||
      output->getImageFormat() != image_format ||
      output->getPixDataType() != pix_data_type) {
    output =
        ImageFactory::createImage(width, height, image_format, pix_data_type,
                                  true, InferencePlatform::AUTOMATIC);
  }

  CVI_S32 format;
  ImageFormatToPixelFormat(image_format, format);
  if (format != PIXEL_FORMAT_YUV_400) {
    printf("Image format not supported for thresholdProcess: %d\n",
           static_cast<int>(image_format));
    printf("Only grayscale images are supported for threshold operation.\n");
    return -1;
  }

  TPU_THRESHOLD_TYPE mode = static_cast<TPU_THRESHOLD_TYPE>(threshold_type);
  // 设备内存
  bm_device_mem_t input_mem;
  bm_device_mem_t output_mem;

#if defined(USE_CMODEL_CV184X)
  // 获取虚拟地址
  std::vector<uint8_t *> input_vir_addrs_init = input->getVirtualAddress();
  std::vector<uint8_t *> output_vir_addrs_init = output->getVirtualAddress();

  // 计算图像大小
  int input_size[3] = {0};
  getChannelSize(format, width, height, input_size);

  // 分配设备内存并拷贝数据
  bm_malloc_device_byte(handle_, &input_mem,
                        sizeof(unsigned char) * input_size[0]);
  bm_malloc_device_byte(handle_, &output_mem,
                        sizeof(unsigned char) * input_size[0]);

  // 将主机内存数据拷贝到设备内存
  bm_memcpy_s2d(handle_, input_mem, input_vir_addrs_init[0]);

#else
  std::unique_ptr<MemoryBlock> &input_mem_block = input->getMemoryBlock();
  std::unique_ptr<MemoryBlock> &output_mem_block = output->getMemoryBlock();
  // input_mem = *reinterpret_cast<bm_device_mem_t *>(input_mem_block->handle);
  // output_mem = *reinterpret_cast<bm_device_mem_t
  // *>(output_mem_block->handle);
  input_mem.u.device.device_addr = input_mem_block->physicalAddress;
  output_mem.u.device.device_addr = output_mem_block->physicalAddress;
  input_mem.size = input_mem_block->size;
  output_mem.size = output_mem_block->size;
#endif

  bm_status_t ret =
      tpu_cv_threshold(handle_, height, aligned_width, mode, threshold,
                       max_value, &input_mem, &output_mem, tpu_module_);

  if (ret != BM_SUCCESS) {
    printf("tpu_cv_threshold failed\n");
    // 释放设备内存
    bm_free_device(handle_, input_mem);
    bm_free_device(handle_, output_mem);
    return static_cast<int32_t>(ret);
  }
#if defined(USE_CMODEL_CV184X)
  // 将结果从设备内存拷贝回主机内存
  bm_memcpy_d2s(handle_, output_vir_addrs_init[0], output_mem);
  // 释放设备内存
  bm_free_device(handle_, input_mem);
  bm_free_device(handle_, output_mem);
#endif
  return 0;
}

int32_t BmImageProcessor::twoWayBlending(std::shared_ptr<BaseImage> &left,
                                         std::shared_ptr<BaseImage> &right,
                                         std::shared_ptr<BaseImage> &output,
                                         CVI_S32 overlay_lx, CVI_S32 overlay_rx,
                                         CVI_U8 *wgt) {
  // 参数校验
  if ((!left || !right)) {
    printf("left or right is nullptr\n");
    return -1;  // 参数无效
  }

  // 从BaseImage获取必要的参数
  CVI_S32 lwidth = left->getWidth();
  CVI_S32 lheight = left->getHeight();
  CVI_S32 rwidth = right->getWidth();
  CVI_S32 rheight = right->getHeight();
  ImageFormat image_format = left->getImageFormat();
  CVI_S32 overlay_w = overlay_rx - overlay_lx + 1;
  CVI_S32 blend_w = lwidth + rwidth - overlay_w;
  CVI_S32 blend_h = lheight;
  CVI_S32 channel = (image_format == ImageFormat::GRAY) ? 1 : 3;

  // 创建输出图像
  if (!output || output->getHeight() != blend_h ||
      output->getWidth() != blend_w ||
      output->getImageFormat() != image_format ||
      output->getPixDataType() != left->getPixDataType()) {
    output = ImageFactory::createImage(blend_w, blend_h, image_format,
                                       left->getPixDataType(), true,
                                       InferencePlatform::AUTOMATIC);
  }

  std::vector<uint32_t> left_strides = left->getStrides();
  std::vector<uint32_t> right_strides = right->getStrides();
  std::vector<uint32_t> output_strides = output->getStrides();
  CVI_S32 aligned_left_width = left_strides[0];
  CVI_S32 aligned_right_width = right_strides[0];
  CVI_S32 aligned_blend_w = output_strides[0];

  // 将ImageFormat转换为PIXEL_FORMAT_E
  CVI_S32 format;
  ImageFormatToPixelFormat(image_format, format);

  // 获取虚拟地址
  std::vector<uint8_t *> left_vir_addrs_init = left->getVirtualAddress();
  std::vector<uint8_t *> right_vir_addrs_init = right->getVirtualAddress();
  std::vector<uint8_t *> output_vir_addrs_init = output->getVirtualAddress();

  // 分配设备内存并拷贝数据
  bm_handle_t handle;
  bm_dev_request(&handle, 0);
  // 设备内存
  bm_device_mem_t left_img_mem[3];
  bm_device_mem_t right_img_mem[3];
  bm_device_mem_t output_img_mem[3];
#if defined(USE_CMODEL_CV184X)
  // 计算各通道大小
  int left_size[3] = {0};
  int right_size[3] = {0};
  int output_size[3] = {0};

  // 计算各通道大小
  getChannelSize(format, lwidth, lheight, left_size);
  getChannelSize(format, rwidth, rheight, right_size);
  getChannelSize(format, blend_w, blend_h, output_size);

  unsigned char *left_vir_addrs[3] = {
      left_vir_addrs_init[0], left_vir_addrs_init[0] + left_size[0],
      left_vir_addrs_init[0] + left_size[0] + left_size[1]};
  unsigned char *right_vir_addrs[3] = {
      right_vir_addrs_init[0], right_vir_addrs_init[0] + right_size[0],
      right_vir_addrs_init[0] + right_size[0] + right_size[1]};
  unsigned char *output_vir_addrs[3] = {
      output_vir_addrs_init[0], output_vir_addrs_init[0] + output_size[0],
      output_vir_addrs_init[0] + output_size[0] + output_size[1]};
  for (int c = 0; c < channel; c++) {
    bm_malloc_device_byte(handle, &left_img_mem[c],
                          sizeof(unsigned char) * left_size[c]);
    bm_malloc_device_byte(handle, &right_img_mem[c],
                          sizeof(unsigned char) * right_size[c]);
    bm_malloc_device_byte(handle, &output_img_mem[c],
                          sizeof(unsigned char) * output_size[c]);

    // 将主机内存数据拷贝到设备内存
    bm_memcpy_s2d(handle, left_img_mem[c], left_vir_addrs[c]);
    bm_memcpy_s2d(handle, right_img_mem[c], right_vir_addrs[c]);
    bm_memcpy_s2d(handle, output_img_mem[c], output_vir_addrs[c]);
  }

#else
  // 获取基础设备内存
  // left_img_mem[0] =
  //     *reinterpret_cast<bm_device_mem_t *>(left->getMemoryBlock()->handle);
  // right_img_mem[0] =
  //     *reinterpret_cast<bm_device_mem_t *>(right->getMemoryBlock()->handle);
  // output_img_mem[0] =
  //     *reinterpret_cast<bm_device_mem_t *>(output->getMemoryBlock()->handle);
  left_img_mem[0].u.device.device_addr =
      left->getMemoryBlock()->physicalAddress;
  right_img_mem[0].u.device.device_addr =
      right->getMemoryBlock()->physicalAddress;
  output_img_mem[0].u.device.device_addr =
      output->getMemoryBlock()->physicalAddress;
  left_img_mem[0].size = left_strides[0] * lheight;
  right_img_mem[0].size = right_strides[0] * rheight;
  output_img_mem[0].size = output_strides[0] * blend_h;
  for (int c = 1; c < channel; c++) {
    memcpy(&left_img_mem[c], &left_img_mem[0], sizeof(bm_device_mem_t));
    memcpy(&right_img_mem[c], &right_img_mem[0], sizeof(bm_device_mem_t));
    memcpy(&output_img_mem[c], &output_img_mem[0], sizeof(bm_device_mem_t));

    left_img_mem[c].u.device.device_addr =
        left_img_mem[c - 1].u.device.device_addr + left_img_mem[c - 1].size;
    right_img_mem[c].u.device.device_addr =
        right_img_mem[c - 1].u.device.device_addr + right_img_mem[c - 1].size;
    output_img_mem[c].u.device.device_addr =
        output_img_mem[c - 1].u.device.device_addr + output_img_mem[c - 1].size;

    left_img_mem[c].size = left_strides[c] * lheight;
    right_img_mem[c].size = right_strides[c] * rheight;
    output_img_mem[c].size = output_strides[c] * blend_h;
  }

#endif

  // 分配设备内存
  bm_device_mem_t wgt_mem[2] = {0};
  int wgt_size = overlay_w * blend_h * sizeof(unsigned char);
  bm_malloc_device_byte(handle, &wgt_mem[0], wgt_size);
  bm_malloc_device_byte(handle, &wgt_mem[1], 1);

#if defined(USE_CMODEL_CV184X)
  bm_memcpy_s2d(handle, wgt_mem[0], wgt);
#else
  wgt_mem[0].flags.u.mem_type = BM_MEM_TYPE_DEVICE;
  wgt_mem[0].size = wgt_size;

  bm_device_mem_t *p_dev = new bm_device_mem_t();
  bm_malloc_device_byte(handle, p_dev, wgt_size);
  unsigned long long addr;
  bm_mem_mmap_device_mem(handle, (bm_device_mem_t *)p_dev, &addr);
  memcpy((unsigned char *)addr, wgt, wgt_size);
  wgt_mem[0] = *p_dev;
#endif

  bm_status_t ret = tpu_2way_blending(
      handle, aligned_left_width, lheight, aligned_right_width, rheight,
      left_img_mem, right_img_mem, aligned_blend_w, blend_h, output_img_mem,
      overlay_lx, overlay_rx, wgt_mem, TPU_BLEND_WGT_MODE::WGT_YUV_SHARE,
      format, channel, tpu_module_);

#if defined(USE_CMODEL_CV184X)
  // 将结果从设备内存拷贝回主机内存
  for (int c = 0; c < channel; c++) {
    bm_memcpy_d2s(handle, output_vir_addrs[c], output_img_mem[c]);
  }

  // int cpu_output_size = output_size[0] + output_size[1] + output_size[2];
  // unsigned char *cpu_blend_output =
  //     (unsigned char *)malloc(cpu_output_size * sizeof(unsigned char));
  // memset(cpu_blend_output, 0, cpu_output_size * sizeof(unsigned char));
  // cpu_2way_blend(lwidth, lheight, left_vir_addrs[0], rwidth, rheight,
  //                right_vir_addrs[0], blend_w, blend_h, cpu_blend_output,
  //                overlay_lx, overlay_rx, wgt, channel, format,
  //                TPU_BLEND_WGT_MODE::WGT_YUV_SHARE);

  // compareResult(output_vir_addrs[0], cpu_blend_output, cpu_output_size);

  // 释放设备内存
  for (int c = 0; c < channel; c++) {
    bm_free_device(handle, left_img_mem[c]);
    bm_free_device(handle, right_img_mem[c]);
    bm_free_device(handle, output_img_mem[c]);
  }
  bm_free_device(handle, wgt_mem[0]);
  bm_free_device(handle, wgt_mem[1]);
#else
  bm_mem_unmap_device_mem(handle, (void *)addr, wgt_size);
  bm_free_device(handle, *p_dev);
  delete p_dev;
#endif
  bm_dev_free(handle);
  return 0;
}

int32_t BmImageProcessor::compareResult(CVI_U8 *tpu_result, CVI_U8 *cpu_result,
                                        CVI_S32 size) {
  for (int i = 0; i < size; i++) {
    if (tpu_result[i] != cpu_result[i]) {
      printf("cpu and tpu result mismatch.\n");
      return -1;
    }
  }
  printf("cpu and tpu result same.\n");
  return 0;
}
