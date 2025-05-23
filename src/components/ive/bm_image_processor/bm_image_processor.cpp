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
}

BmImageProcessor::~BmImageProcessor() {}

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
  CVI_S32 height = static_cast<CVI_S32>(src1->getHeight());
  CVI_S32 width = static_cast<CVI_S32>(src1->getWidth());
  ImageFormat image_format = src1->getImageFormat();
  TDLDataType pix_data_type = src1->getPixDataType();
  CVI_S32 channel = (image_format == ImageFormat::GRAY) ? 1 : 3;

  dst = ImageFactory::createImage(width, height, image_format, pix_data_type,
                                  true, InferencePlatform::AUTOMATIC);

  // 获取虚拟地址
  std::vector<uint8_t *> src1_vir_addrs_init = src1->getVirtualAddress();
  std::vector<uint8_t *> src2_vir_addrs_init = src2->getVirtualAddress();
  std::vector<uint8_t *> dst_vir_addrs_init = dst->getVirtualAddress();

  // 将ImageFormat转换为PIXEL_FORMAT_E
  CVI_S32 format;
  ImageFormatToPixelFormat(image_format, format);

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
  // 创建设备内存
  bm_device_mem_t src1_img_mem[3];
  bm_device_mem_t src2_img_mem[3];
  bm_device_mem_t dst_img_mem[3];
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

  // 调用tpu_cv_subads
  bm_status_t ret = tpu_cv_subads(handle_, height, width, format, channel,
                                  src1_img_mem, src2_img_mem, dst_img_mem);
  if (ret != BM_SUCCESS) {
    printf("tpu_cv_subads failed\n");
    // 释放设备内存
    for (int c = 0; c < channel; c++) {
      bm_free_device(handle_, src1_img_mem[c]);
      bm_free_device(handle_, src2_img_mem[c]);
      bm_free_device(handle_, dst_img_mem[c]);
    }
    return static_cast<int32_t>(ret);
  }

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
  // 释放handle
  bm_dev_free(handle_);
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
  CVI_S32 height = static_cast<CVI_S32>(input->getHeight());
  CVI_S32 width = static_cast<CVI_S32>(input->getWidth());
  ImageFormat image_format = input->getImageFormat();
  TDLDataType pix_data_type = input->getPixDataType();
  output = ImageFactory::createImage(width, height, image_format, pix_data_type,
                                     true, InferencePlatform::AUTOMATIC);

  CVI_S32 format;
  ImageFormatToPixelFormat(image_format, format);
  if (format != PIXEL_FORMAT_YUV_400) {
    printf("Image format not supported for thresholdProcess: %d\n",
           static_cast<int>(image_format));
    printf("Only grayscale images are supported for threshold operation.\n");
    return -1;
  }

  TPU_THRESHOLD_TYPE mode = static_cast<TPU_THRESHOLD_TYPE>(threshold_type);

  // 获取虚拟地址
  std::vector<uint8_t *> input_vir_addrs_init = input->getVirtualAddress();
  std::vector<uint8_t *> output_vir_addrs_init = output->getVirtualAddress();

  // 计算图像大小
  int input_size[3] = {0};
  getChannelSize(format, width, height, input_size);

  // 分配设备内存
  bm_device_mem_t input_mem;
  bm_device_mem_t output_mem;

  // 分配设备内存并拷贝数据
  bm_malloc_device_byte(handle_, &input_mem,
                        sizeof(unsigned char) * input_size[0]);
  bm_malloc_device_byte(handle_, &output_mem,
                        sizeof(unsigned char) * input_size[0]);

  // 将主机内存数据拷贝到设备内存
  bm_memcpy_s2d(handle_, input_mem, input_vir_addrs_init[0]);

  // 调用tpu_cv_threshold
  bm_status_t ret = tpu_cv_threshold(handle_, height, width, mode, threshold,
                                     max_value, &input_mem, &output_mem);

  if (ret != BM_SUCCESS) {
    printf("tpu_cv_threshold failed\n");
    // 释放设备内存
    bm_free_device(handle_, input_mem);
    bm_free_device(handle_, output_mem);
    return static_cast<int32_t>(ret);
  }

  // 将结果从设备内存拷贝回主机内存
  bm_memcpy_d2s(handle_, output_vir_addrs_init[0], output_mem);

  // 释放设备内存
  bm_free_device(handle_, input_mem);
  bm_free_device(handle_, output_mem);
  // 释放handle
  bm_dev_free(handle_);

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
  CVI_S32 lwidth = static_cast<CVI_S32>(left->getWidth());
  CVI_S32 lheight = static_cast<CVI_S32>(left->getHeight());
  CVI_S32 rwidth = static_cast<CVI_S32>(right->getWidth());
  CVI_S32 rheight = static_cast<CVI_S32>(right->getHeight());
  ImageFormat image_format = left->getImageFormat();
  CVI_S32 overlay_w = overlay_rx - overlay_lx + 1;
  CVI_S32 blend_w = lwidth + rwidth - overlay_w;
  CVI_S32 blend_h = lheight;
  CVI_S32 channel = (image_format == ImageFormat::GRAY) ? 1 : 3;

  // 创建输出图像
  output = ImageFactory::createImage(blend_w, blend_h, image_format,
                                     left->getPixDataType(), true,
                                     InferencePlatform::AUTOMATIC);

  // 将ImageFormat转换为PIXEL_FORMAT_E
  CVI_S32 format;
  ImageFormatToPixelFormat(image_format, format);

  // 获取虚拟地址
  std::vector<uint8_t *> left_vir_addrs_init = left->getVirtualAddress();
  std::vector<uint8_t *> right_vir_addrs_init = right->getVirtualAddress();
  std::vector<uint8_t *> output_vir_addrs_init = output->getVirtualAddress();

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

  // 创建设备内存
  bm_device_mem_t left_img_mem[3];
  bm_device_mem_t right_img_mem[3];
  bm_device_mem_t output_img_mem[3];

  // 分配设备内存并拷贝数据
  bm_handle_t handle;
  bm_dev_request(&handle, 0);
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

  // int wgt_size = overlay_w * blend_h;

  // // 分配权重数据内存
  // unsigned char *wgt =
  //     (unsigned char *)malloc(wgt_size * sizeof(unsigned char));

  // // 填充线性权重数据
  // for (int y = 0; y < blend_h; y++) {
  //   for (int x = 0; x < overlay_w; x++) {
  //     wgt[y * overlay_w + x] = 255 * (overlay_w - x) / overlay_w;
  //   }
  // }

  // 分配设备内存
  bm_device_mem_t wgt_mem[2];
  int wgt_size = overlay_w * blend_h;
  bm_malloc_device_byte(handle, &wgt_mem[0], wgt_size);
  bm_malloc_device_byte(handle, &wgt_mem[1], 0);
  bm_memcpy_s2d(handle, wgt_mem[0], wgt);
  // 调用tpu_2way_blending
  bm_status_t ret = tpu_2way_blending(
      handle, lwidth, lheight, rwidth, rheight, left_img_mem, right_img_mem,
      blend_w, blend_h, output_img_mem, overlay_lx, overlay_rx, wgt_mem,
      TPU_BLEND_WGT_MODE::WGT_YUV_SHARE, format, channel);

  // 将结果从设备内存拷贝回主机内存
  if (ret == BM_SUCCESS) {
    for (int c = 0; c < channel; c++) {
      bm_memcpy_d2s(handle, output_vir_addrs[c], output_img_mem[c]);
    }
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
  free(wgt);
  bm_dev_free(handle);

  return 0;
}

int32_t BmImageProcessor::compareResult(CVI_U8 *tpu_result, CVI_U8 *cpu_result,
                                        int size) {
  for (int i = 0; i < size; i++) {
    if (tpu_result[i] != cpu_result[i]) {
      printf("cpu and tpu result mismatch.\n");
      return -1;
    }
  }
  printf("cpu and tpu result same.\n");
  return 0;
}
