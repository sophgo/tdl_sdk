#include <cvi_comm_vpss.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "image/base_image.hpp"
#include "ive/image_processor.hpp"

#define BM_ALIGN(x, a) (((x) + (a)-1) / (a) * (a))

static int32_t ImageFormatToPixelFormat(ImageFormat& image_format,
                                        PIXEL_FORMAT_E& format) {
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
    case ImageFormat::YUV420SP_UV:
      format = PIXEL_FORMAT_NV12;
      break;
    case ImageFormat::YUV420SP_VU:
      format = PIXEL_FORMAT_NV21;
      break;
    default:
      printf("ImageInfo format not supported: %d\n",
             static_cast<int>(image_format));
      return -1;
  }
  return 0;
}

int32_t BaseImage2VideoFrame(const std::shared_ptr<BaseImage>& image,
                             VIDEO_FRAME_INFO_S& video_frame) {
  if (!image) {
    printf("image is nullptr.\n");
    return -1;
  }

  // 基本图像信息
  video_frame.stVFrame.u32Width = image->getWidth();
  video_frame.stVFrame.u32Height = image->getHeight();

  ImageFormat base_fmt = image->getImageFormat();
  PIXEL_FORMAT_E format;
  ImageFormatToPixelFormat(base_fmt, format);
  video_frame.stVFrame.enPixelFormat = format;

  uint32_t plane_num = image->getPlaneNum();

  std::vector<uint32_t> strides = image->getStrides();
  std::vector<uint64_t> phy_addrs = image->getPhysicalAddress();
  std::vector<uint8_t*> vir_addrs = image->getVirtualAddress();

  for (uint32_t i = 0; i < plane_num; ++i) {
    video_frame.stVFrame.u32Stride[i] = strides[i];
    video_frame.stVFrame.u64PhyAddr[i] = phy_addrs[i];
    video_frame.stVFrame.pu8VirAddr[i] = vir_addrs[i];
    video_frame.stVFrame.u32Length[i] =
        video_frame.stVFrame.u32Height * strides[i];
  }

  return 0;
}

void printImageInfo(std::shared_ptr<BaseImage>& image) {
  // 打印图像的基本信息
  std::cout << "图像宽度: " << image->getWidth() << std::endl;
  std::cout << "图像高度: " << image->getHeight() << std::endl;
  std::cout << "图像格式: " << static_cast<int>(image->getImageFormat())
            << std::endl;
  std::cout << "图像数据类型: " << static_cast<int>(image->getPixDataType())
            << std::endl;
  std::cout << "图像数据大小: " << image->getImageByteSize() << std::endl;
  std::cout << "平面数量: " << image->getPlaneNum() << std::endl;
  std::cout << " strides[0]: " << image->getStrides()[0] << std::endl;
  std::cout << " strides[1]: " << image->getStrides()[1] << std::endl;
  std::cout << " strides[2]: " << image->getStrides()[2] << std::endl;
}

void readImageFromNv21(const std::string& nv21_image_path, int width,
                       int height, std::shared_ptr<BaseImage>& nv21_image) {
  std::ifstream file(nv21_image_path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << nv21_image_path << std::endl;
    return;
  }

  // 读取数据
  uint8_t* img_data =
      new uint8_t[width * height + width * BM_ALIGN(height / 2, 2)];
  file.read(reinterpret_cast<char*>(img_data),
            width * height + width * BM_ALIGN(height / 2, 2));

  nv21_image = ImageFactory::createImage(
      width, height, ImageFormat::YUV420SP_VU, TDLDataType::UINT8, true,
      InferencePlatform::AUTOMATIC);

  int stride = nv21_image->getStrides()[0];
  uint8_t* dst = (uint8_t*)nv21_image->getVirtualAddress()[0];

  // 逐行读取并拷贝Y分量
  for (int h = 0; h < height + BM_ALIGN(height / 2, 2); h++) {
    memcpy(dst + h * stride, img_data + h * width, width);
  }

  delete[] img_data;
}

void readFrameFromNv21(const std::string& nv21_image_path, int width,
                       int height, std::shared_ptr<BaseImage>& temp_image,
                       VIDEO_FRAME_INFO_S& video_frame) {
  readImageFromNv21(nv21_image_path, width, height, temp_image);
  BaseImage2VideoFrame(temp_image, video_frame);
}

void saveImageToNv21(const std::string& image_path,
                     std::shared_ptr<BaseImage>& image) {
  std::ofstream file(image_path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << image_path << std::endl;
  }
  int width = image->getWidth();
  int height = image->getHeight();
  int stride = image->getStrides()[0];

  uint8_t* src = (uint8_t*)image->getVirtualAddress()[0];
  uint8_t* dst = new uint8_t[width * height + width * BM_ALIGN(height / 2, 2)];
  // 逐行读取并拷贝Y分量
  for (int h = 0; h < height + BM_ALIGN(height / 2, 2); h++) {
    memcpy(dst + h * width, src + h * stride, width);
  }
  file.write(reinterpret_cast<char*>(dst),
             width * height + width * BM_ALIGN(height / 2, 2));
  delete[] dst;
}

int32_t processGrayImage(char* argv[]) {
  // 从命令行参数获取输入
  std::string image_path1 = argv[1];
  std::string image_path2 = argv[2];
  int overlay_w = std::stoi(argv[3]);

  // ============================读取灰度图像=============================
  std::cout << "读取图像1: " << image_path1 << std::endl;
  std::shared_ptr<BaseImage> left_image =
      ImageFactory::readImage(image_path1, ImageFormat::GRAY);

  std::cout << "读取图像2: " << image_path2 << std::endl;
  std::shared_ptr<BaseImage> right_image =
      ImageFactory::readImage(image_path2, ImageFormat::GRAY);

  std::cout << "左图尺寸: " << left_image->getWidth() << "x"
            << left_image->getHeight() << std::endl;
  std::cout << "右图尺寸: " << right_image->getWidth() << "x"
            << right_image->getHeight() << std::endl;

  // 创建输出图像
  std::shared_ptr<BaseImage> output_image;

  std::shared_ptr<ImageProcessor> processor =
      ImageProcessor::getImageProcessor();

  // 执行twoWayBlending操作
  int blend_h = left_image->getHeight();

  // 创建权重
  std::shared_ptr<BaseImage> weight =
      ImageFactory::createImage(overlay_w, blend_h + BM_ALIGN(blend_h / 2, 2),
                                ImageFormat::GRAY, TDLDataType::UINT8, true);

  // 获取权重图像的虚拟地址
  std::vector<uint8_t*> wgt_virtual_addresses = weight->getVirtualAddress();

  unsigned char* wgt = wgt_virtual_addresses[0];
  // 填充x线性权重数据
  for (int y = 0; y < blend_h + BM_ALIGN(blend_h / 2, 2); y++) {
    for (int x = 0; x < overlay_w; x++) {
      wgt[y * overlay_w + x] = 255 * (overlay_w - x) / overlay_w;
    }
  }

  int result =
      processor->twoWayBlending(left_image, right_image, weight, output_image);

  std::cout << "输出图像尺寸: " << output_image->getWidth() << "x"
            << output_image->getHeight() << std::endl;

  // 保存结果图像
  std::string output_path = "output_blending.jpg";
  ImageFactory::writeImage(output_path, output_image);
  std::cout << "输出图像已保存至: " << output_path << std::endl;
  return 0;
}

int32_t processYuvImage(int argc, char* argv[]) {
  // 从命令行参数获取输入
  std::string nv21_left_image_path = argv[1];
  int left_width = std::stoi(argv[2]);
  int left_height = std::stoi(argv[3]);
  std::string nv21_right_image_path = argv[4];
  int right_width = std::stoi(argv[5]);
  int right_height = std::stoi(argv[6]);
  int overlay_w = std::stoi(argv[7]);
  std::string weight_path = "";
  if (argc == 9) {
    weight_path = argv[8];
  }
  // ============================读取 yuv 图像=============================

  // // 方式一：将 yuv 文件读取成BaseImage
  // std::shared_ptr<BaseImage> left_image;
  // std::shared_ptr<BaseImage> right_image;
  // readImageFromNv21(nv21_left_image_path, left_width, left_height,
  // left_image);

  // readImageFromNv21(nv21_right_image_path, right_width, right_height,
  // right_image);

  // 方式二：
  /*
  - 将 yuv 文件读取成 VIDEO_FRAME_INFO_S，再 wrap成 BaseImage
  - 如果VIDEO_FRAME_INFO_S已经在内存中，可跳过
  */
  std::shared_ptr<BaseImage> temp_left_image;
  std::shared_ptr<BaseImage> temp_right_image;
  VIDEO_FRAME_INFO_S left_video_frame;
  VIDEO_FRAME_INFO_S right_video_frame;
  readFrameFromNv21(nv21_left_image_path, left_width, left_height,
                    temp_left_image, left_video_frame);
  readFrameFromNv21(nv21_right_image_path, right_width, right_height,
                    temp_right_image, right_video_frame);

  /*
  - 将 VIDEO_FRAME_INFO_S 转换为BaseImage
    如果是 VIDEO_FRAME_S 需要包装成 VIDEO_FRAME_INFO_S, 参考如下
    VIDEO_FRAME_S video_frame_s;
    VIDEO_FRAME_INFO_S video_frame_info;
    video_frame_info.stVFrame = video_frame_s;

  - 从 VIDEO_FRAME_INFO_S 中可以直接 VIDEO_FRAME_S, 参考如下
    VIDEO_FRAME_INFO_S video_frame_info;
    VIDEO_FRAME_S video_frame_s;
    video_frame_s = video_frame_info.stVFrame;
  */

  std::shared_ptr<BaseImage> left_image;
  std::shared_ptr<BaseImage> right_image;
  left_image = ImageFactory::wrapVPSSFrame(&left_video_frame, false);
  right_image = ImageFactory::wrapVPSSFrame(&right_video_frame, false);

  // 打印图像尺寸

  std::cout << "左图尺寸: " << left_image->getWidth() << "x"
            << left_image->getHeight() << std::endl;
  std::cout << "右图尺寸: " << right_image->getWidth() << "x"
            << right_image->getHeight() << std::endl;

  // ============================创建权重=============================
  // 创建权重
  int blend_h = left_image->getHeight();
  std::shared_ptr<BaseImage> weight =
      ImageFactory::createImage(overlay_w, blend_h + BM_ALIGN(blend_h / 2, 2),
                                ImageFormat::GRAY, TDLDataType::UINT8, true);

  std::vector<uint8_t*> wgt_virtual_addresses = weight->getVirtualAddress();
  unsigned char* wgt = wgt_virtual_addresses[0];
  if (weight_path == "") {
    // 填充x线性权重数据
    for (int y = 0; y < blend_h + BM_ALIGN(blend_h / 2, 2); y++) {
      for (int x = 0; x < overlay_w; x++) {
        wgt[y * overlay_w + x] = 255 * (overlay_w - x) / overlay_w;
      }
    }
  } else {
    // 从文件读取权重
    std::ifstream file(weight_path, std::ios::binary);
    file.read(reinterpret_cast<char*>(wgt),
              overlay_w * blend_h + overlay_w * BM_ALIGN(blend_h / 2, 2));
  }

  // ============================执行twoWayBlending=============================
  std::shared_ptr<BaseImage> output_image;
  std::shared_ptr<ImageProcessor> processor =
      ImageProcessor::getImageProcessor();

  int result =
      processor->twoWayBlending(left_image, right_image, weight, output_image);

  // 转成 VIDEO_FRAME_INFO_S
  VIDEO_FRAME_INFO_S output_video_frame;
  BaseImage2VideoFrame(output_image, output_video_frame);

  std::cout << "输出图像尺寸: " << output_image->getWidth() << "x"
            << output_image->getHeight() << std::endl;

  // ============================保存输出图像=============================
  saveImageToNv21("yuv_output.nv21", output_image);
  return 0;
}

int main(int argc, char* argv[]) {
  // 检查参数数量
  if (argc != 4 && argc != 8 && argc != 9) {
    std::cerr << "Usage: " << argv[0]
              << " <image_path1> <image_path2> <overlay_w>" << std::endl;
    std::cerr
        << "Usage: " << argv[0]
        << " <nv21_left_image_path> <width> <height> <nv21_right_image_path> "
           "<width> <height> <overlay_w> [<weight_path>]"
        << std::endl;
    return -1;
  }
  if (argc == 4) {
    processGrayImage(argv);
  } else if (argc == 8 || argc == 9) {
    processYuvImage(argc, argv);
  }
  return 0;
}