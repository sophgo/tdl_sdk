#include <cvi_comm_vpss.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "image/base_image.hpp"
#include "ive/image_processor.hpp"

#define BM_ALIGN(x, a) (((x) + (a)-1) / (a) * (a))

// 读取NV12图像
void readNv12Image(const std::string& path, int width, int height,
                   std::shared_ptr<BaseImage>& image) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "无法打开文件: " << path << std::endl;
    return;
  }

  int y_size = width * height;
  int uv_size = width * BM_ALIGN(height / 2, 2);
  int total_size = y_size + uv_size;

  uint8_t* data = new uint8_t[total_size];
  file.read(reinterpret_cast<char*>(data), total_size);

  image = ImageFactory::createImage(width, height, ImageFormat::YUV420SP_UV,
                                    TDLDataType::UINT8, true,
                                    InferencePlatform::AUTOMATIC);

  int stride = image->getStrides()[0];
  uint8_t* dst = image->getVirtualAddress()[0];

  // 按行拷贝（处理stride对齐）
  for (int h = 0; h < height + BM_ALIGN(height / 2, 2); h++) {
    memcpy(dst + h * stride, data + h * width, width);
  }

  delete[] data;
}

// 保存NV12图像
void saveNv12Image(const std::string& path, std::shared_ptr<BaseImage>& image) {
  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "无法创建文件: " << path << std::endl;
    return;
  }

  int width = image->getWidth();
  int height = image->getHeight();
  int stride = image->getStrides()[0];

  int y_size = width * height;
  int uv_size = width * BM_ALIGN(height / 2, 2);
  int total_size = y_size + uv_size;

  uint8_t* src = image->getVirtualAddress()[0];
  uint8_t* data = new uint8_t[total_size];

  // 按行拷贝（去除stride对齐）
  for (int h = 0; h < height + BM_ALIGN(height / 2, 2); h++) {
    memcpy(data + h * width, src + h * stride, width);
  }

  file.write(reinterpret_cast<char*>(data), total_size);
  delete[] data;
}

// 读取权重文件
void readWeightFile(const std::string& path, unsigned char* wgt, int size) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "无法打开权重文件: " << path << std::endl;
    return;
  }
  file.read(reinterpret_cast<char*>(wgt), size);
}

// 创建权重图像（当 overlay_w > 0 时才创建）
std::shared_ptr<BaseImage> createWeightImage(const std::string& path,
                                             int overlay_w, int blend_h) {
  if (overlay_w <= 0) {
    // 当没有重叠区域时，创建一个最小的占位图像（不会被实际使用）
    std::shared_ptr<BaseImage> weight = ImageFactory::createImage(
        1, 1, ImageFormat::GRAY, TDLDataType::UINT8, true);
    return weight;
  }

  int wgt_y_size = overlay_w * blend_h;
  int wgt_uv_size = overlay_w * BM_ALIGN(blend_h / 2, 2);
  int wgt_total_size = wgt_y_size + wgt_uv_size;

  std::shared_ptr<BaseImage> weight =
      ImageFactory::createImage(overlay_w, blend_h + BM_ALIGN(blend_h / 2, 2),
                                ImageFormat::GRAY, TDLDataType::UINT8, true);

  unsigned char* wgt = weight->getVirtualAddress()[0];
  readWeightFile(path, wgt, wgt_total_size);

  return weight;
}

int main(int argc, char* argv[]) {
  // 参数检查
  if (argc != 17) {
    std::cerr << "用法: " << argv[0] << std::endl;
    std::cerr << "  <图0.nv12> <图0宽> <图0高>" << std::endl;
    std::cerr << "  <图1.nv12> <图1宽>" << std::endl;
    std::cerr << "  <图2.nv12> <图2宽>" << std::endl;
    std::cerr << "  <图3.nv12> <图3宽>" << std::endl;
    std::cerr << "  <重叠0宽度> <权重0.bin>" << std::endl;
    std::cerr << "  <重叠1宽度> <权重1.bin>" << std::endl;
    std::cerr << "  <重叠2宽度> <权重2.bin>" << std::endl;
    std::cerr << std::endl;
    std::cerr
        << "示例: " << argv[0]
        << " img0.nv12 960 1080 img1.nv12 960 img2.nv12 960 img3.nv12 960 "
        << "100 wgt0.bin 100 wgt1.bin 100 wgt2.bin" << std::endl;
    return -1;
  }

  // 解析参数
  std::string img0_path = argv[1];
  int img0_width = std::stoi(argv[2]);
  int img_height = std::stoi(argv[3]);

  std::string img1_path = argv[4];
  int img1_width = std::stoi(argv[5]);

  std::string img2_path = argv[6];
  int img2_width = std::stoi(argv[7]);

  std::string img3_path = argv[8];
  int img3_width = std::stoi(argv[9]);

  int overlay0 = std::stoi(argv[10]);
  std::string wgt0_path = argv[11];

  int overlay1 = std::stoi(argv[12]);
  std::string wgt1_path = argv[13];

  int overlay2 = std::stoi(argv[14]);
  std::string wgt2_path = argv[15];

  std::string output_path = argv[16];

  std::cout << "========== 4路NV12拼接测试 ==========" << std::endl;
  std::cout << "图0: " << img0_path << " (" << img0_width << "x" << img_height
            << ")" << std::endl;
  std::cout << "图1: " << img1_path << " (" << img1_width << "x" << img_height
            << ")" << std::endl;
  std::cout << "图2: " << img2_path << " (" << img2_width << "x" << img_height
            << ")" << std::endl;
  std::cout << "图3: " << img3_path << " (" << img3_width << "x" << img_height
            << ")" << std::endl;
  std::cout << "重叠0: " << overlay0 << ", 权重: " << wgt0_path << std::endl;
  std::cout << "重叠1: " << overlay1 << ", 权重: " << wgt1_path << std::endl;
  std::cout << "重叠2: " << overlay2 << ", 权重: " << wgt2_path << std::endl;

  // 读取4张输入图像
  std::shared_ptr<BaseImage> img0, img1, img2, img3;
  readNv12Image(img0_path, img0_width, img_height, img0);
  std::cout << "读取图0完成" << std::endl;

  readNv12Image(img1_path, img1_width, img_height, img1);
  std::cout << "读取图1完成" << std::endl;

  readNv12Image(img2_path, img2_width, img_height, img2);
  std::cout << "读取图2完成" << std::endl;

  readNv12Image(img3_path, img3_width, img_height, img3);
  std::cout << "读取图3完成" << std::endl;

  // 创建3个权重图像
  std::shared_ptr<BaseImage> wgt0 =
      createWeightImage(wgt0_path, overlay0, img_height);
  std::cout << "读取权重0完成" << std::endl;

  std::shared_ptr<BaseImage> wgt1 =
      createWeightImage(wgt1_path, overlay1, img_height);
  std::cout << "读取权重1完成" << std::endl;

  std::shared_ptr<BaseImage> wgt2 =
      createWeightImage(wgt2_path, overlay2, img_height);
  std::cout << "读取权重2完成" << std::endl;

  // 执行4路拼接
  std::shared_ptr<BaseImage> output_image;
  std::shared_ptr<ImageProcessor> processor =
      ImageProcessor::getImageProcessor();

  std::cout << "开始4路拼接..." << std::endl;
  int result =
      processor->fourWayBlending(img0, img1, img2, img3, wgt0, wgt1, wgt2,
                                 overlay0, overlay1, overlay2, output_image);

  if (result != 0) {
    std::cerr << "拼接失败! 错误码: " << result << std::endl;
    return -1;
  }

  std::cout << "拼接完成!" << std::endl;
  std::cout << "输出尺寸: " << output_image->getWidth() << "x"
            << output_image->getHeight() << std::endl;

  // 保存结果
  saveNv12Image(output_path, output_image);
  std::cout << "结果已保存至: " << output_path << std::endl;

  return 0;
}