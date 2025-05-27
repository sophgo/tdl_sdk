#include <unistd.h>
#include <iostream>
#include <memory>
#include <string>

#include "image/base_image.hpp"
#include "ive/image_processor.hpp"

#define BM_ALIGN(x, a) (((x) + (a)-1) / (a) * (a))

int main(int argc, char* argv[]) {
  // 检查参数数量
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " <image_path1> <image_path2> <overlay_w>" << std::endl;
    return -1;
  }

  // 从命令行参数获取输入
  std::string image_path1 = argv[1];
  std::string image_path2 = argv[2];
  int overlay_w = std::stoi(argv[3]);

  // 读取图像
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

// ./sample_img_blend image_path1 image_path2 overlay_w