#include <unistd.h>
#include <iostream>
#include <memory>
#include <string>

#include "image/base_image.hpp"
#include "ive/image_processor.hpp"

int main(int argc, char* argv[]) {
  // 检查参数数量
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << " <image_path1> <image_path2> <overlay_lx> <overlay_rx>"
              << std::endl;
    return -1;
  }

  // 从命令行参数获取输入
  std::string image_path1 = argv[1];
  std::string image_path2 = argv[2];
  int overlay_lx = std::stoi(argv[3]);
  int overlay_rx = std::stoi(argv[4]);

  // 读取图像
  std::cout << "读取图像1: " << image_path1 << std::endl;
  std::shared_ptr<BaseImage> left_image =
      ImageFactory::readImage(image_path1, ImageFormat::GRAY);
  std::string output_path1 = "input_blend1.jpg";
  ImageFactory::writeImage(output_path1, left_image);
  std::cout << "输入图像1已保存至: " << output_path1 << std::endl;

  std::cout << "读取图像2: " << image_path2 << std::endl;
  std::shared_ptr<BaseImage> right_image =
      ImageFactory::readImage(image_path2, ImageFormat::GRAY);
  std::string output_path2 = "input_blend2.jpg";
  ImageFactory::writeImage(output_path2, right_image);
  std::cout << "输入图像2已保存至: " << output_path2 << std::endl;

  std::cout << "左图尺寸: " << left_image->getWidth() << "x"
            << left_image->getHeight() << std::endl;
  std::cout << "右图尺寸: " << right_image->getWidth() << "x"
            << right_image->getHeight() << std::endl;

  // 创建输出图像
  std::shared_ptr<BaseImage> output_image;

  std::shared_ptr<ImageProcessor> processor =
      ImageProcessor::getImageProcessor("bm");

  // 执行twoWayBlending操作
  int overlay_w = overlay_rx - overlay_lx + 1;
  int blend_h = left_image->getHeight();
  int wgt_size = overlay_w * blend_h;

  // 分配权重数据内存
  CVI_U8* wgt = (unsigned char*)malloc(wgt_size * sizeof(unsigned char));

  // 填充线性权重数据
  for (int y = 0; y < blend_h; y++) {
    for (int x = 0; x < overlay_w; x++) {
      wgt[y * overlay_w + x] = 255 * (overlay_w - x) / overlay_w;
    }
  }
  int result = processor->twoWayBlending(left_image, right_image, output_image,
                                         overlay_lx, overlay_rx, wgt);

  std::cout << "输出图像尺寸: " << output_image->getWidth() << "x"
            << output_image->getHeight() << std::endl;

  // 保存结果图像
  std::string output_path = "output_blending.jpg";
  ImageFactory::writeImage(output_path, output_image);
  std::cout << "输出图像已保存至: " << output_path << std::endl;
  return 0;
}

// ./sample_img_blend image_path1 image_path2 overlay_lx overlay_rx