#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "encoder/image_encoder/image_encoder.hpp"

#define BM_ALIGN(x, a) (((x) + (a)-1) / (a) * (a))

void readImageFromNv21(const std::string& nv21_image_path, int width,
                       int height, std::shared_ptr<BaseImage>& nv21_image) {
  std::ifstream file(nv21_image_path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << nv21_image_path << std::endl;
    return;
  }

  // 计算NV21数据大小
  size_t y_size = width * height;
  size_t uv_size = width * BM_ALIGN(height / 2, 2);
  size_t total_size = y_size + uv_size;

  // 分配并读取数据
  uint8_t* img_data = new uint8_t[total_size];
  file.read(reinterpret_cast<char*>(img_data), total_size);

  // 创建 BaseImage 对象
  nv21_image = ImageFactory::createImage(
      width, height, ImageFormat::YUV420SP_VU, TDLDataType::UINT8, true,
      InferencePlatform::AUTOMATIC);

  int stride = nv21_image->getStrides()[0];
  uint8_t* dst = (uint8_t*)nv21_image->getVirtualAddress()[0];

  // 拷贝 Y 和 UV 数据
  for (int h = 0; h < height + BM_ALIGN(height / 2, 2); ++h) {
    memcpy(dst + h * stride, img_data + h * width, width);
  }

  delete[] img_data;
}

int main(int argc, char** argv) {
  if (argc != 2 && argc != 4) {
    std::cerr << "Usage:\n"
              << "  For standard image file: " << argv[0] << " <image_path>\n"
              << "  For NV21 raw file:       " << argv[0]
              << " <nv21_file> <width> <height>\n";
    return -1;
  }

  std::shared_ptr<BaseImage> image;

  // 判断是否为 NV21 输入模式
  if (argc == 4) {
    int width = std::stoi(argv[2]);
    int height = std::stoi(argv[3]);
    readImageFromNv21(argv[1], width, height, image);
    if (!image) {
      std::cerr << "Failed to load NV21 image\n";
      return -1;
    }
  } else {
    // 使用 ImageFactory 加载标准图像文件
    image = ImageFactory::readImage(argv[1]);
    if (!image) {
      std::cerr << "Failed to load image from path: " << argv[1] << "\n";
      return -1;
    }
  }

  ImageEncoder encoder;
  std::vector<uint8_t> encoded_data;
  bool ret = encoder.encodeFrame(image, encoded_data);
  if (!ret) {
    std::cerr << "Image encoding failed.\n";
    return -1;
  }

  std::cout << "Image encoding succeeded, encoded size: " << encoded_data.size()
            << " bytes\n";

  std::string output_path = "./encoded_output.jpg";
  std::ofstream ofs(output_path, std::ios::binary);
  if (!ofs) {
    std::cerr << "Failed to open output file for writing.\n";
    return -1;
  }
  ofs.write(reinterpret_cast<const char*>(encoded_data.data()),
            encoded_data.size());
  ofs.close();

  std::cout << "Encoded image saved to: " << output_path << std::endl;

  return 0;
}