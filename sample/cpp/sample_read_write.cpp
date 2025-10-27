#include "tdl_model_factory.hpp"

int main(int argc, char** argv) {
  std::string image_path = "images/hg1.jpeg";
  std::shared_ptr<BaseImage> image;
  ImageFormat image_format;
  // BGR_PACKED
  printf("Read BGR_PACKED\n");
  image_format = ImageFormat::BGR_PACKED;
  image = ImageFactory::readImage(image_path, image_format);
  printf("Write BGR_PACKED\n");
  ImageFactory::writeImage("BGR_PACKED.jpg", image);

  // RGB_PACKED
  printf("Read RGB_PACKED\n");
  image_format = ImageFormat::RGB_PACKED;
  image = ImageFactory::readImage(image_path, image_format);
  printf("Write RGB_PACKED\n");
  ImageFactory::writeImage("RGB_PACKED.jpg", image);

  // GRAY
  printf("Read GRAY\n");
  image_format = ImageFormat::GRAY;
  image = ImageFactory::readImage(image_path, image_format);
  printf("Write GRAY\n");
  ImageFactory::writeImage("GRAY.jpg", image);

  return 0;
}
