#include "cv/motion_detect/motion_detect.hpp"
#include "image/base_image.hpp"

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("Usage: %s <background_image> <detect_image>\n", argv[0]);
    return -1;
  }

  // 读取背景图像和检测图像
  std::shared_ptr<BaseImage> background_image =
      ImageFactory::readImage(argv[1], ImageFormat::GRAY);
  std::shared_ptr<BaseImage> detect_image =
      ImageFactory::readImage(argv[2], ImageFormat::GRAY);

  // 设置背景图像
  printf("set background image\n");
  std::shared_ptr<MotionDetection> md = MotionDetection::getMotionDetection();
  md->setBackground(background_image);

  // 设置ROI区域
  printf("set roi\n");
  std::vector<ObjectBoxInfo> roi;
  ObjectBoxInfo roi1;
  roi1.x1 = 0;
  roi1.y1 = 0;
  roi1.x2 = 512;
  roi1.y2 = 512;
  roi.push_back(roi1);
  ObjectBoxInfo roi2;
  roi2.x1 = 1000;
  roi2.y1 = 150;
  roi2.x2 = 1150;
  roi2.y2 = 250;
  roi.push_back(roi2);
  md->setROI(roi);

  // 执行运动检测
  printf("detect\n");
  std::vector<std::vector<float>> detected_objects;
  md->detect(detect_image, 20, 50, detected_objects);

  // 打印检测结果
  for (size_t i = 0; i < detected_objects.size(); i++) {
    printf("[%d,%d,%d,%d]\n", static_cast<int>(detected_objects[i][0]),
           static_cast<int>(detected_objects[i][1]),
           static_cast<int>(detected_objects[i][2]),
           static_cast<int>(detected_objects[i][3]));
  }

  printf("done\n");

  return 0;
}
