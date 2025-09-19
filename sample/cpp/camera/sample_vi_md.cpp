#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>
#include <atomic>

#include "cv/motion_detect/motion_detect.hpp"
#include "image/base_image.hpp"
#include "video_decoder/video_decoder_type.hpp"

int main(int argc, char* argv[]) {
  int vi_chn = 0;
  // 创建视频解码器
  std::shared_ptr<VideoDecoder> vi_decoder =
      VideoDecoderFactory::createVideoDecoder(VideoDecoderType::VI);
  vi_decoder->initialize(960, 540, ImageFormat::YUV420SP_VU, 3);
  // 创建运动检测实例
  std::shared_ptr<MotionDetection> md = MotionDetection::getMotionDetection();

  // 设置终端为非规范模式
  struct termios oldt, newt;
  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);

  printf("按任意键退出...\n");

  std::shared_ptr<BaseImage> image;
  std::shared_ptr<BaseImage> gray_frame;
  // std::shared_ptr<BaseImage>* foreground_frame;
  uint8_t* src;
  uint8_t* dst;
  uint32_t src_stride;
  uint32_t dst_stride;
  uint32_t height;
  uint32_t width;
  bool is_have_background = false;
  std::vector<ObjectBoxInfo> detected_objects;

  while (1) {
    // 检查键盘输入
    fd_set rfds;
    struct timeval tv = {0, 0};
    FD_ZERO(&rfds);
    FD_SET(STDIN_FILENO, &rfds);
    int key_pressed = select(STDIN_FILENO + 1, &rfds, NULL, NULL, &tv);
    if (key_pressed > 0 && FD_ISSET(STDIN_FILENO, &rfds)) {
      break;  // 有键盘输入，退出循环
    }
    vi_decoder->read(image, vi_chn);
    if (!gray_frame || gray_frame->getWidth() != image->getWidth() ||
        gray_frame->getHeight() != image->getHeight()) {
      gray_frame = ImageFactory::createImage(
          image->getWidth(), image->getHeight(), ImageFormat::GRAY,
          TDLDataType::UINT8, true, InferencePlatform::AUTOMATIC);
    }

    // 复制Y分量数据（灰度信息）
    src = image->getVirtualAddress()[0];  // Y分量
    dst = gray_frame->getVirtualAddress()[0];
    src_stride = image->getStrides()[0];
    dst_stride = gray_frame->getStrides()[0];
    height = image->getHeight();
    width = image->getWidth();

    // 逐行复制Y分量数据
    for (uint32_t i = 0; i < height; i++) {
      memcpy(dst + i * dst_stride, src + i * src_stride, width);
    }
    gray_frame->flushCache();  // Flush cache after memcpy

    if (!is_have_background) {
      is_have_background = true;
      md->setBackground(gray_frame);
      gray_frame.reset();
      vi_decoder->release(vi_chn);
      continue;
    }
    md->detect(gray_frame, 20, 50, detected_objects);
    md->setBackground(gray_frame);
    gray_frame.reset();
    // 打印检测结果
    if (detected_objects.size() == 0) {
      printf("移动物体数量: %zu\n", detected_objects.size());
    } else {
      printf("移动物体数量: %zu, 移动物体坐标: \n", detected_objects.size());
      for (size_t i = 0; i < detected_objects.size(); i++) {
        printf("[%d,%d,%d,%d]\n", static_cast<int>(detected_objects[i].x1),
               static_cast<int>(detected_objects[i].y1),
               static_cast<int>(detected_objects[i].x2),
               static_cast<int>(detected_objects[i].y2));
      }
    }
    vi_decoder->release(vi_chn);
    image.reset();

    usleep(40 * 1000);  // Match Vi Frame Rate
  }

  // 恢复终端设置
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  printf("程序退出\n");
  return 0;
}