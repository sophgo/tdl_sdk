#include <inttypes.h>
#include <sys/time.h>
#include <iostream>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>

#include "bm_motion_detect.hpp"
#include "common/ccl.hpp"
#include "utils/tdl_log.hpp"

BmMotionDetection::BmMotionDetection()
    : ccl_instance_(nullptr), use_roi_(false), im_width_(0), im_height_(0) {
  // 创建连通域标记实例
  ccl_instance_ = createConnectInstance();

  // 创建图像处理器
  image_processor_ = ImageProcessor::getImageProcessor();
  if (!image_processor_) {
    LOGE("Failed to create image processor\n");
  }
}

BmMotionDetection::~BmMotionDetection() {
  // 释放连通域标记实例
  destroyConnectedComponent(ccl_instance_);
}

int32_t BmMotionDetection::setBackground(
    const std::shared_ptr<BaseImage> &background_image) {
  if (!background_image) {
    LOGE("Background image is null\n");
    return -1;
  }

  // 只支持单通道灰度图像
  if (background_image->getImageFormat() != ImageFormat::GRAY) {
    LOGE("Background image format should be GRAY, got %d\n",
         static_cast<int>(background_image->getImageFormat()));
    return -1;
  }

  // 记录图像尺寸
  im_width_ = background_image->getWidth();
  im_height_ = background_image->getHeight();

  // 设置背景图像
  background_image_ = background_image;

  return 0;
}

int32_t BmMotionDetection::setROI(const std::vector<ObjectBoxInfo> &_roi_s) {
  if (_roi_s.empty()) {
    LOGE("ROI list is empty\n");
    return -1;
  }

  // 确保背景图像已设置
  if (!background_image_) {
    LOGE("Background image is not set\n");
    return -1;
  }

  int imw = background_image_->getWidth();
  int imh = background_image_->getHeight();

  // 验证ROI区域有效性
  for (size_t i = 0; i < _roi_s.size(); i++) {
    auto p = _roi_s[i];
    if (p.x2 < p.x1 || p.x1 < 0 || p.x2 >= imw) {
      LOGE("roi[%zu] x overflow, x1:%d, x2:%d, imgw:%d\n", i, p.x1, p.x2, imw);
      use_roi_ = false;
      return -1;
    }
    if (p.y2 < p.y1 || p.y1 < 0 || p.y2 >= imh) {
      LOGE("roi[%zu] y overflow, y1:%d, y2:%d, imgh:%d\n", i, p.y1, p.y2, imh);
      use_roi_ = false;
      return -1;
    }
  }

  // 设置ROI区域
  roi_s_ = _roi_s;
  use_roi_ = true;

  return 0;
}

// 实现运动检测功能
int32_t BmMotionDetection::detect(const std::shared_ptr<BaseImage> &image,
                                  uint8_t threshold, double min_area,
                                  std::vector<std::vector<float>> &objs) {
  if (!image) {
    LOGE("Input image is null\n");
    return -1;
  }

  if (!background_image_) {
    LOGE("Background image is not set\n");
    return -1;
  }

  if (image->getHeight() != background_image_->getHeight() ||
      image->getWidth() != background_image_->getWidth()) {
    LOGE(
        "Height and width of image isn't equal to background image in "
        "BmMotionDetection\n");
    return -1;
  }

  if (image->getImageFormat() != ImageFormat::GRAY) {
    LOGE("Processed image format should be GRAY, got %d\n",
         static_cast<int>(image->getImageFormat()));
    return -1;
  }

  if (!image_processor_) {
    LOGE("Image processor is not initialized\n");
    return -1;
  }

  md_timer_.TicToc("start");
  int32_t ret = 0;

  if (!md_output_) {
    md_output_ =
        ImageFactory::createImage(image->getWidth(), image->getHeight(),
                                  ImageFormat::GRAY, TDLDataType::UINT8, true);
  }

  // 创建非const的共享指针用于传递给image_processor_
  std::shared_ptr<BaseImage> img_input =
      std::const_pointer_cast<BaseImage>(image);

  // 使用subads进行图像差分
  md_timer_.TicToc("preprocess");
  ret = image_processor_->subads(img_input, background_image_, md_output_);
  if (ret != 0) {
    LOGE("Failed to perform subads, ret=%d\n", ret);
    return -1;
  }

  // 阈值处理
  ret = image_processor_->thresholdProcess(md_output_,
                                           0,                 // 二值化类型
                                           threshold,         // 阈值
                                           255, md_output_);  // 最大值

  md_timer_.TicToc("image_process");

  if (ret != 0) {
    LOGE("Failed to perform threshold processing, ret=%d\n", ret);
    return -1;
  }

  // 执行腐蚀和膨胀操作
  md_output_->invalidateCache();
  std::vector<uint8_t *> virtual_addresses = md_output_->getVirtualAddress();
  uint8_t *ptr_src = virtual_addresses[0];
  std::vector<uint32_t> strides = md_output_->getStrides();
  cv::Mat img(md_output_->getHeight(), md_output_->getWidth(), CV_8UC1, ptr_src,
              strides[0]);

  // 创建5x5的矩形结构元素
  cv::Mat kernel = cv::Mat::zeros(5, 5, CV_8U);
  for (int i = 0; i < 5; i++) {
    kernel.at<uchar>(2, i) = 1;  // 中间行全1
    kernel.at<uchar>(i, 2) = 1;  // 中间列全1
  }

  // 腐蚀操作
  cv::erode(img, img, kernel);
  // 膨胀操作
  cv::dilate(img, img, kernel);
  // 获取图像信息
  int wstride = img.step[0];
  int num_boxes = 0;
  int *p_boxes = nullptr;

  // 提取连通域
  if (use_roi_) {
    int offsetx = 0, offsety = 0, offset = 0;
    int imw = im_width_;
    int imh = im_height_;
    objs.clear();

    for (uint8_t i = 0; i < roi_s_.size(); i++) {
      auto pnt = roi_s_[i];
      offsetx = pnt.x1;
      offsety = pnt.y1;
      offset = pnt.y1 * wstride + pnt.x1;
      imw = pnt.x2 - pnt.x1;
      imh = pnt.y2 - pnt.y1;

      // 使用连通域标记提取区域
      uint8_t *vir_addr = img.data;
      p_boxes = extractConnectedComponent(vir_addr + offset, imw, imh, wstride,
                                          min_area, ccl_instance_, &num_boxes);

      // 添加检测到的区域
      for (uint32_t j = 0; j < (uint32_t)num_boxes; ++j) {
        std::vector<float> box;
        box.push_back(p_boxes[j * 5 + 2] + offsetx);  // x1
        box.push_back(p_boxes[j * 5 + 1] + offsety);  // y1
        box.push_back(p_boxes[j * 5 + 4] + offsetx);  // x2
        box.push_back(p_boxes[j * 5 + 3] + offsety);  // y2
        objs.push_back(box);
      }
    }
  } else {
    // 对整个图像进行处理
    uint8_t *vir_addr = img.data;
    p_boxes =
        extractConnectedComponent(vir_addr, im_width_, im_height_, wstride,
                                  min_area, ccl_instance_, &num_boxes);

    objs.clear();
    for (uint32_t i = 0; i < (uint32_t)num_boxes; ++i) {
      std::vector<float> box;
      box.push_back(p_boxes[i * 5 + 2]);  // x1
      box.push_back(p_boxes[i * 5 + 1]);  // y1
      box.push_back(p_boxes[i * 5 + 4]);  // x2
      box.push_back(p_boxes[i * 5 + 3]);  // y2
      objs.push_back(box);
    }
  }

  md_timer_.TicToc("post");
  return 0;
}

bool BmMotionDetection::isROIEmpty() { return roi_s_.empty(); }
