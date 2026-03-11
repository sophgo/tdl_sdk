#include "cv/target_search/grabcut_segment.hpp"
#include <image/base_image.hpp>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

static float iou_xyxy(const cv::Rect& box1, const cv::Rect& box2,
                      float eps = 1e-12) {
  int x1a = min(box1.x, box1.x + box1.width);
  int y1a = min(box1.y, box1.y + box1.height);
  int x2a = max(box1.x, box1.x + box1.width);
  int y2a = max(box1.y, box1.y + box1.height);

  int x1b = min(box2.x, box2.x + box2.width);
  int y1b = min(box2.y, box2.y + box2.height);
  int x2b = max(box2.x, box2.x + box2.width);
  int y2b = max(box2.y, box2.y + box2.height);

  float area_a = (x2a - x1a) * (y2a - y1a);
  float area_b = (x2b - x1b) * (y2b - y1b);
  if (area_a <= 0 || area_b <= 0) return 0.0f;

  int inter_x1 = max(x1a, x1b);
  int inter_y1 = max(y1a, y1b);
  int inter_x2 = min(x2a, x2b);
  int inter_y2 = min(y2a, y2b);
  float inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1);
  if (inter_area <= 0) return 0.0f;

  return inter_area / (area_a + area_b - inter_area + eps);
}

GrabCutSegmentor::GrabCutSegmentor(const cvtdl_grabcut_params_t& params)
    : params_(params) {}

GrabCutSegmentor::GrabCutSegmentor() : params_() {}

void GrabCutSegmentor::setParams(const cvtdl_grabcut_params_t& params) {
  params_ = params;
}

int GrabCutSegmentor::segment(std::shared_ptr<BaseImage> image,
                              cv::Point seed_point,
                              cvtdl_grabcut_result_t* result) {
  // 输入检查
  if (!image || !result) {
    cerr << "Invalid input: image is null or result pointer is null" << endl;
    result->success = false;
    return -1;
  }

  int img_w = image->getWidth();
  int img_h = image->getHeight();
  auto strides = image->getStrides();
  auto virAddrs = image->getVirtualAddress();

  if (strides.empty() || virAddrs.empty()) {
    cerr << "Empty strides or virtual addresses" << endl;
    result->success = false;
    return -1;
  }

  int seed_x = seed_point.x;
  int seed_y = seed_point.y;
  if (seed_x < 0 || seed_y < 0 || seed_x >= img_w || seed_y >= img_h) {
    cerr << "Seed point out of bounds: (" << seed_x << "," << seed_y << ")"
         << endl;
    result->success = false;
    return -1;
  }

  // 计算裁剪区域
  float expand_ratio = params_.expand_h_ratio;
  int crop_y1 = max(0, seed_y - static_cast<int>(img_h * expand_ratio));
  int crop_y2 = min(img_h, seed_y + static_cast<int>(img_h * expand_ratio));
  int crop_x1 = max(0, seed_x - static_cast<int>(img_w * expand_ratio));
  int crop_x2 = min(img_w, seed_x + static_cast<int>(img_w * expand_ratio));

  preprocessor =
      PreprocessorFactory::createPreprocessor(InferencePlatform::CVITEK);

  std::shared_ptr<BaseImage> crop_img_ = preprocessor->cropResize(
      image, crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1,
      crop_x2 - crop_x1, crop_y2 - crop_y1, ImageFormat::BGR_PACKED);

  int crop_w = crop_img_->getWidth();
  int crop_h = crop_img_->getHeight();
  uint32_t stride = crop_img_->getStrides()[0];
  uint8_t* data_ptr = crop_img_->getVirtualAddress()[0];

  Mat crop_img(crop_h, crop_w, CV_8UC3, data_ptr, stride);

  // 种子点映射到裁剪区域
  int crop_seed_x = seed_x - crop_x1;
  int crop_seed_y = seed_y - crop_y1;
  if (crop_seed_x < 0 || crop_seed_x >= crop_w || crop_seed_y < 0 ||
      crop_seed_y >= crop_h) {
    cerr << "Mapped seed out of crop bounds" << endl;
    result->success = false;
    return -1;
  }
  Point crop_seed(crop_seed_x, crop_seed_y);

  // 多半径迭代
  const vector<int> seed_radii = {3, 5, 11, 13};
  bool found_valid = false;
  Rect crop_bbox;
  Mat crop_fg_mask, crop_result_mask;

  for (int radius : seed_radii) {
    Mat crop_mask = Mat::ones(crop_h, crop_w, CV_8UC1) * GC_PR_BGD;

    // 标记确定前景
    circle(crop_mask, crop_seed, radius, Scalar(GC_FGD), -1);

    // GrabCut迭代
    Mat bgdModel(1, 65, CV_64F);
    Mat fgdModel(1, 65, CV_64F);

    try {
      grabCut(crop_img, crop_mask, Rect(), bgdModel, fgdModel,
              params_.iter_count, GC_INIT_WITH_MASK);
    } catch (const cv::Exception& e) {
      // cerr << "GrabCut failed for radius " << radius << ": " << e.what()
      //      << endl;
      continue;  // 直接继续下一个半径的尝试
    }

    // 生成前景掩码
    Mat temp1, temp2;
    Mat current_fg_mask = Mat::zeros(crop_h, crop_w, CV_8UC1);
    compare(crop_mask, GC_FGD, temp1, CMP_EQ);
    compare(crop_mask, GC_PR_FGD, temp2, CMP_EQ);
    bitwise_or(temp1, temp2, current_fg_mask);

    // 连通域分析
    Mat labels, stats, centroids;
    int num_components = connectedComponentsWithStats(
        current_fg_mask, labels, stats, centroids, 8, CV_32S);
    int seed_label = -1;

    try {
      seed_label = labels.at<int>(crop_seed_y, crop_seed_x);
    } catch (const cv::Exception& e) {
      cerr << "Failed to get seed label: " << e.what() << endl;
      continue;
    }

    if (seed_label <= 0) {
      continue;
    }

    // IoU检查 - 使用0.6作为阈值
    Rect seed_box(
        max(0, crop_seed_x - radius), max(0, crop_seed_y - radius),
        min(crop_w, crop_seed_x + radius) - max(0, crop_seed_x - radius),
        min(crop_h, crop_seed_y + radius) - max(0, crop_seed_y - radius));
    int cbox_x = stats.at<int>(seed_label, CC_STAT_LEFT);
    int cbox_y = stats.at<int>(seed_label, CC_STAT_TOP);
    int cbox_w = stats.at<int>(seed_label, CC_STAT_WIDTH);
    int cbox_h = stats.at<int>(seed_label, CC_STAT_HEIGHT);
    crop_bbox = Rect(cbox_x, cbox_y, cbox_w, cbox_h);

    if (iou_xyxy(seed_box, crop_bbox) < 0.6) {
      crop_result_mask = crop_mask.clone();
      crop_fg_mask = current_fg_mask.clone();
      found_valid = true;
      break;
    }
  }

  // 处理迭代结果
  if (!found_valid) {
    cerr << "All radii failed to meet IoU threshold" << endl;
    result->success = false;
    return -1;
  }

  result->bbox = Rect(crop_bbox.x + crop_x1, crop_bbox.y + crop_y1,
                      crop_bbox.width, crop_bbox.height);

  result->fg_mask = Mat::zeros(img_h, img_w, CV_8UC1);
  crop_fg_mask.copyTo(result->fg_mask(Rect(crop_x1, crop_y1, crop_w, crop_h)));

  result->result_mask = Mat::ones(img_h, img_w, CV_8UC1) * GC_BGD;
  crop_result_mask.copyTo(
      result->result_mask(Rect(crop_x1, crop_y1, crop_w, crop_h)));

  result->success = true;

  return 0;
}
