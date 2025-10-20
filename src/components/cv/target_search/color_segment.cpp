#include "cv/target_search/color_segment.hpp"
#include <image/base_image.hpp>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>

#include "utils/tdl_log.hpp"
using namespace cv;
using namespace std;

ColorSegmentor::ColorSegmentor() : params_(cvtdl_color_params_t()) {}

ColorSegmentor::ColorSegmentor(const cvtdl_color_params_t& params)
    : params_(params) {}

void ColorSegmentor::setParams(const cvtdl_color_params_t& params) {
  params_ = params;
}

bool ColorSegmentor::in_center(const cv::Rect& bbox, const cv::Point& seed) {
  int center_x_min = bbox.x + static_cast<int>(bbox.width * 0.1);
  int center_x_max = bbox.x + static_cast<int>(bbox.width * 0.9);
  int center_y_min = bbox.y + static_cast<int>(bbox.height * 0.1);
  int center_y_max = bbox.y + static_cast<int>(bbox.height * 0.9);

  return (seed.x >= center_x_min && seed.x <= center_x_max &&
          seed.y >= center_y_min && seed.y <= center_y_max);
}

int ColorSegmentor::segment(std::shared_ptr<BaseImage> image,
                            cv::Point seed_point,
                            cvtdl_color_result_t* result) {
  // 输入检查
  if (!image || !result) {
    cerr << "Invalid input: image is null or result pointer is null" << endl;
    result->success = false;
    return -1;
  }

  int sx = seed_point.x;
  int sy = seed_point.y;

  int img_w = image->getWidth();
  int img_h = image->getHeight();

  int crop_x1 =
      std::max(0, sx - static_cast<int>(img_w * params_.expand_h_ratio));
  int crop_x2 =
      std::min(img_w, sx + static_cast<int>(img_w * params_.expand_h_ratio));
  int crop_y1 =
      std::max(0, sy - static_cast<int>(img_h * params_.expand_h_ratio));
  int crop_y2 =
      std::min(img_h, sy + static_cast<int>(img_h * params_.expand_h_ratio));

  preprocessor =
      PreprocessorFactory::createPreprocessor(InferencePlatform::AUTOMATIC);

  std::shared_ptr<BaseImage> crop_img_ = preprocessor->cropResize(
      image, crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1,
      crop_x2 - crop_x1, crop_y2 - crop_y1, ImageFormat::BGR_PACKED);

  int crop_w = crop_img_->getWidth();
  int crop_h = crop_img_->getHeight();
  uint32_t stride = crop_img_->getStrides()[0];
  uint8_t* data_ptr = crop_img_->getVirtualAddress()[0];

  Mat crop_img(crop_h, crop_w, CV_8UC3, data_ptr, stride);

  Rect crop_rect(crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1);

  if (crop_h <= 0 || crop_w <= 0) {
    printf("[DEBUG] Invalid crop size, returning -2\n");
    return -2;
  }

  int roi_x0_org = std::max(0, sx - params_.patch_radius);
  int roi_y0_org = std::max(0, sy - params_.patch_radius);
  int roi_x1_org = std::min(img_w, sx + params_.patch_radius + 1);
  int roi_y1_org = std::min(img_h, sy + params_.patch_radius + 1);

  int roi_x0_crop = roi_x0_org - crop_x1;
  int roi_y0_crop = roi_y0_org - crop_y1;
  int roi_x1_crop = roi_x1_org - crop_x1;
  int roi_y1_crop = roi_y1_org - crop_y1;

  cv::Rect roi_rect(std::max(0, roi_x0_crop), std::max(0, roi_y0_crop),
                    std::min(crop_w, roi_x1_crop) - std::max(0, roi_x0_crop),
                    std::min(crop_h, roi_y1_crop) - std::max(0, roi_y0_crop));

  cv::Mat roi = crop_img(roi_rect);

  cv::Scalar mean = cv::mean(roi);
  float mean_b = mean[0];
  float mean_g = mean[1];
  float mean_r = mean[2];

  cv::Rect valid_bbox;
  cv::Mat valid_mask;
  bool found = false;

  for (int diff : params_.diff_list) {
    cv::Scalar lower(mean_b - diff, mean_g - diff, mean_r - diff);
    cv::Scalar upper(mean_b + diff, mean_g + diff, mean_r + diff);

    cv::Mat crop_mask;
    cv::inRange(crop_img, lower, upper, crop_mask);

    // 统计掩码中非零像素数量
    int non_zero_count = cv::countNonZero(crop_mask);

    cv::Mat labels, stats, centroids;
    int num_components = cv::connectedComponentsWithStats(
        crop_mask, labels, stats, centroids, 8, CV_32S);

    int crop_sx = sx - crop_x1;
    int crop_sy = sy - crop_y1;

    if (crop_sx < 0 || crop_sx >= crop_w || crop_sy < 0 || crop_sy >= crop_h) {
      printf("[DEBUG] Seed point out of crop bounds, skipping\n");
      continue;
    }

    int seed_label = labels.at<int>(crop_sy, crop_sx);

    if (seed_label == 0) {
      printf("[DEBUG] Seed point is background, skipping\n");
      continue;
    }

    int c_x = stats.at<int>(seed_label, 0);
    int c_y = stats.at<int>(seed_label, 1);
    int c_w = stats.at<int>(seed_label, 2);
    int c_h = stats.at<int>(seed_label, 3);
    int c_area = stats.at<int>(seed_label, 4);

    if (c_area < params_.min_area) {
      continue;
    }

    cv::Rect crop_bbox(c_x, c_y, c_w, c_h);
    bool center_check = in_center(crop_bbox, cv::Point(crop_sx, crop_sy));

    if (!center_check) continue;

    valid_bbox = cv::Rect(c_x + crop_x1, c_y + crop_y1, c_w, c_h);

    valid_mask = cv::Mat::zeros(img_h, img_w, CV_8UC1);
    crop_mask.copyTo(valid_mask(crop_rect));

    found = true;
    break;
  }

  if (found) {
    result->bbox = valid_bbox;
    result->fg_mask = valid_mask.clone();
    result->result_mask = valid_mask.clone();
    result->success = true;
    return 0;
  }
  printf("[DEBUG] No valid target found, returning -3\n");
  return -1;
}
