#include "cv/occlusion_detect/occlusion_detect.hpp"
#include <numeric>
#include <opencv2/opencv.hpp>
#include <vector>

OcclusionDetector::OcclusionDetector() = default;

int OcclusionDetector::detect(std::shared_ptr<BaseImage> image,
                              cvtdl_occlusion_meta_t* meta) {
  if (!image || !meta) {
    std::cerr << "Invalid input" << std::endl;
    return -1;
  }

  int width = image->getWidth();
  int height = image->getHeight();
  auto strides = image->getStrides();
  auto virAddrs = image->getVirtualAddress();

  if (strides.empty() || virAddrs.empty()) {
    std::cerr << "Empty strides or virtual addresses" << std::endl;
    return -1;
  }

  uint8_t* data_ptr = virAddrs[0];
  uint32_t stride = strides[0];

  cv::Mat frame(height, width, CV_8UC3, data_ptr, stride);

  int frame_h = frame.rows;
  int frame_w = frame.cols;
  cv::Rect roi(int(meta->crop_bbox.x1 * frame_w),
               int(meta->crop_bbox.y1 * frame_h),
               int((meta->crop_bbox.x2 - meta->crop_bbox.x1) * frame_w),
               int((meta->crop_bbox.y2 - meta->crop_bbox.y1) * frame_h));
  cv::Mat sub_frame = frame(roi);

  cv::Mat gray;
  cv::cvtColor(sub_frame, gray, cv::COLOR_BGR2GRAY);

  cv::Mat laplacian;
  cv::Laplacian(gray, laplacian, CV_64F);

  cv::Mat laplacianAbs;
  cv::threshold(laplacian, laplacianAbs, 0, 0, cv::THRESH_TRUNC);
  laplacianAbs = laplacian - 2 * laplacianAbs;

  cv::Mat laplacian_8u;
  cv::convertScaleAbs(laplacianAbs, laplacian_8u);

  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
  cv::Mat closed;
  cv::morphologyEx(laplacian_8u, closed, cv::MORPH_CLOSE, kernel);

  cv::Mat binary;
  cv::threshold(closed, binary, meta->laplacian_th, 255, cv::THRESH_BINARY);

  cv::Mat binary_not;
  cv::bitwise_not(binary, binary_not);
  cv::Mat labels;
  int num_labels = cv::connectedComponents(binary_not, labels, 4);

  std::vector<int> sizes(num_labels, 0);
  for (int i = 0; i < labels.rows; i++) {
    for (int j = 0; j < labels.cols; j++) {
      sizes[labels.at<int>(i, j)]++;
    }
  }
  int max_area =
      (num_labels > 1) ? *std::max_element(sizes.begin() + 1, sizes.end()) : 0;
  int total_pixels = labels.rows * labels.cols;
  float occ_ratio = static_cast<float>(max_area) / total_pixels;

  meta->occ_score = occ_ratio;
  meta->occ_class = (occ_ratio >= meta->occ_ratio_th) ? 1 : 0;

  occlusionStates_.push_back(meta->occ_class);
  if ((int)occlusionStates_.size() > meta->sensitive_th) {
    occlusionStates_.erase(occlusionStates_.begin());
    int occludedCount =
        std::accumulate(occlusionStates_.begin(), occlusionStates_.end(), 0);
    int pre_class = (occludedCount > meta->sensitive_th / 2) ? 1 : 0;
    meta->occ_class = pre_class;
  }

  return 0;
}

void OcclusionDetector::reset() { occlusionStates_.clear(); }
