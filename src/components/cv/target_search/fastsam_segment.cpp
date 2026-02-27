#include "cv/target_search/fastsam_segment.hpp"
#include <algorithm>
#include <limits>
#include <opencv2/opencv.hpp>
#include "utils/tdl_log.hpp"

static int findSmallestBboxContainingPoint(
    std::shared_ptr<ModelBoxSegmentationInfo> obj_meta, uint32_t image_height,
    uint32_t image_width, cv::Point point, cv::Rect* out_bbox) {
  if (!obj_meta || !out_bbox || obj_meta->box_seg.empty()) return -1;

  const int proto_h = static_cast<int>(obj_meta->mask_height);
  const int proto_w = static_cast<int>(obj_meta->mask_width);

  int smallest_bbox_index = -1;
  float min_area = std::numeric_limits<float>::max();

  for (uint32_t i = 0; i < obj_meta->box_seg.size(); ++i) {
    const auto& seg = obj_meta->box_seg[i];
    float x1 = seg.x1, y1 = seg.y1, x2 = seg.x2, y2 = seg.y2;
    if (x2 <= x1 || y2 <= y1) continue;
    if (point.x >= x1 && point.x <= x2 && point.y >= y1 && point.y <= y2) {
      float area = (x2 - x1) * (y2 - y1);
      if (area < min_area) {
        min_area = area;
        smallest_bbox_index = static_cast<int>(i);
      }
    }
  }

  if (smallest_bbox_index < 0) return -1;

  const auto& seg = obj_meta->box_seg[smallest_bbox_index];
  out_bbox->x = static_cast<int>(seg.x1);
  out_bbox->y = static_cast<int>(seg.y1);
  out_bbox->width = static_cast<int>(seg.x2 - seg.x1);
  out_bbox->height = static_cast<int>(seg.y2 - seg.y1);
  return 0;
}

// 默认以 seed 为中心抠图尺寸，与 FastSAM 模型输入一致
static const int kDefaultCropSize = 320;

FastSAMSegmentor::FastSAMSegmentor(const std::string& model_path)
    : model_path_(model_path) {
  if (model_path_.empty()) {
    LOGE("FastSAMSegmentor: model_path is empty");
    return;
  }

  TDLModelFactory& model_factory = TDLModelFactory::getInstance();
  model_od_ = model_factory.getModel(ModelType::FASTSAM_SEG, model_path_);
  if (!model_od_) {
    LOGE("FastSAMSegmentor: get model failed");
    return;
  }
}

int FastSAMSegmentor::segment(std::shared_ptr<BaseImage> image,
                              cv::Point seed_point,
                              cvtdl_fastsam_result_t* result) {
  if (!image || !result) {
    result->success = false;
    return -1;
  }

  if (!model_od_) {
    result->success = false;
    return -1;
  }

  const int img_w = static_cast<int>(image->getWidth());
  const int img_h = static_cast<int>(image->getHeight());

  // 以 seed_point 为中心抠 320x320，边界处截断到图像内
  int crop_x1 = seed_point.x - kDefaultCropSize / 2;
  int crop_y1 = seed_point.y - kDefaultCropSize / 2;
  crop_x1 = std::max(0, crop_x1);
  crop_y1 = std::max(0, crop_y1);
  int crop_x2 = std::min(img_w, crop_x1 + kDefaultCropSize);
  int crop_y2 = std::min(img_h, crop_y1 + kDefaultCropSize);
  int crop_w = crop_x2 - crop_x1;
  int crop_h = crop_y2 - crop_y1;
  if (crop_w <= 0 || crop_h <= 0) {
    result->success = false;
    return -1;
  }

  std::shared_ptr<BasePreprocessor> preprocessor = model_od_->getPreprocessor();
  if (!preprocessor) {
    result->success = false;
    return -1;
  }
  std::shared_ptr<BaseImage> crop_image =
      preprocessor->crop(image, crop_x1, crop_y1, crop_w, crop_h);
  if (!crop_image) {
    result->success = false;
    return -1;
  }

  std::vector<std::shared_ptr<BaseImage>> input_images = {crop_image};
  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  model_od_->inference(input_images, out_datas);

  if (out_datas.empty()) {
    result->success = false;
    return -1;
  }

  std::shared_ptr<ModelBoxSegmentationInfo> obj_meta =
      std::static_pointer_cast<ModelBoxSegmentationInfo>(out_datas[0]);
  if (!obj_meta || obj_meta->box_seg.empty()) {
    result->success = false;
    return -1;
  }

  // 模型输出为模型输入空间（一般为 320x320），将 seed 映射到该空间
  const uint32_t model_h = obj_meta->image_height;
  const uint32_t model_w = obj_meta->image_width;
  int seed_in_model_x = static_cast<int>(
      (seed_point.x - crop_x1) * static_cast<float>(model_w) / crop_w + 0.5f);
  int seed_in_model_y = static_cast<int>(
      (seed_point.y - crop_y1) * static_cast<float>(model_h) / crop_h + 0.5f);
  seed_in_model_x =
      std::max(0, std::min(seed_in_model_x, static_cast<int>(model_w) - 1));
  seed_in_model_y =
      std::max(0, std::min(seed_in_model_y, static_cast<int>(model_h) - 1));

  cv::Rect bbox_model;
  int ret = findSmallestBboxContainingPoint(
      obj_meta, model_h, model_w, cv::Point(seed_in_model_x, seed_in_model_y),
      &bbox_model);
  if (ret != 0) {
    result->success = false;
    return -1;
  }

  // 将 bbox 从模型输入空间映射回原图坐标
  result->bbox.x =
      static_cast<int>(bbox_model.x * static_cast<float>(crop_w) / model_w +
                       0.5f) +
      crop_x1;
  result->bbox.y =
      static_cast<int>(bbox_model.y * static_cast<float>(crop_h) / model_h +
                       0.5f) +
      crop_y1;
  result->bbox.width = static_cast<int>(
      bbox_model.width * static_cast<float>(crop_w) / model_w + 0.5f);
  result->bbox.height = static_cast<int>(
      bbox_model.height * static_cast<float>(crop_h) / model_h + 0.5f);
  result->success = true;
  return 0;
}
