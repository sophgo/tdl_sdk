#include "sot.hpp"
#include "utils/mot_box_helper.hpp"
#include "utils/tdl_log.hpp"

SOT::SOT() {
  preprocessor_ =
      PreprocessorFactory::createPreprocessor(InferencePlatform::AUTOMATIC);
}

SOT::~SOT() {}

void SOT::getStatus(const std::vector<float>& bbox,
                    const std::vector<float>& kalman_bbox, float score,
                    float score_ratio, float iou, float size_ratio) {
  float size_ratio_abs = std::abs(size_ratio - size_ratio_threshold_);
  // 实际得分与丢失得分阈值的差值
  float score_abs = std::max(occluded_score_threshold_ - score, 0.0f);
  // 实际得分比率与丢失得分比率阈值的差值
  float score_ratio_abs =
      std::max(occluded_score_ratio_threshold_ - score_ratio, 0.0f);
  // 实际IoU与丢失IoU阈值的差值
  float iou_abs = std::max(occluded_iou_threshold_ - iou, 0.0f);
  // 实际得分与重现得分阈值的差值
  float reappear_score_abs = std::max(score - reappear_score_threshold_, 0.0f);
  // 实际得分比率与重现得分比率阈值的差值
  float reappear_score_ratio_abs =
      std::max(score_ratio - reappear_score_ratio_threshold_, 0.0f);
  // 实际IoU与重现IoU阈值的差值
  float reappear_iou_abs = iou - reappear_iou_threshold_;
  float confidence_of_occluded = 0.1 * score_abs + 0.7 * score_ratio_abs +
                                 0.5 * iou_abs + 0.5 * size_ratio_abs;
  float confidence_of_reappear = 2 * reappear_score_abs +
                                 2 * reappear_score_ratio_abs +
                                 0.1 * reappear_iou_abs - 0.2 * size_ratio_abs;
  sot_info_.is_occluded = confidence_of_occluded > occluded_threshold_;
  sot_info_.is_reappear = confidence_of_reappear > reappear_threshold_;
}

void SOT::ensureBBoxBoundaries(std::vector<float>& bbox,
                               const std::shared_ptr<BaseImage>& image) {
  float img_w = image->getWidth();
  float img_h = image->getHeight();
  float x1 = bbox[0];
  float y1 = bbox[1];
  float x2 = bbox[0] + bbox[2];
  float y2 = bbox[1] + bbox[3];
  x1 = std::min(std::max(0.0f, x1), img_w);
  y1 = std::min(std::max(0.0f, y1), img_h);
  x2 = std::min(std::max(0.0f, x2), img_w);
  y2 = std::min(std::max(0.0f, y2), img_h);
  bbox[0] = x1;
  bbox[1] = y1;
  bbox[2] = x2 - x1;
  bbox[3] = y2 - y1;
}

void SOT::clampBBox(std::vector<float>& bbox,
                    const std::shared_ptr<BaseImage>& image, int min_side) {
  ensureBBoxBoundaries(bbox, image);
  float img_w = image->getWidth();
  float img_h = image->getHeight();
  if (bbox[2] < min_side) {
    bbox[2] = min_side;
    bbox[0] -= std::max(0.0f, bbox[0] + bbox[2] - img_w);
  }
  if (bbox[3] < min_side) {
    bbox[3] = min_side;
    bbox[1] -= std::max(0.0f, bbox[1] + bbox[3] - img_h);
  }
}

std::shared_ptr<BaseImage> SOT::preprocess(
    const std::shared_ptr<BaseImage>& image, const std::vector<float>& bbox,
    float offset, int crop_size, std::vector<int>& context) {
  if (!image || bbox.size() < 4) {
    LOGE("预处理输入无效");
    return nullptr;
  }

  int width = image->getWidth();
  int height = image->getHeight();

  // 计算上下文区域
  float x = bbox[0];
  float y = bbox[1];
  float w = bbox[2];
  float h = bbox[3];

  context[0] = static_cast<int>(x - w * offset);
  context[1] = static_cast<int>(y - h * offset);
  context[2] = static_cast<int>(w * (1.0f + 2 * offset));
  context[3] = static_cast<int>(h * (1.0f + 2 * offset));

  // 确保裁剪区域在图像内
  context[0] = std::max(0, std::min(context[0], width));
  context[1] = std::max(0, std::min(context[1], height));
  context[2] = std::max(0, std::min(context[2], width - context[0]));
  context[3] = std::max(0, std::min(context[3], height - context[1]));

  PreprocessParams params;
  memset(&params, 0, sizeof(PreprocessParams));
  params.dst_image_format = image->getImageFormat();
  params.dst_pixdata_type = image->getPixDataType();
  params.dst_width = crop_size;
  params.dst_height = crop_size;
  params.crop_x = context[0];
  params.crop_y = context[1];
  params.crop_width = context[2];
  params.crop_height = context[3];
  params.mean[0] = 0;
  params.mean[1] = 0;
  params.mean[2] = 0;
  params.scale[0] = 1;
  params.scale[1] = 1;
  params.scale[2] = 1;
  params.keep_aspect_ratio = false;
  LOGI(
      "dst_image_format: %d\n dst_pixdata_type: %d\n dst_width: %d\n "
      "dst_height: %d\n crop_x: %d\n crop_y: %d\n crop_width: %d\n "
      "crop_height: %d",
      params.dst_image_format, params.dst_pixdata_type, params.dst_width,
      params.dst_height, params.crop_x, params.crop_y, params.crop_width,
      params.crop_height);
  std::shared_ptr<BaseImage> crop_image =
      preprocessor_->preprocess(image, params, nullptr);
  return crop_image;
}

void SOT::updateScoreLst(float score) {
  if (frame_id_ > 10) {
    float avg_score =
        std::accumulate(score_lst_.begin(), score_lst_.end(), 0.0f) /
        score_lst_.size();
    score_ratio_ = avg_score > 0 ? score / avg_score : 0;
    if (score_ratio_ > occluded_score_ratio_threshold_) {
      score_lst_.pop_front();
      score_lst_.push_back(score);
    }
  } else {
    score_lst_.push_back(score);
  }
}

int32_t SOT::setModel(std::shared_ptr<BaseModel> sot_model) {
  if (!sot_model) {
    LOGE("sot_model is null");
    return -1;
  }
  sot_model_ = sot_model;
  return 0;
}

int32_t SOT::initialize(const std::shared_ptr<BaseImage>& image,
                        const std::vector<ObjectBoxInfo>& detect_boxes,
                        const ObjectBoxInfo& bbox) {
  if (detect_boxes.empty()) {
    // 如果检测框为空，则直接使用 bbox 初始化
    initBBox(image, bbox);
    return 0;
  }
  ObjectBoxInfo max_area_bbox;
  float max_area = 0.0f;
  for (auto& detect_box : detect_boxes) {
    if (detect_box.x1 >= bbox.x1 && detect_box.x2 <= bbox.x2 &&
        detect_box.y1 >= bbox.y1 && detect_box.y2 <= bbox.y2) {
      float area =
          (detect_box.x2 - detect_box.x1) * (detect_box.y2 - detect_box.y1);
      if (area > max_area) {
        max_area = area;
        max_area_bbox = detect_box;
      }
    }
  }
  if (max_area > 0) {
    // 如果bbox内的最大面积的检测框不为空，则使用 max_area_bbox 初始化
    initBBox(image, max_area_bbox);
  } else {
    // 如果bbox内的最大面积的检测框为空，则使用 bbox 初始化
    initBBox(image, bbox);
  }

  return 0;
}

int32_t SOT::initialize(const std::shared_ptr<BaseImage>& image,
                        const std::vector<ObjectBoxInfo>& detect_boxes, float x,
                        float y) {
  if (detect_boxes.empty()) {
    LOGE("该位置无检测框");
    return -1;
  }
  for (auto& detect_box : detect_boxes) {
    if (detect_box.x1 <= x && detect_box.x2 >= x && detect_box.y1 <= y &&
        detect_box.y2 >= y) {
      initBBox(image, detect_box);
      return 0;
    }
  }
  LOGE("该位置无检测框");
  return -1;
}

int32_t SOT::initBBox(const std::shared_ptr<BaseImage>& image,
                      const ObjectBoxInfo& init_bbox) {
  float x = init_bbox.x1;
  float y = init_bbox.y1;
  float w = init_bbox.x2 - init_bbox.x1;
  float h = init_bbox.y2 - init_bbox.y1;
  std::vector<float> init_bbox_xywh_format = {x, y, w, h};
  std::vector<int> context;
  context.resize(4);
  clampBBox(init_bbox_xywh_format, image);
  x = init_bbox_xywh_format[0];
  y = init_bbox_xywh_format[1];
  w = init_bbox_xywh_format[2];
  h = init_bbox_xywh_format[3];
  current_bbox_ = {x, y, w, h};
  kalman_tracker_ = std::make_shared<KalmanBoxTracker>(current_bbox_);
  template_image_ = preprocess(image, current_bbox_, template_bbox_offset_,
                               template_size_, context);
  if (!template_image_) {
    LOGE("模板提取失败");
    return -1;
  }
  is_initialized_ = true;
  LOGI("跟踪器初始化成功");
  return 0;
}

int32_t SOT::track(const std::shared_ptr<BaseImage>& image, uint64_t frame_id,
                   TrackerInfo& tracker_info) {
  if (!image) {
    LOGE("输入图像为空");
    return -1;
  }
  std::vector<int> context;
  context.resize(4);
  std::shared_ptr<BaseImage> search_image = preprocess(
      image, current_bbox_, search_bbox_offset_, instance_size_, context);

  std::vector<std::vector<std::shared_ptr<BaseImage>>> input_images = {
      {template_image_, search_image}};
  std::vector<std::shared_ptr<ModelOutputInfo>> output_datas;

  sot_model_->inference(input_images, output_datas);

  if (output_datas.empty()) {
    LOGE("跟踪结果为空");
    return -1;
  }

  std::shared_ptr<ModelBoxInfo> track_result =
      std::dynamic_pointer_cast<ModelBoxInfo>(output_datas[0]);
  if (!track_result) {
    LOGE("跟踪结果转换失败");
    return -1;
  }

  if (track_result->bboxes.empty()) {
    tracker_info.status_ = TrackStatus::LOST;
    return 0;
  }

  float x1 = track_result->bboxes[0].x1;
  float y1 = track_result->bboxes[0].y1;
  float w = track_result->bboxes[0].x2 - x1;
  float h = track_result->bboxes[0].y2 - y1;

  std::vector<float> bbox = {x1, y1, w, h};
  float score = track_result->bboxes[0].score;
  std::vector<float> kalman_bbox;

  kalman_bbox = kalman_tracker_->predict();
  if (frame_id - frame_id_ > 1) {
    for (int i = 0; i < frame_id - frame_id_ - 1; i++) {
      kalman_tracker_->update(kalman_bbox, false);
      kalman_bbox = kalman_tracker_->predict();
    }
  }

  float iou;
  if (kalman_tracker_->update_count_ > kalman_update_count_) {
    ObjectBoxInfo bbox1, bbox2;
    bbox1.x1 = bbox[0];
    bbox1.y1 = bbox[1];
    bbox1.x2 = bbox[0] + bbox[2];
    bbox1.y2 = bbox[1] + bbox[3];
    bbox2.x1 = kalman_bbox[0];
    bbox2.y1 = kalman_bbox[1];
    bbox2.x2 = kalman_bbox[0] + kalman_bbox[2];
    bbox2.y2 = kalman_bbox[1] + kalman_bbox[3];
    iou = MotBoxHelper::calculateIOU(bbox1, bbox2);
  } else {
    iou = 1.0f;
    kalman_bbox = bbox;
  }

  // 计算缩放比例
  float w_scale = context[2] / static_cast<float>(instance_size_);
  float h_scale = context[3] / static_cast<float>(instance_size_);

  // 创建边界框
  std::vector<float> scaled_bbox = {x1 * w_scale + context[0],
                                    y1 * h_scale + context[1], w * w_scale,
                                    h * h_scale};
  clampBBox(scaled_bbox, image);
  // 更新边界框, 用于下一帧跟踪
  current_bbox_ = scaled_bbox;
  updateScoreLst(score);
  float size_ratio;
  float current_w_h_ratio = w / h;
  if (prev_w_h_ratio_ != 0) {
    size_ratio = current_w_h_ratio / prev_w_h_ratio_;
  } else {
    size_ratio = 1.0f;
  }
  if (status_ == TrackStatus::TRACKED) {
    prev_w_h_ratio_ = current_w_h_ratio;
  }
  getStatus(bbox, kalman_bbox, score, score_ratio_, iou, size_ratio);
  // 输出结果
  tracker_info.box_info_.x1 = scaled_bbox[0];
  tracker_info.box_info_.y1 = scaled_bbox[1];
  tracker_info.box_info_.x2 = scaled_bbox[0] + scaled_bbox[2];
  tracker_info.box_info_.y2 = scaled_bbox[1] + scaled_bbox[3];
  tracker_info.box_info_.score = score;
  tracker_info.box_info_.class_id = 0;
  tracker_info.status_ = TrackStatus::TRACKED;
  if (status_ == TrackStatus::TRACKED) {
    if (sot_info_.is_occluded) {
      status_ = TrackStatus::LOST;
      kalman_tracker_->update(kalman_bbox, false);
      last_reliable_template_bbox_ = scaled_bbox;
      current_bbox_ = last_reliable_template_bbox_;
      sot_info_.template_bbox = last_reliable_template_bbox_;
      lost_frames_ = 1;
    } else {
      kalman_tracker_->update(bbox, true);
      last_reliable_template_bbox_ = scaled_bbox;
      current_bbox_ = last_reliable_template_bbox_;
      sot_info_.template_bbox = last_reliable_template_bbox_;
      lost_frames_ = 0;
    }
  } else {
    if (sot_info_.is_reappear) {
      status_ = TrackStatus::TRACKED;
      kalman_tracker_->update(bbox, true);
      last_reliable_template_bbox_ = scaled_bbox;
      current_bbox_ = last_reliable_template_bbox_;
      sot_info_.template_bbox = last_reliable_template_bbox_;
      lost_frames_ = 0;
    } else {
      kalman_tracker_->update(kalman_bbox, false);
      lost_frames_ += 1;
      float expand_ratio = 1.0f;
      if (last_reliable_template_bbox_.size() > 0) {
        expand_ratio =
            std::min(1.0f + (lost_frames_ / 20.0f) * 1.0f, max_expand_ratio_);
        float x = last_reliable_template_bbox_[0];
        float y = last_reliable_template_bbox_[1];
        float w = last_reliable_template_bbox_[2];
        float h = last_reliable_template_bbox_[3];
        float cx = x + w / 2;
        float cy = y + h / 2;
        float new_w = w * expand_ratio;
        float new_h = h * expand_ratio;
        float new_x = cx - new_w / 2;
        float new_y = cy - new_h / 2;
        sot_info_.template_bbox = {new_x, new_y, new_w, new_h};
        current_bbox_ = sot_info_.template_bbox;
      }
    }
  }
  frame_id_ = frame_id;
  if (sot_info_.is_occluded && lost_frames_ > 3) {
    tracker_info.status_ = TrackStatus::LOST;
  }
  return 0;
}