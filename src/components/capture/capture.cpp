#include "capture.hpp"
#include "tdl_model_factory.hpp"
#include "utils/tdl_log.hpp"
#include "utils/capture_helper.hpp"
#define EYE_DISTANCE_STANDARD 80.
#define ABS(x) ((x) >= 0 ? (x) : (-(x)))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// FaceCapture 实现
FaceCapture::FaceCapture(const std::string& capture_dir,
                       std::shared_ptr<BaseModel> face_ld_model,
                       const std::string& model_path)
    : ObjectCapture(capture_dir, ModelType::KEYPOINT_FACE_V2, model_path),
      face_ld_model(std::move(face_ld_model)) {}

void FaceCapture::updateCaptureData(
    std::shared_ptr<BaseImage> image,
    uint64_t frame_id,
    const std::vector<ObjectBoxInfo>& boxes,
    const std::vector<TrackerInfo>& tracks
) {
    for (const auto& track : tracks){
        auto track_id = track.track_id_;
        if(track.obj_idx_ == -1){
          if (capture_infos_.find(track_id) == capture_infos_.end()){
              continue;
          } else {
              capture_infos_[track_id].miss_counter++;
              // 检查 miss_counter 是否超过阈值
              if (capture_infos_[track_id].miss_counter > MISS_TIME_LIMIT) {
                  std::cout << "Removing track ID: " << track_id << " due to high miss_counter: " 
                            << capture_infos_[track_id].miss_counter << std::endl;
                  capture_infos_.erase(track_id);
              }
              continue;
          }
        } 
        float obj_quality = computeFaceQuality(image, boxes, tracks, AREA_RATIO);
        // 如果当前 track_id 不在缓存中，则初始化
        if (capture_infos_.find(track_id) == capture_infos_.end()) {
            capture_infos_[track_id] = ObjectCaptureInfo{
                .quality = obj_quality,
                .image = image,
                .last_capture_frame_id = frame_id,
                .miss_counter = 0
            };
        } else {
            // 更新缓存逻辑：如果当前帧质量更优，则更新缓存图像和质量
            auto& info = capture_infos_[track_id];
            if (obj_quality > info.quality) {
                info.quality = obj_quality;
                info.image = image;
                info.miss_counter = 0;
            }
        }
    }
    // capture_infos_超过数量，保留最近更新
    if (capture_infos_.size() > MAX_CAPTURE_NUM) {
        std::vector<uint64_t> keys;
        for (const auto& item : capture_infos_) {
            keys.push_back(item.first); 
        }
        std::sort(keys.begin(), keys.end(),
            [&](uint64_t a, uint64_t b) {
                return capture_infos_[a].last_capture_frame_id < capture_infos_[b].last_capture_frame_id;
            });
        
        for (size_t i = 0; i < keys.size() - MAX_CAPTURE_NUM; ++i) {
            capture_infos_.erase(keys[i]);
        }
    }
}

void FaceCapture::getCaptureData(
    uint64_t frame_id,
    std::vector<ObjectCaptureInfo>& captures
) {
    for (auto& item : capture_infos_) {
        auto& track_id = item.first;
        auto& info = item.second;
        if (frame_id - info.last_capture_frame_id >= capture_interval) {
            info.last_capture_frame_id = frame_id;
            captures.push_back(ObjectCaptureInfo{
                info.quality,
                info.image,
                info.last_capture_frame_id,
                info.miss_counter
            });
        }
    }
}

float FaceCapture::get_score(ObjectBoxInfo bbox, std::shared_ptr<ModelLandmarksInfo> landmarks_meta, TrackerInfo tracker,uint32_t img_w, uint32_t img_h, bool fl_model) {
  // Predict
  Pose* pose = (Pose*)malloc(sizeof(Pose));
  int ret = Predict(landmarks_meta,pose);
  if (ret != 0) {
    throw std::runtime_error("Unknown error in pose estimation");
  }
  float velx = tracker.velocity_x_;
  float vely = tracker.velocity_y_;
  float blurness = landmarks_meta->attributes[TDLObjectAttributeType::OBJECT_CLS_ATTRIBUTE_FACE_BLURNESS];
  float nose_x = landmarks_meta->landmarks_x[2];

  float left_max = MIN(landmarks_meta->landmarks_x[0], landmarks_meta->landmarks_x[3]);
  float right_max = MAX(landmarks_meta->landmarks_x[1], landmarks_meta->landmarks_x[4]);
  float width = bbox.x2 - bbox.x1;
  float height = bbox.y2 - bbox.y1;
  float l_ = nose_x - left_max;
  float r_ = right_max - nose_x;

  float eye_diff_x = landmarks_meta->landmarks_x[1] - landmarks_meta->landmarks_x[0];
  float eye_diff_y = landmarks_meta->landmarks_y[1] - landmarks_meta->landmarks_y[0];
  float eye_size = sqrt(eye_diff_x * eye_diff_x + eye_diff_y * eye_diff_y);

  float mouth_diff_x = landmarks_meta->landmarks_x[4] - landmarks_meta->landmarks_x[3];
  float mouth_diff_y = landmarks_meta->landmarks_y[4] - landmarks_meta->landmarks_y[3];
  float mouth_size = sqrt(mouth_diff_x * mouth_diff_x + mouth_diff_y * mouth_diff_y);
  float vel = sqrt(velx * velx + vely * vely);

  if (landmarks_meta->landmarks_x[1] > bbox.x2 || landmarks_meta->landmarks_x[2] > bbox.x2 || landmarks_meta->landmarks_x[4] > bbox.x2 ||
      landmarks_meta->landmarks_x[0] < bbox.x1 || landmarks_meta->landmarks_x[2] < bbox.x1 || landmarks_meta->landmarks_x[3] < bbox.x1) {
    return 0.0;
  } else if ((l_ + 0.01 * width) < 0 || (r_ + 0.01 * width) < 0 || (eye_size / width) < 0.25 ||
             (mouth_size / width) < 0.15) {
    return 0.0;
  } else if ((landmarks_meta->landmarks_y[0] < bbox.y1 || landmarks_meta->landmarks_y[1] < bbox.y1 || landmarks_meta->landmarks_y[3] > bbox.y2 ||
              landmarks_meta->landmarks_y[4] > bbox.y2)) {
    return 0.0;
  } else if (width * height < (25 * 25)) {
    return 0.0;
  } else if (pose != NULL) {
    float face_size = ((bbox.y2 - bbox.y1) + (bbox.x2 - bbox.x1)) / 2;
    float size_score = 0;
    float pose_score = 1. - (ABS(pose->yaw) + ABS(pose->pitch) + ABS(pose->roll) * 0.5) / 3.;
    // printf("pose_score_angle: %f, pose->yaw: %f, pose->pitch: %f, pose->roll: %f\n", pose_score,
    // pose->yaw, pose->pitch, pose->roll);

    float area_score;
    float wpose = 0.8;
    float wsize = 0.2;

    float h_ratio = face_size / (float)img_h;

    if (h_ratio < 0.06) {  // 64/1080
      wpose = 0.4;
      area_score = 0;
    } else if (h_ratio < 0.0685)  // 74/1080
    {
      wpose = 0.6;
      // area_score = log(face_size/(float)img_h)/log(4.0);
      area_score = log(h_ratio * 20.0) / log(4.0);
      if (pose_score > 0.8) {
        pose_score = 0.8;
      }
      size_score = 0.75;

    } else {
      area_score =
          0.23 + (2.0 - 1.0 / (h_ratio * 4.38 + 0.2)) / 5.0;  // 0.23 ~= log(0.0685*20.0)/log(4.0)
      size_score = eye_size / (bbox.x2 - bbox.x1);
      size_score += mouth_size / (bbox.x2 - bbox.x1);
    }
    if (fl_model && h_ratio > 0.06) {
      wpose = 0.8;
    }

    float velscore = vel * 0.04;
    if (velscore > 0.2) {
      velscore = 0.2;
    }

    // printf("img_h: %d, face_size:%f, size_score:%f, area_score: %f, vel:%f, velscore: %f,
    // blurness:%f\n", img_h, face_size,size_score , area_score, vel, velscore, blurness);

    pose_score = pose_score * wpose + wsize * size_score + area_score - blurness * 0.2;

    if (bbox.x1 < 0.5 * width || img_w - bbox.x2 < 0.5 * width || bbox.y1 < 0.5 * height ||
        img_h - bbox.y2 < 0.5 * height) {
      pose_score -= 0.2;
    }
    return pose_score;
  } else {
    return 0.5;
  }
}
float FaceCapture::computeFaceQuality(
    std::shared_ptr<BaseImage> image,
    const std::vector<ObjectBoxInfo>& boxes,
    const std::vector<TrackerInfo>& tracks,
    quality_assessment_e qa_method
) {
    std::shared_ptr<ModelBoxInfo> box_info = std::make_shared<ModelBoxInfo>();
    box_info->bboxes = boxes;
    std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
    quality_model->inference(image, box_info ,out_datas);
    std::vector<float> quality_score;

    std::shared_ptr<ModelLandmarksInfo> landmarks_meta =
        std::static_pointer_cast<ModelLandmarksInfo>(out_datas[0]);
    if (qa_method == AREA_RATIO) {
      landmarks_meta->attributes[TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_POSE] = get_score(box_info->bboxes[0],landmarks_meta, tracks[0],image->getWidth(), image->getHeight(), false);
    } else if (qa_method == EYES_DISTANCE) {
      float dx = landmarks_meta->landmarks_x[0] - landmarks_meta->landmarks_x[1];
      float dy = landmarks_meta->landmarks_y[0] - landmarks_meta->landmarks_y[1];
      float dist_score = sqrt(dx * dx + dy * dy) / EYE_DISTANCE_STANDARD;
      landmarks_meta->attributes[TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_POSE] = (dist_score >= 1.) ? 1. : dist_score;
    } 
    
    return landmarks_meta->attributes[TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_POSE];
}
