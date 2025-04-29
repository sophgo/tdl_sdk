
#include "mot/kalman_tracker.hpp"
#include <math.h>
#include <iostream>
#include "utils/mot_box_helper.hpp"
#include "utils/tdl_log.hpp"
KalmanTracker::~KalmanTracker() { LOGI("destroy tracker:%lu", id_); }

KalmanTracker::KalmanTracker(const uint64_t &frame_id, const KalmanFilter &kf,
                             const uint64_t &id, const ObjectBoxInfo &box,
                             int img_width, int img_height) {
  this->id_ = id;
  this->box_ = box;
  this->last_updated_frame_id_ = frame_id;

  this->bounding_ = false;
  this->status_ = TrackStatus::NEW;
  this->matched_times_ = 1;
  this->unmatched_times_ = 0;
  this->false_update_times_ = 0;
  this->ages_ = 1;
  this->velocity_x_ = 0;
  this->velocity_y_ = 0;
  this->img_width_ = img_width;
  this->img_height_ = img_height;

  DETECTBOX bbox_xyah = MotBoxHelper::convertToXYAH(box);
  auto init_data = kf.initiate(bbox_xyah);

  this->mean = init_data.first;
  this->covariance = init_data.second;
  LOGI(
      "init "
      "trackid:%d,box:[%.2f,%.2f,%.2f,%.2f],xyah:[%.2f,%.2f,%.2f,%.2f],mean:[%."
      "2f,%.2f,%.2f,%.2f],covariance:[%.2f,%.2f,%.2f,%.2f]",
      id_, box.x1, box.y1, box.x2, box.y2, bbox_xyah(0), bbox_xyah(1),
      bbox_xyah(2), bbox_xyah(3), mean(0), mean(1), mean(2), mean(3),
      covariance(0, 0), covariance(1, 1), covariance(2, 2), covariance(3, 3));
  // this->x_.block(0, 0, DIM_Z, 1) = bbox_xyah.transpose();
  // this->x_.block(DIM_Z, 0, DIM_Z, 1) = Eigen::MatrixXf::Zero(DIM_Z, 1);
  // this->P_ = Eigen::MatrixXf::Zero(DIM_X, DIM_X);
  // float diag_val[DIM_X] = {1, 1, 10, 10, 100, 100, 10000, 10000};
  // for (int i = 0; i < DIM_X; i++) {
  //   this->P_(i, i) = diag_val[i];
  // }
}

uint64_t KalmanTracker::getPairTrackID() {
  if (pair_track_infos_.size() == 0) {
    return 0;
  }
  if (pair_track_infos_.size() > 1) {
    std::stringstream ss;
    ss << "trackid:" << id_
       << " has multiple pairtracks,num: " << pair_track_infos_.size()
       << ",tracks:";
    for (auto &pair : pair_track_infos_) {
      ss << pair.first << ",";
    }
    LOGW("%s", ss.str().c_str());
  }
  return pair_track_infos_.begin()->first;
}
void KalmanTracker::predict(const KalmanFilter &kf) {
  kf.predict(mean, covariance);
  unmatched_times_ += 1;
  ages_ += 1;
}
//
void KalmanTracker::update(const uint64_t &frame_id, const KalmanFilter &kf,
                           const ObjectBoxInfo *p_bbox,
                           const TrackerConfig &conf) {
  if (p_bbox != nullptr) {
    unmatched_times_ = 0;
    matched_times_ += 1;
    false_update_times_ = 0;

    DETECTBOX xyah = MotBoxHelper::convertToXYAH(*p_bbox);

    auto update_data = kf.update(mean, covariance, xyah);
    mean = update_data.first;
    covariance = update_data.second;
    if (status_ == TrackStatus::NEW &&
        matched_times_ >= conf.track_confirmed_frames_) {
      status_ = TrackStatus::TRACKED;
      LOGI("trackid:%d,update to tracked", id_);
    }
    LOGI(
        "update "
        "trackid:%d,inputbox:[%.1f,%.1f,%.1f,%.1f],lastbox:[%.1f,%.1f,%."
        "1f,%.1f],inputxyah:[%.1f,%.1f,%.1f,%.1f],updatexyah:[%.1f,%.1f,%."
        "1f,%.1f]",
        id_, p_bbox->x1, p_bbox->y1, p_bbox->x2, p_bbox->y2, box_.x1, box_.y1,
        box_.x2, box_.y2, xyah(0), xyah(1), xyah(2), xyah(3), mean(0), mean(1),
        mean(2), mean(3));

    DETECTBOX tlwh = getBBoxTLWH();
    float x1 = tlwh(0);
    float y1 = tlwh(1);
    float x2 = tlwh(0) + tlwh(2);
    float y2 = tlwh(1) + tlwh(3);
    int frame_diff = frame_id - last_updated_frame_id_;

    LOGI(
        "update "
        "trackid:%d,box:[%.1f,%.1f,%.1f,%.1f],oldbox:[%.1f,%.1f,%.1f,%.1f],"
        "score:%.1f",
        id_, x1, y1, x2, y2, box_.x1, box_.y1, box_.x2, box_.y2, box_.score);
    box_.x1 = x1;
    box_.y1 = y1;
    box_.x2 = x2;
    box_.y2 = y2;
    box_.score = p_bbox->score;
    float vel_x = mean(4) / frame_diff;
    float vel_y = mean(5) / frame_diff;
    if (ages_ == 1) {
      velocity_x_ = vel_x;
      velocity_y_ = vel_y;
    } else {
      velocity_x_ = 0.9 * velocity_x_ + vel_x * 0.1;
      velocity_y_ = 0.9 * velocity_y_ + vel_y * 0.1;
    }
    last_updated_frame_id_ = frame_id;

  } else {
    // do not update velocity
    LOGI(
        "missed track id:%d, tracker_state: %d, unmatched_times:%d, "
        "max_unmatched_times:%d\n",
        id_, status_, unmatched_times_, conf.max_unmatched_times_);
    if (status_ == TrackStatus::NEW) {
      status_ = TrackStatus::REMOVED;
    } else if (unmatched_times_ >= conf.max_unmatched_times_) {
      status_ = TrackStatus::REMOVED;
    } else {
      status_ = TrackStatus::LOST;
    }
  }
  updateBoundaryState();
}
void KalmanTracker::falseUpdateFromPair(const uint64_t &frame_id,
                                        const KalmanFilter &kf,
                                        KalmanTracker *p_other,
                                        const TrackerConfig &conf) {
  LOGI("false update pairtrack:%d,with:%d\n", id_, p_other->id_);

  if (p_other->pair_track_infos_.count(id_) == 0) {
    LOGE("false update current trackid:%d not found in pair track:%d\n",
         (int)id_, (int)p_other->id_);
    return;
  }
  DETECTBOX false_box;
  stCorrelateInfo corre = p_other->pair_track_infos_[id_];

  DETECTBOX pairbox = p_other->getBBoxTLWH();
  false_box(0) = p_other->mean(0) + pairbox(2) * corre.offset_scale_x;
  false_box(1) = p_other->mean(1) + pairbox(3) * corre.offset_scale_y;
  false_box(2) = this->mean(2);  // use current aspect ratio
  false_box(3) = pairbox(3) * corre.pair_size_scale_y;

  // to keep this track alive
  if (unmatched_times_ >=
      conf.max_unmatched_times_ - conf.track_pair_update_missed_times_) {
    unmatched_times_ =
        conf.max_unmatched_times_ - conf.track_pair_update_missed_times_;
  }
  false_update_times_ += 1;

  auto update_data = kf.update(mean, covariance, false_box);
  mean = update_data.first;
  covariance = update_data.second;
  int frame_diff = frame_id - last_updated_frame_id_;
  float vel_x = mean(4) / frame_diff;
  float vel_y = mean(5) / frame_diff;
  if (ages_ == 1) {
    assert(false);
  } else {
    velocity_x_ = 0.9 * velocity_x_ + vel_x * 0.1;
    velocity_y_ = 0.9 * velocity_y_ + vel_y * 0.1;
  }
  last_updated_frame_id_ = frame_id;
  updateBoundaryState();
}
void KalmanTracker::updatePairInfo(KalmanTracker *p_other) {
  LOGI("update pairtrack:%d,with:%d\n", id_, p_other->id_);
  if (pair_track_infos_.size() != 0 &&
      pair_track_infos_.count(p_other->id_) == 0) {
    LOGW("trackid:%d already has pairtrack:%d,now to add pairtrack:%d", id_,
         pair_track_infos_.begin()->first, p_other->id_);
  }

  DETECTBOX cur_box = getBBoxTLWH();
  DETECTBOX pair_box = p_other->getBBoxTLWH();

  if (p_other->pair_track_infos_.count(id_) == 0) {
    stCorrelateInfo corre;
    memset((void *)&corre, 0, sizeof(corre));
    p_other->pair_track_infos_[id_] = corre;
  }
  updateCorre(pair_box, cur_box, p_other->pair_track_infos_[id_], 0.5);

  if (pair_track_infos_.count(p_other->id_) == 0) {
    stCorrelateInfo corre;
    memset((void *)&corre, 0, sizeof(corre));
    pair_track_infos_[p_other->id_] = corre;
  }
  updateCorre(cur_box, pair_box, pair_track_infos_[p_other->id_], 0.5);
}
void KalmanTracker::resetPairInfo() {
  LOGI("reset pairinfo of trackid:%d", id_);
  pair_track_infos_.clear();
}
DETECTBOX KalmanTracker::getBBoxTLWH() const {
  DETECTBOX bbox_tlwh;
  bbox_tlwh(2) = mean(2) * mean(3);  // H
  bbox_tlwh(3) = mean(3);            // W
  bbox_tlwh(0) = mean(0) - 0.5 * bbox_tlwh(2);
  bbox_tlwh(1) = mean(1) - 0.5 * bbox_tlwh(3);
  return bbox_tlwh;
}
ObjectBoxInfo KalmanTracker::getBoxInfo() const {
  ObjectBoxInfo box;
  if (status_ == TrackStatus::NEW) {
    box = box_;
  } else {
    DETECTBOX tlwh = getBBoxTLWH();
    box.x1 = tlwh(0);
    box.y1 = tlwh(1);
    box.x2 = tlwh(0) + tlwh(2);
    box.y2 = tlwh(1) + tlwh(3);
    box.score = box_.score;
    box.object_type = box_.object_type;
  }
  return box;
}
void KalmanTracker::updateCorre(const DETECTBOX &cur_tlwh,
                                const DETECTBOX &pair_tlwh,
                                stCorrelateInfo &cur_corre, float w_cur) {
  float w_prev = 1 - w_cur;
  float cur_ctx = cur_tlwh(0) + cur_tlwh(2) / 2;
  float cur_cty = cur_tlwh(1) + cur_tlwh(3) / 2;
  float pair_ctx = pair_tlwh(0) + pair_tlwh(2) / 2;
  float pair_cty = pair_tlwh(1) + pair_tlwh(3) / 2;
  float cur_w = cur_tlwh(2);
  float cur_h = cur_tlwh(3);

  stCorrelateInfo cur;
  cur.offset_scale_x = (pair_ctx - cur_ctx) / cur_w;
  cur.offset_scale_y = (pair_cty - cur_cty) / cur_h;
  cur.pair_size_scale_x = float(pair_tlwh(2)) / cur_w;
  cur.pair_size_scale_y = float(pair_tlwh(3)) / cur_h;
  if (cur_corre.votes == 0) {
    cur_corre = cur;
    cur_corre.votes = 1;
    cur_corre.time_since_correlated = 0;

    return;
  }
  cur_corre.offset_scale_x =
      cur_corre.offset_scale_x * w_prev + cur.offset_scale_x * w_cur;
  cur_corre.offset_scale_y =
      cur_corre.offset_scale_y * w_prev + cur.offset_scale_y * w_cur;
  cur_corre.pair_size_scale_x =
      cur_corre.pair_size_scale_x * w_prev + cur.pair_size_scale_x * w_cur;
  cur_corre.pair_size_scale_y =
      cur_corre.pair_size_scale_y * w_prev + cur.pair_size_scale_y * w_cur;
  cur_corre.time_since_correlated = 0;
  cur_corre.votes += 1;
}

void KalmanTracker::updateBoundaryState() {
  if (img_width_ == 0 || img_height_ == 0) {
    return;
  }
  ObjectBoxInfo box = getBoxInfo();
  ObjectBoxInfo img_box;
  img_box.x1 = 0;
  img_box.y1 = 0;
  img_box.x2 = img_width_;
  img_box.y2 = img_height_;
  float iou = MotBoxHelper::calculateIOUOnFirst(box, img_box);
  if (iou < 0.5) {
    bounding_ = true;
    status_ = TrackStatus::REMOVED;
    LOGI("trackid:%d,boundary,iou:%.2f,imgw:%d,imgh:%d", id_, iou, img_width_,
         img_height_);
  }
}
