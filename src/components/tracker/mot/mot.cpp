#include "mot/mot.hpp"
#include <iostream>
#include <set>
#include "mot/munkres.hpp"
#include "utils/cost_matrix_helper.hpp"
#include "utils/mot_box_helper.hpp"
#include "utils/tdl_log.hpp"
#define DEFAULT_X_CONSTRAINT_A_MIN 0.25
#define DEFAULT_X_CONSTRAINT_A_MAX 4.0
#define DEFAULT_X_CONSTRAINT_H_MIN 32
#define DEFAULT_X_CONSTRAINT_H_MAX 512
MOT::MOT() {}
MOT::~MOT() {
  trackers_.clear();
  det_track_ids_.clear();
  pair_obj_idxes_.clear();
}
int32_t MOT::track(std::vector<ObjectBoxInfo> &boxes, uint64_t frame_id,
                   std::vector<TrackerInfo> &tracker_infos) {
  if (img_width_ == 0 || img_height_ == 0) {
    LOGW("img_width_ or img_height_ is not set,boundary check will be skipped");
  }
  det_track_ids_.clear();
  det_track_ids_.resize(boxes.size());
  pair_obj_idxes_.clear();
  pair_obj_idxes_.resize(boxes.size(), -1);

  LOGI("frame_id:%lu,boxes.size:%d,trackers.size:%d", frame_id, boxes.size(),
       trackers_.size());
  for (auto &t : trackers_) {
    t->predict(kalman_filter_);
  }
  std::set<TDLObjectType> obj_types;
  std::map<TDLObjectType, int> obj_type_size;
  int i = 0;
  for (auto &box : boxes) {
    if (box.object_type == TDLObjectType::OBJECT_TYPE_UNDEFINED) {
      LOGW("skip undefined object type:%d", box.object_type);
      continue;
    }
    LOGI("boxi:%d,obj_type:%d,[%.1f,%.1f,%.1f,%.1f]", i++, box.object_type,
         box.x1, box.y1, box.x2, box.y2);
    obj_types.insert(box.object_type);
    if (obj_type_size.count(box.object_type) == 0) {
      obj_type_size[box.object_type] = 0;
    }
    obj_type_size[box.object_type] += 1;
  }

  // use single tracking for single object types
  for (auto &type : obj_types) {
    LOGI("single_type:%d", type);
    trackAlone(boxes, type);
  }

  for (auto &pair : object_pair_config_) {
    if (obj_types.find(pair.first) == obj_types.end() ||
        obj_types.find(pair.second) == obj_types.end()) {
      continue;
    }
    LOGI("paired_types:%d,%d,obj_size:%d,%d", pair.first, pair.second,
         obj_type_size[pair.first], obj_type_size[pair.second]);

    trackFuse(boxes, pair.first, pair.second);
  }

  std::vector<ModelFeatureInfo> features;  // not supported yet
  updateTrackers(boxes, features);
  std::map<uint64_t, int> exported_tracks;
  std::map<uint64_t, int> track_id_to_idx;
  std::map<uint64_t, uint64_t> pair_track_ids = getPairTrackIds();

  for (size_t i = 0; i < trackers_.size(); i++) {
    track_id_to_idx[trackers_[i]->id_] = i;
  }
  for (size_t i = 0; i < det_track_ids_.size(); i++) {
    uint64_t track_id = det_track_ids_[i];
    if (track_id == 0) {
      continue;
    }
    TrackerInfo tinfo;
    tinfo.track_id_ = track_id;
    tinfo.box_info_ = trackers_[track_id_to_idx[track_id]]->getBoxInfo();
    tinfo.status_ = trackers_[track_id_to_idx[track_id]]->status_;
    tinfo.velocity_x_ = trackers_[track_id_to_idx[track_id]]->velocity_x_;
    tinfo.velocity_y_ = trackers_[track_id_to_idx[track_id]]->velocity_y_;
    tinfo.obj_idx_ = i;
    tinfo.pair_track_idx_ = pair_track_ids[track_id];
    exported_tracks[track_id] = 1;
    tracker_infos.push_back(tinfo);
  }
  for (auto &t : trackers_) {
    if (exported_tracks.count(t->id_) == 0) {
      TrackerInfo tinfo;
      tinfo.track_id_ = t->id_;
      tinfo.box_info_ = t->getBoxInfo();
      tinfo.status_ = t->status_;
      tinfo.velocity_x_ = t->velocity_x_;
      tinfo.velocity_y_ = t->velocity_y_;
      tinfo.obj_idx_ = -1;
      tinfo.pair_track_idx_ = pair_track_ids[t->id_];
      tracker_infos.push_back(tinfo);
    }
  }

  return 0;
}
void MOT::trackAlone(std::vector<ObjectBoxInfo> &boxes,
                     TDLObjectType obj_type) {
  std::vector<ModelFeatureInfo> features;
  std::vector<int> unmatched_bbox_idxes_high;
  std::vector<int> unmatched_bbox_idxes_low;
  std::vector<int> unmatched_tracker_idxes;
  std::map<uint64_t, int> trackid_idx_map;
  for (size_t i = 0; i < trackers_.size(); ++i) {
    trackid_idx_map[trackers_[i]->id_] = i;
  }
  for (size_t i = 0; i < boxes.size(); ++i) {
    if (boxes[i].object_type != obj_type && obj_type != OBJECT_TYPE_UNDEFINED) {
      // LOGI("boxid:%d,obj_type:%d,expect_obj_type:%d,skip", i,
      //      boxes[i].object_type, obj_type);
      continue;
    }

    if (boxes[i].score > tracker_config_.high_score_thresh_) {
      unmatched_bbox_idxes_high.push_back(i);
    } else {
      unmatched_bbox_idxes_low.push_back(i);
    }
  }

  for (size_t i = 0; i < trackers_.size(); ++i) {
    if (trackers_[i]->box_.object_type != obj_type &&
        obj_type != OBJECT_TYPE_UNDEFINED) {
      continue;
    }
    // it could only be recalled by pair object
    // if (trackers_[i].false_update_times_ >= 2) {
    //   LOGI("tracker:%lu,false_update_times:%d,skip", trackers_[i].id_,
    //        trackers_[i].false_update_times_);
    //   continue;
    // }

    unmatched_tracker_idxes.push_back(i);
  }
  LOGI("unmatched_tracker:%d,unmatched_bbox_high:%d,unmatched_bbox_low:%d",
       unmatched_tracker_idxes.size(), unmatched_bbox_idxes_high.size(),
       unmatched_bbox_idxes_low.size());
  MatchResult match_high = match(
      boxes, features, unmatched_tracker_idxes, unmatched_bbox_idxes_high,
      TrackCostType::BBOX_IOU, tracker_config_.high_score_iou_dist_thresh_);
  LOGI("match_high,matched_pairs:%d,unmatched_tracker:%d,unmatched_bbox:%d",
       match_high.matched_pairs.size(),
       match_high.unmatched_tracker_idxes.size(),
       match_high.unmatched_bbox_idxes.size());
  MatchResult match_low =
      match(boxes, features, match_high.unmatched_tracker_idxes,
            unmatched_bbox_idxes_low, TrackCostType::BBOX_IOU,
            tracker_config_.low_score_iou_dist_thresh_);
  LOGI("match_low,matched_pairs:%d,unmatched_tracker:%d,unmatched_bbox:%d",
       match_low.matched_pairs.size(), match_low.unmatched_tracker_idxes.size(),
       match_low.unmatched_bbox_idxes.size());
  match_high.matched_pairs.insert(match_high.matched_pairs.end(),
                                  match_low.matched_pairs.begin(),
                                  match_low.matched_pairs.end());
  for (size_t i = 0; i < match_high.matched_pairs.size(); i++) {
    int tracker_idx = match_high.matched_pairs[i].first;
    int bbox_idx = match_high.matched_pairs[i].second;
    det_track_ids_[bbox_idx] = trackers_[tracker_idx]->id_;
    LOGI("trackAlone,tracker_idx:%d,bbox_idx:%d,trackid:%lu,obj_type:%d",
         tracker_idx, bbox_idx, trackers_[tracker_idx]->id_,
         boxes[bbox_idx].object_type);
  }
}
void MOT::trackFuse(std::vector<ObjectBoxInfo> &boxes,
                    TDLObjectType priority_type, TDLObjectType secondary_type) {
  std::vector<int> priority_idxes;
  std::vector<int> secondary_idxes;
  for (size_t i = 0; i < boxes.size(); i++) {
    if (boxes[i].object_type == priority_type) {
      priority_idxes.push_back(i);
    } else if (boxes[i].object_type == secondary_type) {
      secondary_idxes.push_back(i);
    }
  }

  updatePairInfo(boxes, priority_idxes, secondary_idxes, priority_type,
                 secondary_type, 0.1);

  // recall unmatched object with pair info
  std::map<uint64_t, int> trackid_idx_map;
  for (size_t i = 0; i < trackers_.size(); i++) {
    trackid_idx_map[trackers_[i]->id_] = i;
  }

  // need to remove conflict pair info
  for (size_t i = 0; i < pair_obj_idxes_.size(); i++) {
    if (pair_obj_idxes_[i] == -1 || det_track_ids_[i] == 0) {
      continue;
    }
    int pair_obj_idx = pair_obj_idxes_[i];
    uint64_t pair_obj_trackid = det_track_ids_[pair_obj_idx];
    if (pair_obj_trackid == 0) {
      continue;
    }
    int tracker_idx = trackid_idx_map[det_track_ids_[i]];
    int pair_tracker_idx = trackid_idx_map[pair_obj_trackid];
    uint64_t trackid = trackers_[pair_tracker_idx]->getPairTrackID();
    if (trackid != 0 && pair_obj_trackid != 0 && det_track_ids_[i] != trackid) {
      LOGI(
          "got conflict pair info,det obji:%d,pair_obj_idx:%d,matched "
          "trackid:%lu,pair_trackid:%lu,pairtrackid_pair:%lu",
          i, pair_obj_idx, det_track_ids_[i], pair_obj_trackid, trackid);
      trackers_[tracker_idx]->resetPairInfo();
      trackers_[pair_tracker_idx]->resetPairInfo();
    }
  }

  // recall object i with pair info
  for (size_t i = 0; i < pair_obj_idxes_.size(); i++) {
    if (pair_obj_idxes_[i] == -1 || det_track_ids_[i] != 0) {
      continue;
    }
    int pair_obj_idx = pair_obj_idxes_[i];
    uint64_t pair_obj_trackid = det_track_ids_[pair_obj_idx];
    if (pair_obj_trackid == 0) {
      continue;
    }
    int pair_tracker_idx = trackid_idx_map[pair_obj_trackid];
    uint64_t trackid = trackers_[pair_tracker_idx]->getPairTrackID();
    if (trackid != 0) {
      det_track_ids_[i] = trackid;
      LOGI("recall pair obj:%d,pair_obj_idx:%d,trackid:%lu,pair_trackid:%lu", i,
           pair_obj_idx, trackid, pair_obj_trackid);
    }
  }
}

void MOT::updatePairInfo(std::vector<ObjectBoxInfo> &boxes,
                         const std::vector<int> &priority_idxes,
                         const std::vector<int> &secondary_idxes,
                         TDLObjectType priority_type,
                         TDLObjectType secondary_type, float corre_thresh) {
  // TODO: implement

  if (priority_idxes.empty() || secondary_idxes.empty()) {
    return;
  }
  float cost_thresh = 1 - corre_thresh;

  COST_MATRIX cost_matrix(priority_idxes.size(), secondary_idxes.size());
  for (size_t i = 0; i < priority_idxes.size(); i++) {
    for (size_t j = 0; j < secondary_idxes.size(); j++) {
      LOGI(
          "priority_idx:%d,secondary_idx:%d,prioritybox[%.1f,%.1f,%.1f,%.1f],"
          "secondarybox[%.1f,%.1f,%.1f,%.1f]",
          priority_idxes[i], secondary_idxes[j], boxes[priority_idxes[i]].x1,
          boxes[priority_idxes[i]].y1, boxes[priority_idxes[i]].x2,
          boxes[priority_idxes[i]].y2, boxes[secondary_idxes[j]].x1,
          boxes[secondary_idxes[j]].y1, boxes[secondary_idxes[j]].x2,
          boxes[secondary_idxes[j]].y2);
      cost_matrix(i, j) =
          1 - MotBoxHelper::calObjectPairScore(boxes[priority_idxes[i]],
                                               boxes[secondary_idxes[j]],
                                               priority_type, secondary_type);
      LOGI("cost_matrix(%d,%d):%f", i, j, cost_matrix(i, j));
    }
  }
  std::stringstream ss;
  ss << "priority_idxes:[";
  for (auto &idx : priority_idxes) {
    ss << idx << "-" << det_track_ids_[idx] << ",";
  }
  ss << "]\nsecondary_idxes:[";
  for (auto &idx : secondary_idxes) {
    ss << idx << "-" << det_track_ids_[idx] << ",";
  }
  ss << "]\ncost_matrix:\n[";
  for (size_t i = 0; i < priority_idxes.size(); i++) {
    ss << "[";
    for (size_t j = 0; j < secondary_idxes.size(); j++) {
      ss << cost_matrix(i, j) << ",";
    }
    ss << "]\n";
  }
  ss << "]\n";
  LOGI("%s", ss.str().c_str());
  Munkres munkres_solver(&cost_matrix);
  if (munkres_solver.solve() == MUNKRES_FAILURE) {
    LOGW("MUNKRES algorithm failed.");
    return;
  }
  for (size_t i = 0; i < priority_idxes.size(); i++) {
    int priority_idx = priority_idxes[i];
    int bbox_j = munkres_solver.m_match_result[i];
    if (bbox_j != -1 && cost_matrix(i, bbox_j) < cost_thresh) {
      int secondary_idx = secondary_idxes[bbox_j];
      pair_obj_idxes_[priority_idx] = secondary_idx;
      pair_obj_idxes_[secondary_idx] = priority_idx;
      LOGI(
          "construct "
          "pair,priority_type:%d,secondary_type:%d,priority_idx:%d,secondary_"
          "idx:%d",
          priority_type, secondary_type, priority_idx, secondary_idx);
    }
  }
}

MatchResult MOT::match(const std::vector<ObjectBoxInfo> &dets,
                       const std::vector<ModelFeatureInfo> &features,
                       const std::vector<int> &tracker_idxes,
                       const std::vector<int> &det_idxes,
                       TrackCostType cost_method, float max_distance) {
  MatchResult match_result;
  if (det_idxes.empty() || tracker_idxes.empty()) {
    match_result.unmatched_bbox_idxes = det_idxes;
    match_result.unmatched_tracker_idxes = tracker_idxes;
    return match_result;
  }
  COST_MATRIX cost_matrix;
  switch (cost_method) {
    case TrackCostType::FEATURE: {
      cost_matrix = CostMatrixHelper::getCostMatrixFeature(
          trackers_, dets, features, tracker_idxes, det_idxes);
      break;
    }
    case TrackCostType::BBOX_IOU: {
      cost_matrix = CostMatrixHelper::getCostMatrixBBox(
          trackers_, dets, tracker_idxes, det_idxes);
      break;
    }
    default: {
      LOGW("Unknown cost method %d", cost_method);
      return match_result;
    }
  }
  Munkres munkres_solver(&cost_matrix);
  if (munkres_solver.solve() == MUNKRES_FAILURE) {
    LOGW("MUNKRES algorithm failed.");
    // return empty results if failed to solve
    match_result.unmatched_tracker_idxes.clear();
    match_result.unmatched_bbox_idxes.clear();
    return match_result;
  }
  int bbox_num = det_idxes.size();
  int tracker_num = tracker_idxes.size();
  bool *matched_tracker_i = new bool[tracker_num];
  bool *matched_bbox_j = new bool[bbox_num];
  memset(matched_tracker_i, false, tracker_num * sizeof(bool));
  memset(matched_bbox_j, false, bbox_num * sizeof(bool));

  for (int i = 0; i < tracker_num; i++) {
    int bbox_j = munkres_solver.m_match_result[i];
    if (bbox_j != -1) {
      int tracker_idx = tracker_idxes[i];
      int bbox_idx = det_idxes[bbox_j];
      ObjectBoxInfo tracker_box = trackers_[tracker_idx]->getBoxInfo();
      ObjectBoxInfo det_box = dets[bbox_idx];
      float matched_iou =
          MotBoxHelper::calculateIOUOnFirst(tracker_box, det_box);
      if (cost_matrix(i, bbox_j) < max_distance && matched_iou > 0.3) {
        matched_tracker_i[i] = true;
        matched_bbox_j[bbox_j] = true;
        LOGI("matched,tracker_idx:%d,trackid:%lu,bbox_idx:%d,iou:%f",
             tracker_idx, trackers_[tracker_idx]->id_, bbox_idx, matched_iou);
        match_result.matched_pairs.push_back(
            std::make_pair(tracker_idx, bbox_idx));
      }
    }
  }

  for (int i = 0; i < tracker_num; i++) {
    if (!matched_tracker_i[i]) {
      int tracker_idx = tracker_idxes[i];
      match_result.unmatched_tracker_idxes.push_back(tracker_idx);
    }
  }

  for (int j = 0; j < bbox_num; j++) {
    if (!matched_bbox_j[j]) {
      int bbox_idx = det_idxes[j];
      match_result.unmatched_bbox_idxes.push_back(bbox_idx);
    }
  }

  delete[] matched_tracker_i;
  delete[] matched_bbox_j;

  return match_result;
}

void MOT::updateTrackers(const std::vector<ObjectBoxInfo> &boxes,
                         const std::vector<ModelFeatureInfo> &features) {
  std::map<uint64_t, int> trackid_idx_map;
  for (size_t i = 0; i < trackers_.size(); i++) {
    trackid_idx_map[trackers_[i]->id_] = i;
    // LOGI("trackid:%lu,idx:%d", trackers_[i]->id_, i);
  }
  std::map<int, int> paired_track_idxes;
  for (size_t i = 0; i < trackers_.size(); i++) {
    uint64_t pair_trackid = trackers_[i]->getPairTrackID();
    if (pair_trackid == 0) {
      continue;
    }
    if (trackid_idx_map.count(pair_trackid) == 0) {
      LOGW("pair_trackid:%lu not found,skip", pair_trackid);
      continue;
    }
    int pair_idx = trackid_idx_map[pair_trackid];
    paired_track_idxes[i] = pair_idx;
  }

  // update matched trackers
  std::map<uint64_t, int> matched_trackid_flag;
  for (size_t i = 0; i < boxes.size(); i++) {
    uint64_t trackid = det_track_ids_[i];
    if (trackid == 0) continue;
    if (trackid_idx_map.count(trackid) == 0) {
      LOGE("trackid not found:%lu", trackid);
      assert(false);
    }
    int idx = trackid_idx_map[trackid];
    trackers_[idx]->update(current_frame_id_, kalman_filter_, &boxes[i],
                           tracker_config_);
    matched_trackid_flag[trackid] = 1;
  }

  // create new trackers
  for (size_t i = 0; i < boxes.size(); i++) {
    uint64_t trackid = det_track_ids_[i];
    if (trackid != 0 ||
        boxes[i].score < tracker_config_.track_init_score_thresh_) {
      LOGI("boxid:%d,trackid:%lu,has been matched,score:%f,thresh:%f,skip", i,
           trackid, boxes[i].score, tracker_config_.track_init_score_thresh_);
      continue;
    }
    id_counter_++;
    uint64_t new_id = id_counter_;
    std::shared_ptr<KalmanTracker> tracker = std::make_shared<KalmanTracker>(
        current_frame_id_, kalman_filter_, new_id, boxes[i], img_width_,
        img_height_);
    trackers_.emplace_back(tracker);

    LOGI("create new tracker:%lu,box_id:%d,objtype:%d,x1:%f,y1:%f,x2:%f,y2:%f",
         new_id, i, boxes[i].object_type, boxes[i].x1, boxes[i].y1, boxes[i].x2,
         boxes[i].y2);
    int pair_idx = pair_obj_idxes_[i];
    if (pair_idx != -1) {
      uint64_t pair_trackid = det_track_ids_[pair_idx];
      if (pair_trackid != 0 && trackid_idx_map.count(pair_trackid) != 0) {
        int pair_track_idx = trackid_idx_map[pair_trackid];
        std::shared_ptr<KalmanTracker> &pair_track = trackers_[pair_track_idx];
        if (pair_track->unmatched_times_ == 0 &&
            pair_track->status_ == TrackStatus::TRACKED &&
            boxes[i].score > 0.5) {
          tracker->status_ = TrackStatus::TRACKED;
          LOGI("confirm track directly ,track:%lu,pair:%lu", tracker->id_,
               pair_track->id_);
        }
      }
    }
    det_track_ids_[i] = new_id;
    trackid_idx_map[new_id] = trackers_.size() - 1;
    matched_trackid_flag[new_id] = 1;
    LOGI("add new tracker:%lu,idx:%d", new_id, trackid_idx_map[new_id]);
  }
  // update paired trackers
  std::map<uint64_t, uint64_t> pair_track_ids = getPairTrackIds();
  for (auto &p : pair_track_ids) {
    uint64_t trackid = p.first;
    uint64_t pair_trackid = p.second;
    if (trackid_idx_map.count(trackid) == 0 ||
        trackid_idx_map.count(pair_trackid) == 0) {
      LOGW("trackid or pair_trackid not found,skip,track:%lu,pair:%lu", trackid,
           pair_trackid);
      continue;
    }
    int track_a_idx = trackid_idx_map[trackid];
    int track_b_idx = trackid_idx_map[pair_trackid];
    LOGI("update paired trackers,track:%lu,idx:%d,pair:%lu,idx:%d", trackid,
         track_a_idx, pair_trackid, track_b_idx);
    auto &track_a = trackers_[track_a_idx];
    auto &track_b = trackers_[track_b_idx];
    if (track_a->unmatched_times_ == 0 && track_b->unmatched_times_ == 0) {
      track_a->updatePairInfo(track_b.get());
    } else if (track_a->unmatched_times_ == 0 &&
               track_b->unmatched_times_ != 0) {
      track_b->falseUpdateFromPair(current_frame_id_, kalman_filter_,
                                   track_a.get(), tracker_config_);
      matched_trackid_flag[pair_trackid] = 1;
    } else if (track_a->unmatched_times_ != 0 &&
               track_b->unmatched_times_ == 0) {
      track_a->falseUpdateFromPair(current_frame_id_, kalman_filter_,
                                   track_b.get(), tracker_config_);
      matched_trackid_flag[trackid] = 1;
    }
  }
  // update unmatched trackers
  for (size_t i = 0; i < trackers_.size(); i++) {
    if (matched_trackid_flag.count(trackers_[i]->id_) == 0) {
      LOGI("update unmatched tracker:%lu", trackers_[i]->id_);
      trackers_[i]->update(current_frame_id_, kalman_filter_, nullptr,
                           tracker_config_);
    }
  }
  // erase trackers with unmatched_times_ >
  // max_unmatched_times_for_bbox_matching
  for (auto it = trackers_.begin(); it != trackers_.end();) {
    if ((*it)->status_ == TrackStatus::REMOVED) {
      LOGI("erase tracker:%lu", (*it)->id_);
      resetPairTrackerOfRemovedTracker((*it)->id_);
      it = trackers_.erase(it);
    } else {
      it++;
    }
  }
}

std::map<uint64_t, uint64_t> MOT::getPairTrackIds() {
  std::map<uint64_t, uint64_t> pair_track_ids;
  auto swap_trackid = [](uint64_t &a, uint64_t &b) {
    if (b < a) {
      uint64_t tmp = a;
      a = b;
      b = tmp;
    }
  };
  for (size_t i = 0; i < pair_obj_idxes_.size(); i++) {
    if (pair_obj_idxes_[i] == -1) {
      continue;
    }
    uint64_t trackid = det_track_ids_[i];
    uint64_t pair_trackid = det_track_ids_[pair_obj_idxes_[i]];
    if (trackid == 0 || pair_trackid == 0) {
      LOGW(
          "trackid or pair_trackid is "
          "0,track:%lu,pair:%lu,boxid:%d,pair_boxid:%d",
          trackid, pair_trackid, i, pair_obj_idxes_[i]);
      continue;
    }
    swap_trackid(trackid, pair_trackid);
    if (pair_track_ids.count(trackid) &&
        pair_track_ids[trackid] != pair_trackid) {
      LOGW("found duplicate pair,track:%lu,src_pair:%lu,new_pair:%lu", trackid,
           pair_track_ids[trackid], pair_trackid);
      continue;
    }
    pair_track_ids[trackid] = pair_trackid;
  }
  for (auto &track : trackers_) {
    uint64_t trackid = track->id_;
    uint64_t pair_trackid = track->getPairTrackID();
    if (track->unmatched_times_ == 0 && pair_trackid != 0) {
      swap_trackid(trackid, pair_trackid);
      if (pair_track_ids.count(trackid) == 0) {
        LOGI("add extra pair track:%lu,pair:%lu", trackid, pair_trackid);
        pair_track_ids[trackid] = pair_trackid;
      }
    }
  }
  return pair_track_ids;
}

void MOT::resetPairTrackerOfRemovedTracker(uint64_t tracker_id) {
  if (tracker_id == 0) {
    return;
  }
  for (auto &p : trackers_) {
    if (p->getPairTrackID() == tracker_id) {
      p->resetPairInfo();
    }
  }
}