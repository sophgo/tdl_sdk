
#include "smooth_keypoints.hpp"
#include <map>
#include <vector>
#include "core/core/cvtdl_core_types.h"
#include "core/core/cvtdl_errno.h"

SmoothKeypoints::SmoothKeypoints() {
  smooth_param_.image_width = 1920;
  smooth_param_.image_height = 1080;
  smooth_param_.fc_d = 0.1;
  smooth_param_.fc_min = 0.1;
  smooth_param_.beta = 0.05;
  smooth_param_.thres_mult = 0.3;
  smooth_param_.te = 1.0f;
  smooth_param_.smooth_frames = 5;
  smooth_param_.smooth_type = 0;
}

SmoothAlgParam SmoothKeypoints::get_algparam() { return smooth_param_; }
void SmoothKeypoints::set_algparam(SmoothAlgParam smooth_param) {
  smooth_param_.image_width = smooth_param.image_width;
  smooth_param_.image_height = smooth_param.image_height;
  smooth_param_.fc_d = smooth_param.fc_d;
  smooth_param_.fc_min = smooth_param.fc_min;
  smooth_param_.beta = smooth_param.beta;
  smooth_param_.thres_mult = smooth_param.thres_mult;
  smooth_param_.te = smooth_param.te;
  smooth_param_.smooth_frames = smooth_param.smooth_frames;
  smooth_param_.smooth_type = smooth_param.smooth_type;
}

int SmoothKeypoints::smooth(cvtdl_object_t* obj_meta) {
  std::map<int, int> track_index;
  std::vector<int> new_index;
  for (uint32_t i = 0; i < obj_meta->size; i++) {
    if (obj_meta->info[i].track_state == cvtdl_trk_state_type_t::CVI_TRACKER_NEW) {
      new_index.push_back(i);

    } else {
      track_index[obj_meta->info[i].unique_id] = i;
    }
  }

  for (auto it = muti_keypoints.begin(); it != muti_keypoints.end();) {
    if (track_index.count(it->uid) == 0) {
      it->unmatched_times += 1;

      if (it->unmatched_times == it->MAX_UNMATCHED_TIME) {
        it = muti_keypoints.erase(it);
      } else {
        it++;
      }

    } else {
      it->smooth_keypoints(&obj_meta->info[track_index[it->uid]].pedestrian_properity->pose_17);
      it->unmatched_times = 0;
      it++;
    }
  }

  for (uint32_t i = 0; i < new_index.size(); i++) {
    HumanKeypoints hk(obj_meta->info[new_index[i]].unique_id, smooth_param_);

    hk.smooth_keypoints(&obj_meta->info[new_index[i]].pedestrian_properity->pose_17);

    muti_keypoints.push_back(hk);
  }

  return CVI_TDL_SUCCESS;
}