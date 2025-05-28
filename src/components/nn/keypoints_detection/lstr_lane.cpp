#include "keypoints_detection/lstr_lane.hpp"

#include <cstdint>
#include <memory>
#include <sstream>
#include <vector>

#include "utils/detection_helper.hpp"
#include "utils/tdl_log.hpp"

const int NUM_SLICES = 100;
const float CUT_SLICES_RATIO = 0.25f;      // cutting and straightening ratio
const float DETECTION_UPPER_LIMIT = 0.6f;  // detection range upper limit ratio
const float DETECTION_LOWER_LIMIT = 0.8f;  // detection range lower limit ratio

LstrLane::LstrLane() {
  net_param_.model_config.mean = {1.79226 / 0.014598, 1.752097 / 0.0150078,
                                  1.48022 / 0.0142201};
  net_param_.model_config.std = {1.0 / 0.014598, 1.0 / 0.0150078,
                                 1.0 / 0.0142201};

  keep_aspect_ratio_ = false;
  net_param_.model_config.rgb_order = "rgb";
}
LstrLane::~LstrLane() {}

float LstrLane::gen_x_by_y(float ys, std::vector<float> &point_line) {
  return point_line[2] / ((ys - point_line[3]) * (ys - point_line[3])) +
         point_line[4] / (ys - point_line[3]) + point_line[5] +
         point_line[6] * ys - point_line[7];
}

int32_t LstrLane::outputParse(
    const std::vector<std::shared_ptr<BaseImage>> &images,
    std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) {
  std::string input_tensor_name = net_->getInputNames()[0];
  TensorInfo input_tensor = net_->getTensorInfo(input_tensor_name);

  LOGI("outputParse,batch size:%d,input shape:%d,%d,%d,%d", images.size(),
       input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2],
       input_tensor.shape[3]);

  std::string out_conf_name = net_->getOutputNames()[0];
  std::string out_feature_name = net_->getOutputNames()[1];

  TensorInfo out_feature_info = net_->getTensorInfo(out_feature_name);
  std::shared_ptr<BaseTensor> out_feature_tensor =
      net_->getOutputTensor(out_feature_name);

  TensorInfo out_conf_info = net_->getTensorInfo(out_conf_name);
  std::shared_ptr<BaseTensor> out_conf_tensor =
      net_->getOutputTensor(out_conf_name);

  for (uint32_t b = 0; b < (uint32_t)input_tensor.shape[0]; b++) {
    uint32_t image_width = images[b]->getWidth();
    uint32_t image_height = images[b]->getHeight();

    float *out_conf = out_conf_tensor->getBatchPtr<float>(b);
    float *out_feature = out_feature_tensor->getBatchPtr<float>(b);
    std::shared_ptr<ModelBoxLandmarkInfo> obj =
        std::make_shared<ModelBoxLandmarkInfo>();

    obj->image_width = image_width;
    obj->image_height = image_height;

    if (export_feature) {
      int feature_size = 7 * 2 + 7 * 8;
      obj->feature.resize(feature_size);
      memcpy(obj->feature.data(), out_feature, sizeof(float) * 7 * 8);
      memcpy(obj->feature.data() + 7 * 8, out_conf, sizeof(float) * 7 * 2);
    }

    std::vector<std::vector<float>> point_map;
    std::vector<float> lane_dis;
    for (int i = 0; i < 7; i++) {
      int cls = 0;
      if (out_conf[i * 2 + 1] > out_conf[i * 2]) {
        cls = 1;
      }
      if (cls == 1) {
        std::vector<float> line_info(out_feature + i * 8,
                                     out_feature + i * 8 + 8);
        point_map.push_back(line_info);
        float cur_dis = gen_x_by_y(1.0, point_map.back());
        cur_dis = (cur_dis - 0.5) * obj->image_width;
        lane_dis.push_back(cur_dis);
      }
    }
    std::vector<int> sort_index(point_map.size(), 0);
    for (int i = 0; i != point_map.size(); i++) {
      sort_index[i] = i;
    }
    std::sort(sort_index.begin(), sort_index.end(),
              [&](const int &a, const int &b) {
                return (lane_dis[a] < lane_dis[b]);
              });
    std::vector<int> final_index;
    for (int i = 0; i != sort_index.size(); i++) {
      if (lane_dis[sort_index[i]] < 0) {
        if (i == sort_index.size() - 1 || lane_dis[sort_index[i + 1]] > 0) {
          if (point_map[sort_index[i]][1] - point_map[sort_index[i]][0] > 0.2)
            final_index.push_back(sort_index[i]);
        }

      } else {
        if (point_map[sort_index[i]][1] - point_map[sort_index[i]][0] > 0.2) {
          final_index.push_back(sort_index[i]);
          break;
        }
      }
    }

    for (int i = 0; i < final_index.size(); i++) {
      ObjectBoxLandmarkInfo lane_landmark;
      float upper = std::min(1.0f, point_map[final_index[i]][1]);
      float lower = std::max(0.0f, point_map[final_index[i]][0]);
      float slice = (upper - lower) / NUM_SLICES;
      float true_y1 = lower + NUM_SLICES * CUT_SLICES_RATIO * slice;
      float true_x1 = gen_x_by_y(true_y1, point_map[final_index[i]]);
      float true_y2 = lower + NUM_SLICES * (1 - CUT_SLICES_RATIO) * slice;
      float true_x2 = gen_x_by_y(true_y2, point_map[final_index[i]]);

      lane_landmark.landmarks_y.push_back(DETECTION_UPPER_LIMIT *
                                          obj->image_height);
      lane_landmark.landmarks_x.push_back(
          (true_x1 + (DETECTION_UPPER_LIMIT - true_y1) * (true_x2 - true_x1) /
                         (true_y2 - true_y1)) *
          obj->image_width);
      lane_landmark.landmarks_y.push_back(DETECTION_LOWER_LIMIT *
                                          obj->image_height);
      lane_landmark.landmarks_x.push_back(
          (true_x1 + (DETECTION_LOWER_LIMIT - true_y1) * (true_x2 - true_x1) /
                         (true_y2 - true_y1)) *
          obj->image_width);

      obj->box_landmarks.push_back(lane_landmark);
    }

    out_datas.push_back(obj);
  }

  return 0;
}

void LstrLane::setExportFeature(int flag) { export_feature = flag; }