#include "face/face_ddfa.hpp"
#include <log/Logger.hpp>
using nncompact::Net;
using nncompact::Tensor;

bmStatus_t FaceDDFA::setup() {
  setup_net(net_param_);
  return BM_COMMON_SUCCESS;
}

bmStatus_t FaceDDFA::detect(const std::vector<cv::Mat> &images,
                            std::vector<std::vector<FaceRect>> &faceRects) {
  timer_.store_timestamp("pack data");
  std::vector<cv::Mat> crop_faces;
  for (int i = 0; i < images.size(); i++) {
    crop(images[i], faceRects[i], crop_faces);
  }
  timer_.store_timestamp("pack data");
  auto img_iter = crop_faces.cbegin();
  auto left_size = crop_faces.size();

#ifdef TIME_PRINT
  timer_.store_timestamp("total detection");
#endif
  std::vector<FacePose> poses;
  while (left_size > 0) {
    int batch_size = get_fit_n(left_size);

#ifdef TIME_PRINT
    timer_.store_timestamp("preprocess");
#endif

    BM_CHECK_STATUS(preprocess_opencv(img_iter, batch_size));

#ifdef TIME_PRINT
    timer_.store_timestamp("preprocess");
    timer_.store_timestamp("forward");
#endif

    BM_CHECK_STATUS(forward());

#ifdef TIME_PRINT
    timer_.store_timestamp("forward");
    timer_.store_timestamp("postprocess");
#endif

    BM_CHECK_STATUS(postprocess(img_iter, poses));
    img_iter += batch_size;
    left_size -= batch_size;

#ifdef TIME_PRINT
    timer_.store_timestamp("postprocess");
#endif
  }
  for (int i = 0; i < faceRects.size(); i++) {
    std::vector<FaceRect> &rects = faceRects[i];
    for (int j = 0; j < rects.size(); j++) {
      FacePose &pose = poses[i * rects.size() + j];
      rects[j].facepose = pose;
    }
  }

#ifdef TIME_PRINT
  timer_.store_timestamp("total detection");
  timer_.show();
  timer_.clear();
#endif

  return BM_COMMON_SUCCESS;
}

void FaceDDFA::crop(const cv::Mat &ori_image,
                    const std::vector<FaceRect> &rects,
                    std::vector<cv::Mat> &crop_faces) {
  for (int i = 0; i < rects.size(); i++) {
    const FaceRect &rect = rects[i];
    float x1 = rect.x1;
    float x2 = rect.x2;
    float y1 = rect.y1;
    float y2 = rect.y2;
    float old_size = (x2 - x1 + y2 - y1) / 2.0;
    float center_x = x2 - (x2 - x1) / 2.0;
    float center_y = y2 - (y2 - y1) / 2.0 + old_size * 0.14;
    int size = old_size * 1.58;
    int square_x = center_x - size / 2.0;
    int square_y = center_y - size / 2.0;
    cv::Rect crop_rect(square_x, square_y, size, size);
    cv::Rect inside_rect =
        crop_rect & cv::Rect(0, 0, ori_image.cols, ori_image.rows);
    cv::Mat crop_face = ori_image(inside_rect);
    crop_faces.push_back(crop_face);
  }
}

void FaceDDFA::parse_pose(const float *output_data, const int len,
                          FacePose &pose) {
  float *param = new float[len];
  for (int i = 0; i < len; ++i) {
    param[i] = output_data[i] * param_std_[i] + param_mean_[i];
  }
  // reshape param to [3, 4] ...
  // matrix to angle
  float *R1 = param;
  float *R2 = param + 4;
  float R1_norm = sqrt(R1[0] * R1[0] + R1[1] * R1[1] + R1[2] * R1[2]);
  float R2_norm = sqrt(R2[0] * R2[0] + R2[1] * R2[1] + R2[2] * R2[2]);
  for (int i = 0; i < 3; ++i) {
    R1[i] /= R1_norm;
    R2[i] /= R2_norm;
  }
  float R20 = R1[1] * R2[2] - R1[2] * R2[1];
  if (R20 != -1 || R20 != 1) {
    pose.yaw = asin(R20);
    float R21 = R1[2] * R2[0] - R1[0] * R2[2];
    float R22 = R1[0] * R2[1] - R1[1] * R2[0];
    pose.pitch = atan2(R21 / cos(pose.yaw), R22 / cos(pose.yaw));
    pose.roll = atan2(R2[0] / cos(pose.yaw), R1[0] / cos(pose.yaw));
  } else {
    pose.roll = 0.;
    if (R1[2] == -1) {
      pose.yaw = M_PI / 2.0;
      pose.pitch = atan2(R1[1], R1[2]);
    } else {
      pose.yaw = -M_PI / 2.0;
      pose.pitch = atan2(-R1[1], -R1[2]);
    }
  }
  pose.pitch *= 180. / M_PI;
  pose.yaw *= 180. / M_PI;
  pose.roll *= 180. / M_PI;
  delete[] param;
}

bmStatus_t FaceDDFA::postprocess(std::vector<cv::Mat>::const_iterator &img_iter,
                                 std::vector<FacePose> &poses) {
  int batch_n = get_input_n();

  net_->update_output_tensors();
  nncompact::Tensor *output_tensor =
      net_->get_output_tensor(output_layer_).get();
  std::vector<int> shape = output_tensor->get_shape();
  int len = 1;
  for (int i = 1; i < shape.size(); i++)
    len *= shape[i];
  float *out = output_tensor->get_data();
  for (int i = 0; i < batch_n; i++) {
    const float *output_data = out + i * output_tensor->batch_num_elems();
    FacePose pose;
    parse_pose(output_data, len, pose);
    poses.push_back(pose);
  }

  return BM_COMMON_SUCCESS;
}
