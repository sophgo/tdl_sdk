#include "cvi_kit.h"
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include "cvi_tdl_log.hpp"
#include "opencv2/opencv.hpp"
#include "utils/clip_postprocess.hpp"
#include "utils/token.hpp"
#ifndef NO_OPENCV

using namespace cvitdl;
std::vector<cv::Scalar> color = {cv::Scalar(51, 153, 255), cv::Scalar(0, 153, 76),
                                 cv::Scalar(255, 215, 0), cv::Scalar(255, 128, 0),
                                 cv::Scalar(0, 255, 0)};

int color_map[17] = {0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 4, 3, 4, 3, 4, 3};
int line_map[19] = {4, 4, 3, 3, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0};
int skeleton[19][2] = {{15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12},
                       {5, 6},   {5, 7},   {6, 8},   {7, 9},   {8, 10},  {1, 2},  {0, 1},
                       {0, 2},   {1, 3},   {2, 4},   {3, 5},   {4, 6}};

CVI_S32 CVI_TDL_ShowLanePoints(VIDEO_FRAME_INFO_S *bg, cvtdl_lane_t *lane_meta, char *save_path) {
  bg->stVFrame.pu8VirAddr[0] =
      (CVI_U8 *)CVI_SYS_Mmap(bg->stVFrame.u64PhyAddr[0], bg->stVFrame.u32Length[0]);

  cv::Mat img_rgb(bg->stVFrame.u32Height, bg->stVFrame.u32Width, CV_8UC3,
                  bg->stVFrame.pu8VirAddr[0], bg->stVFrame.u32Stride[0]);

  int det_num = lane_meta->size;

  for (int i = 0; i < det_num; i++) {
    int x0 = (int)lane_meta->lane[i].x[0];
    int y0 = (int)lane_meta->lane[i].y[0];
    int x1 = (int)lane_meta->lane[i].x[1];
    int y1 = (int)lane_meta->lane[i].y[1];

    cv::line(img_rgb, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(0, 255, 0), 3);
  }

  cv::imwrite(save_path, img_rgb);
  CVI_SYS_Munmap((void *)bg->stVFrame.pu8VirAddr[0], bg->stVFrame.u32Length[0]);
  return CVI_SUCCESS;
}
cv::Scalar gen_random_color(uint64_t seed, int min) {
  float scale = (256. - (float)min) / 256.;
  srand((uint32_t)seed);

  int r = (int)((floor(((float)rand() / (RAND_MAX)) * 256.)) * scale) + min;
  int g = (int)((floor(((float)rand() / (RAND_MAX)) * 256.)) * scale) + min;
  int b = (int)((floor(((float)rand() / (RAND_MAX)) * 256.)) * scale) + min;

  return cv::Scalar(r, g, b);
}

DLL_EXPORT CVI_S32 CVI_TDL_Draw_ADAS(cvitdl_app_handle_t app_handle, VIDEO_FRAME_INFO_S *bg,
                                     char *save_path) {
  static const char *enumStr[] = {"N", "S", "C", "D"};
  int g_count = 0;
  bg->stVFrame.pu8VirAddr[0] =
      (CVI_U8 *)CVI_SYS_Mmap(bg->stVFrame.u64PhyAddr[0], bg->stVFrame.u32Length[0]);

  cv::Mat img_rgb(bg->stVFrame.u32Height, bg->stVFrame.u32Width, CV_8UC3,
                  bg->stVFrame.pu8VirAddr[0], bg->stVFrame.u32Stride[0]);

  cvtdl_object_t *obj_meta = &app_handle->adas_info->last_objects;
  cvtdl_tracker_t *track_meta = &app_handle->adas_info->last_trackers;
  cvtdl_lane_t *lane_meta = &app_handle->adas_info->lane_meta;

  cv::Scalar box_color;
  for (uint32_t oid = 0; oid < obj_meta->size; oid++) {
    // if (track_meta->info[oid].state == CVI_TRACKER_NEW) {
    //     box_color = cv::Scalar(0, 255, 0);
    // } else if (track_meta->info[oid].state == CVI_TRACKER_UNSTABLE) {
    //     box_color = cv::Scalar(105, 105, 105);
    // } else {  // CVI_TRACKER_STABLE
    //     box_color = gen_random_color(obj_meta->info[oid].unique_id, 64);
    // }
    box_color = gen_random_color(obj_meta->info[oid].unique_id, 64);

    cv::Point top_left((int)obj_meta->info[oid].bbox.x1, (int)obj_meta->info[oid].bbox.y1);
    cv::Point bottom_right((int)obj_meta->info[oid].bbox.x2, (int)obj_meta->info[oid].bbox.y2);

    cv::rectangle(img_rgb, top_left, bottom_right, box_color, 2);

    char txt_info[256];
    snprintf(txt_info, sizeof(txt_info), "[%d][%s]S:%.1f,V:%.1f", obj_meta->info[oid].classes,
             enumStr[obj_meta->info[oid].adas_properity.state],
             obj_meta->info[oid].adas_properity.dis, obj_meta->info[oid].adas_properity.speed);

    if (obj_meta->info[oid].adas_properity.state != 0) box_color = cv::Scalar(0, 0, 255);

    cv::putText(img_rgb, txt_info, cv::Point(top_left.x, top_left.y - 10), 0, 0.5, box_color, 1);
  }

  if (app_handle->adas_info->det_type) {
    int size = bg->stVFrame.u32Width >= 1080 ? 6 : 3;

    for (int i = 0; i < lane_meta->size; i++) {
      int x1 = (int)lane_meta->lane[i].x[0];
      int y1 = (int)lane_meta->lane[i].y[0];
      int x2 = (int)lane_meta->lane[i].x[1];
      int y2 = (int)lane_meta->lane[i].y[1];

      cv::line(img_rgb, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), size);
    }

    char lane_info[64];

    if (app_handle->adas_info->lane_state == 0) {
      strcpy(lane_info, "NORMAL");
      box_color = cv::Scalar(0, 255, 0);
    } else {
      strcpy(lane_info, "LANE DEPARTURE WARNING !");
      box_color = cv::Scalar(0, 0, 255);
    }

    cv::putText(img_rgb, lane_info,
                cv::Point((int)(0.3 * bg->stVFrame.u32Width), (int)(0.8 * bg->stVFrame.u32Height)),
                0, size / 3, box_color, size / 3);

    snprintf(lane_info, sizeof(lane_info), "%d", g_count);
    cv::putText(img_rgb, lane_info, cv::Point(10, 50), 0, size / 3, cv::Scalar(0, 255, 0),
                size / 3);

    g_count++;
  }

  cv::imwrite(save_path, img_rgb);
  CVI_SYS_Munmap((void *)bg->stVFrame.pu8VirAddr[0], bg->stVFrame.u32Length[0]);

  return CVI_SUCCESS;
}

DLL_EXPORT CVI_S32 CVI_TDL_ShowKeypointsAndText(VIDEO_FRAME_INFO_S *bg, cvtdl_object_t *obj_meta,
                                                char *save_path, char *text, float score) {
  bg->stVFrame.pu8VirAddr[0] =
      (CVI_U8 *)CVI_SYS_Mmap(bg->stVFrame.u64PhyAddr[0], bg->stVFrame.u32Length[0]);

  cv::Mat img_rgb(bg->stVFrame.u32Height, bg->stVFrame.u32Width, CV_8UC3,
                  bg->stVFrame.pu8VirAddr[0], bg->stVFrame.u32Stride[0]);

  for (uint32_t i = 0; i < obj_meta->size; i++) {
    for (uint32_t j = 0; j < 17; j++) {
      // printf("j:%d\n",j);
      if (obj_meta->info[i].pedestrian_properity->pose_17.score[i] < score) {
        continue;
      }
      int x = (int)obj_meta->info[i].pedestrian_properity->pose_17.x[j];
      int y = (int)obj_meta->info[i].pedestrian_properity->pose_17.y[j];
      cv::circle(img_rgb, cv::Point(x, y), 5, color[color_map[j]], -1);
    }

    for (uint32_t k = 0; k < 19; k++) {
      // printf("k:%d\n",k);

      int kps1 = skeleton[k][0];
      int kps2 = skeleton[k][1];
      if (obj_meta->info[i].pedestrian_properity->pose_17.score[kps1] < score ||
          obj_meta->info[i].pedestrian_properity->pose_17.score[kps2] < score)
        continue;

      int x1 = (int)obj_meta->info[i].pedestrian_properity->pose_17.x[kps1];
      int y1 = (int)obj_meta->info[i].pedestrian_properity->pose_17.y[kps1];

      int x2 = (int)obj_meta->info[i].pedestrian_properity->pose_17.x[kps2];
      int y2 = (int)obj_meta->info[i].pedestrian_properity->pose_17.y[kps2];

      cv::line(img_rgb, cv::Point(x1, y1), cv::Point(x2, y2), color[line_map[k]], 2);
    }
  }

  cv::putText(img_rgb, text, cv::Point(10, 30), cv::FONT_HERSHEY_COMPLEX, 0.5,
              cv::Scalar(0, 255, 255), 1);

  cv::imwrite(save_path, img_rgb);
  CVI_SYS_Munmap((void *)bg->stVFrame.pu8VirAddr[0], bg->stVFrame.u32Length[0]);
  return CVI_SUCCESS;
}

CVI_S32 CVI_TDL_ShowKeypoints(VIDEO_FRAME_INFO_S *bg, cvtdl_object_t *obj_meta, float score,
                              char *save_path) {
  bg->stVFrame.pu8VirAddr[0] =
      (CVI_U8 *)CVI_SYS_Mmap(bg->stVFrame.u64PhyAddr[0], bg->stVFrame.u32Length[0]);

  cv::Mat img_rgb(bg->stVFrame.u32Height, bg->stVFrame.u32Width, CV_8UC3,
                  bg->stVFrame.pu8VirAddr[0], bg->stVFrame.u32Stride[0]);

  for (uint32_t i = 0; i < obj_meta->size; i++) {
    for (uint32_t j = 0; j < 17; j++) {
      if (obj_meta->info[i].pedestrian_properity->pose_17.score[i] < score) {
        continue;
      }
      int x = (int)obj_meta->info[i].pedestrian_properity->pose_17.x[j];
      int y = (int)obj_meta->info[i].pedestrian_properity->pose_17.y[j];
      cv::circle(img_rgb, cv::Point(x, y), 7, color[color_map[j]], -1);
    }

    for (uint32_t k = 0; k < 19; k++) {
      int kps1 = skeleton[k][0];
      int kps2 = skeleton[k][1];
      if (obj_meta->info[i].pedestrian_properity->pose_17.score[kps1] < score ||
          obj_meta->info[i].pedestrian_properity->pose_17.score[kps2] < score)
        continue;

      int x1 = (int)obj_meta->info[i].pedestrian_properity->pose_17.x[kps1];
      int y1 = (int)obj_meta->info[i].pedestrian_properity->pose_17.y[kps1];

      int x2 = (int)obj_meta->info[i].pedestrian_properity->pose_17.x[kps2];
      int y2 = (int)obj_meta->info[i].pedestrian_properity->pose_17.y[kps2];

      cv::line(img_rgb, cv::Point(x1, y1), cv::Point(x2, y2), color[line_map[k]], 2);
    }
  }

  cv::imwrite(save_path, img_rgb);
  CVI_SYS_Munmap((void *)bg->stVFrame.pu8VirAddr[0], bg->stVFrame.u32Length[0]);
  return CVI_SUCCESS;
}

CVI_S32 CVI_TDL_ShowDetectionBox(cvtdl_object_t *obj_meta, const char *image_path,
                              char *save_path) {
  cv::Mat image = cv::imread(image_path);
  if (image.empty()) {
      printf("Could not open or find the image!\n");
      return CVI_FAILURE;
  }

  for (uint32_t i = 0; i < obj_meta->size; i++) {
      float x1 = obj_meta->info[i].bbox.x1;
      float y1 = obj_meta->info[i].bbox.y1;
      float x2 = obj_meta->info[i].bbox.x2;
      float y2 = obj_meta->info[i].bbox.y2;
      int class_id = obj_meta->info[i].classes;
      float confidence = obj_meta->info[i].bbox.score;
      cv::rectangle(image, cv::Point(int(x1), int(y1)), cv::Point(int(x2), int(y2)), cv::Scalar(255, 0, 0), 2);

      std::string label = "Class: " + std::to_string(class_id) + " Confidence: " + std::to_string(confidence).substr(0, 4);
      cv::putText(image, label, cv::Point(int(x1), int(y1) - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);   
  }

  cv::imwrite(save_path, image);
  return CVI_SUCCESS;
}

CVI_S32 CVI_TDL_SavePicture(VIDEO_FRAME_INFO_S *bg, char *save_path) {
  CVI_U8 *r_plane = (CVI_U8 *)CVI_SYS_Mmap(bg->stVFrame.u64PhyAddr[0], bg->stVFrame.u32Length[0]);
  CVI_U8 *g_plane = (CVI_U8 *)CVI_SYS_Mmap(bg->stVFrame.u64PhyAddr[1], bg->stVFrame.u32Length[1]);
  CVI_U8 *b_plane = (CVI_U8 *)CVI_SYS_Mmap(bg->stVFrame.u64PhyAddr[2], bg->stVFrame.u32Length[2]);

  cv::Mat r_mat(bg->stVFrame.u32Height, bg->stVFrame.u32Width, CV_8UC1, r_plane);
  cv::Mat g_mat(bg->stVFrame.u32Height, bg->stVFrame.u32Width, CV_8UC1, g_plane);
  cv::Mat b_mat(bg->stVFrame.u32Height, bg->stVFrame.u32Width, CV_8UC1, b_plane);

  std::vector<cv::Mat> channels = {r_mat, g_mat, b_mat};
  cv::Mat img_rgb;
  cv::merge(channels, img_rgb);

  cv::imwrite(save_path, img_rgb);

  CVI_SYS_Munmap(r_plane, bg->stVFrame.u32Length[0]);
  CVI_SYS_Munmap(g_plane, bg->stVFrame.u32Length[1]);
  CVI_SYS_Munmap(b_plane, bg->stVFrame.u32Length[2]);
  return CVI_SUCCESS;
}

CVI_S32 CVI_TDL_Cal_Similarity(cvtdl_feature_t feature, cvtdl_feature_t feature1,
                               float *similarity) {
  cv::Mat mat_feature(feature.size, 1, CV_8SC1);
  cv::Mat mat_feature1(feature1.size, 1, CV_8SC1);
  memcpy(mat_feature.data, feature.ptr, feature.size);
  memcpy(mat_feature1.data, feature1.ptr, feature1.size);
  mat_feature.convertTo(mat_feature, CV_32FC1, 1.);
  mat_feature1.convertTo(mat_feature1, CV_32FC1, 1.);
  *similarity = mat_feature.dot(mat_feature1) / (cv::norm(mat_feature) * cv::norm(mat_feature1));
  return CVI_SUCCESS;
}

CVI_S32 CVI_TDL_Set_MaskOutlinePoint(VIDEO_FRAME_INFO_S *frame, cvtdl_object_t *obj_meta) {
  int proto_h = obj_meta->mask_height;
  int proto_w = obj_meta->mask_width;
  for (uint32_t i = 0; i < obj_meta->size; i++) {
    cv::Mat src(proto_h, proto_w, CV_8UC1, obj_meta->info[i].mask_properity->mask,
                proto_w * sizeof(uint8_t));

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    // search for contours
    cv::findContours(src, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    // find the longest contour
    int longest_index = -1;
    size_t max_length = 0;
    for (size_t i = 0; i < contours.size(); i++) {
      if (contours[i].size() > max_length) {
        max_length = contours[i].size();
        longest_index = i;
      }
    }
    if (longest_index >= 0 && max_length >= 1) {
      float ratio_height = (proto_h / static_cast<float>(frame->stVFrame.u32Height));
      float ratio_width = (proto_w / static_cast<float>(frame->stVFrame.u32Width));
      int source_y_offset, source_x_offset;
      if (ratio_height > ratio_width) {
        source_x_offset = 0;
        source_y_offset = (proto_h - frame->stVFrame.u32Height * ratio_width) / 2;
      } else {
        source_x_offset = (proto_w - frame->stVFrame.u32Width * ratio_height) / 2;
        source_y_offset = 0;
      }
      int source_region_height = proto_h - 2 * source_y_offset;
      int source_region_width = proto_w - 2 * source_x_offset;
      // calculate scaling factor
      float height_scale =
          static_cast<float>(frame->stVFrame.u32Height) / static_cast<float>(source_region_height);
      float width_scale =
          static_cast<float>(frame->stVFrame.u32Width) / static_cast<float>(source_region_width);
      obj_meta->info[i].mask_properity->mask_point_size = max_length;
      obj_meta->info[i].mask_properity->mask_point =
          (float *)malloc(2 * max_length * sizeof(float));

      size_t j = 0;
      for (const auto &point : contours[longest_index]) {
        obj_meta->info[i].mask_properity->mask_point[2 * j] =
            (point.x - source_x_offset) * width_scale;
        obj_meta->info[i].mask_properity->mask_point[2 * j + 1] =
            (point.y - source_y_offset) * height_scale;
        j++;
      }
    }
  }
  return CVI_SUCCESS;
}

CVI_S32 CVI_TDL_Set_Occlusion_Laplacian(VIDEO_FRAME_INFO_S *frame,
                                        cvtdl_occlusion_meta_t *occlusion_meta) {
  static std::vector<int> occlusionStates;

  frame->stVFrame.pu8VirAddr[0] =
      (CVI_U8 *)CVI_SYS_MmapCache(frame->stVFrame.u64PhyAddr[0], frame->stVFrame.u32Length[0]);
  cv::Mat cur_frame(frame->stVFrame.u32Height, frame->stVFrame.u32Width, CV_8UC3,
                    frame->stVFrame.pu8VirAddr[0], frame->stVFrame.u32Stride[0]);
  int frame_h = cur_frame.rows;
  int frame_w = cur_frame.cols;
  cvtdl_bbox_t _crop_bbox = occlusion_meta->crop_bbox;
  cv::Rect roi(int(_crop_bbox.x1 * frame_w), int(_crop_bbox.y1 * frame_h),
               int((_crop_bbox.x2 - _crop_bbox.x1) * frame_w),
               int((_crop_bbox.y2 - _crop_bbox.y1) * frame_h));
  cv::Mat sub_frame = cur_frame(roi);

  cv::Mat cur_frame_gray;

#ifdef ENABLE_CVIAI_CV_UTILS
  cvitdl::cvtColor(sub_frame, cur_frame_gray, COLOR_BGR2GRAY);
  std::cout << "cvitdl::cvtColor: " << std::endl;
#else
  cv::cvtColor(sub_frame, cur_frame_gray, cv::COLOR_BGR2GRAY);
#endif
  // use Laplacian
  cv::Mat _laplacian;
  cv::Mat laplacianAbs;

  cv::Laplacian(cur_frame_gray, _laplacian, CV_64F);

  cv::threshold(_laplacian, laplacianAbs, 0, 0, cv::THRESH_TRUNC);
  laplacianAbs = _laplacian - 2 * laplacianAbs;

  cv::Mat _laplacian_8u;
  cv::convertScaleAbs(laplacianAbs, _laplacian_8u);

  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
  // use close
  cv::Mat closed;
  cv::morphologyEx(_laplacian_8u, closed, cv::MORPH_CLOSE, kernel);

  cv::Mat binary;
  cv::Mat binary_not;
  cv::Mat labels;
  cv::threshold(closed, binary, occlusion_meta->laplacian_th, 255, cv::THRESH_BINARY);

  cv::bitwise_not(binary, binary_not);
  int num_labels = cv::connectedComponents(binary_not, labels, 4);

  // Calculate pix number of connectivity context
  std::vector<int> sizes(num_labels, 0);
  for (int i = 0; i < labels.rows; i++) {
    for (int j = 0; j < labels.cols; j++) {
      sizes[labels.at<int>(i, j)]++;
    }
  }
  int max_connected_area = (num_labels > 1) ? *std::max_element(sizes.begin() + 1, sizes.end()) : 0;
  int total_pixels = labels.rows * labels.cols;
  float occ_ratio = static_cast<float>(max_connected_area) / total_pixels;
  // std::cout << "Max connected area (excluding background): " << occ_ratio << std::endl;
  occlusion_meta->occ_score = occ_ratio;
  occlusion_meta->occ_class = (occ_ratio >= occlusion_meta->occ_ratio_th) ? 1 : 0;
  occlusionStates.push_back(occlusion_meta->occ_class);
  if (occlusionStates.size() > occlusion_meta->sensitive_th) {
    occlusionStates.erase(occlusionStates.begin());
    int occludedCount = std::accumulate(occlusionStates.begin(), occlusionStates.end(), 0);
    int pre_class = (occludedCount > occlusion_meta->sensitive_th / 2) ? 1 : 0;
    occlusion_meta->occ_class = pre_class;
  }

  CVI_SYS_Munmap((void *)frame->stVFrame.pu8VirAddr[0], frame->stVFrame.u32Length[0]);
  return CVI_SUCCESS;
}

#endif

CVI_S32 CVI_TDL_Set_ClipPostprocess(float **text_features, int text_features_num,
                                    float **image_features, int image_features_num, float **probs) {
  Eigen::MatrixXf text_features_eigen(text_features_num, 512);
  for (int i = 0; i < text_features_num; ++i) {
    for (int j = 0; j < 512; ++j) {
      text_features_eigen(i, j) = text_features[i][j];
    }
  }
  Eigen::MatrixXf image_features_eigen(image_features_num, 512);
  for (int i = 0; i < image_features_num; ++i) {
    for (int j = 0; j < 512; ++j) {
      image_features_eigen(i, j) = image_features[i][j];
    }
  }

  Eigen::MatrixXf result_eigen;
  // using clip_postprocess which can be found in utils/clip_postpostprocess.cpp
  int res = cvitdl::clip_postprocess(text_features_eigen, image_features_eigen, result_eigen);

  // providing image classification functionality.
  // using softmax after mutil 100 scale
  for (int i = 0; i < result_eigen.rows(); ++i) {
    float sum = 0.0;
    float maxVal = result_eigen.row(i).maxCoeff();
    Eigen::MatrixXf expInput = (result_eigen.row(i).array() - maxVal).exp();
    sum = expInput.sum();
    result_eigen.row(i) = expInput / sum;
  }

  for (int i = 0; i < result_eigen.rows(); i++) {
    float max_score = 0;
    for (int j = 0; j < result_eigen.cols(); j++) {
      probs[i][j] = result_eigen(i, j);
    }
  }

  if (res == 0) {
    return CVI_SUCCESS;
  }

  LOGE("Tokenization error\n");
  return CVI_FAILURE;
}

CVI_S32 CVI_TDL_NormTextFeature(cvtdl_clip_feature **text_features, int text_features_num) {
  for (int i = 0; i < text_features_num; ++i) {
    Eigen::Map<Eigen::VectorXf> text_features_eigen(text_features[i]->out_feature,
                                                    text_features[0]->feature_dim);
    float norm = text_features_eigen.norm();
    if (norm != 0.0f) {
      text_features_eigen /= norm;
    }
  }
  return CVI_SUCCESS;
}

CVI_S32 CVI_TDL_Set_TextPreprocess(const char *encoderFile, const char *bpeFile,
                                   const char *textFile, int32_t **tokens, int numSentences) {
  std::vector<std::vector<int32_t>> tokens_cpp(numSentences);
  // call token_bpe function
  int result = cvitdl::token_bpe(std::string(encoderFile), std::string(bpeFile),
                                 std::string(textFile), tokens_cpp);
  // calculate the total number of elements
  for (int i = 0; i < numSentences; i++) {
    // tokens[i] = new int32_t[tokens_cpp[i].size()];
    tokens[i] = (int32_t *)malloc(tokens_cpp[i].size() * sizeof(int32_t));
    memcpy(tokens[i], tokens_cpp[i].data(), tokens_cpp[i].size() * sizeof(int32_t));
  }

  if (result == 0) {
    return CVI_SUCCESS;
  }
  LOGE("Tokenization error\n");
  return CVI_FAILURE;
}