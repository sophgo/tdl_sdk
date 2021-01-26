#include "license_plate_detection.hpp"

#include "core/cviai_types_mem.h"
#include "core/cviai_types_mem_internal.h"
#include "core_utils.hpp"
#include "cvi_sys.h"
#include "face_utils.hpp"

#include <sstream>
#include "opencv2/opencv.hpp"

#define DEBUG_LICENSE_PLATE_DETECTION 0

#define OUTPUT_NAME_PROBABILITY "conv2d_25_dequant"
#define OUTPUT_NAME_TRANSFORM "conv2d_26_dequant"

namespace cviai {

LicensePlateDetection::LicensePlateDetection() : Core(CVI_MEM_SYSTEM) {}

LicensePlateDetection::~LicensePlateDetection() {}

static std::vector<cv::Mat> crop_vehicle(VIDEO_FRAME_INFO_S *frame, cvai_object_t *vehicle_meta,
                                         float *rs_scale) {
  frame->stVFrame.pu8VirAddr[0] =
      (CVI_U8 *)CVI_SYS_MmapCache(frame->stVFrame.u64PhyAddr[0], frame->stVFrame.u32Length[0]);
  cv::Mat cv_frame(frame->stVFrame.u32Height, frame->stVFrame.u32Width, CV_8UC3,
                   frame->stVFrame.pu8VirAddr[0], frame->stVFrame.u32Stride[0]);
  if (cv_frame.data == nullptr) {
    LOGE("src image is empty!\n");
    return std::vector<cv::Mat>{};
  }

  std::stringstream s_str;

  std::vector<cv::Mat> vehicle_image_list;
  for (uint32_t i = 0; i < vehicle_meta->size; i++) {
    cv::Rect roi;
    roi.x = vehicle_meta->info[i].bbox.x1;
    roi.y = vehicle_meta->info[i].bbox.y1;
    roi.width = vehicle_meta->info[i].bbox.x2 - roi.x;
    roi.height = vehicle_meta->info[i].bbox.y2 - roi.y;
    cv::Mat vehicle_image = cv_frame(roi);

    int cols = vehicle_image.cols;
    int rows = vehicle_image.rows;
    assert(roi.width == cols);
    assert(roi.height == rows);
    float h_scale = (float)VEHICLE_HEIGHT / rows;
    float w_scale = (float)VEHICLE_WIDTH / cols;
    float scale = MIN(h_scale, w_scale);
    rs_scale[i] = scale;
    cv::resize(vehicle_image, vehicle_image, cv::Size(floor(cols * scale), floor(rows * scale)));
    cols = vehicle_image.cols;
    rows = vehicle_image.rows;

    cv::Mat out_image = cv::Mat::zeros(cv::Size(VEHICLE_WIDTH, VEHICLE_HEIGHT), CV_8UC3);
    vehicle_image.copyTo(out_image(cv::Rect(0, 0, cols, rows)));

    vehicle_image_list.push_back(out_image);
  }

  return vehicle_image_list;
}

int LicensePlateDetection::inference(VIDEO_FRAME_INFO_S *frame, cvai_object_t *vehicle_meta,
                                     cvai_object_t *license_plate_meta) {
#if DEBUG_LICENSE_PLATE_DETECTION
  printf("[%s:%d] inference\n", __FILE__, __LINE__);
#endif
  if (vehicle_meta->size == 0) {
    std::cout << "vehicle count is zero" << std::endl;
    return CVI_SUCCESS;
  }
  float *rs_scale = new float[vehicle_meta->size];  // resize scale
  std::vector<cv::Mat> input_mats = crop_vehicle(frame, vehicle_meta, rs_scale);
  if (input_mats.size() != vehicle_meta->size) {
    LOGE("input mat size != vehicle meta size\n");
    std::cout << "input mat size != vehicle meta size" << std::endl;
    return CVI_FAILURE;
  }
  frame->stVFrame.pu8VirAddr[0] =
      (CVI_U8 *)CVI_SYS_MmapCache(frame->stVFrame.u64PhyAddr[0], frame->stVFrame.u32Length[0]);
  cv::Mat cv_frame(frame->stVFrame.u32Height, frame->stVFrame.u32Width, CV_8UC3,
                   frame->stVFrame.pu8VirAddr[0], frame->stVFrame.u32Stride[0]);
  std::stringstream s_str;

  for (uint32_t n = 0; n < input_mats.size(); n++) {
    prepareInputTensor(input_mats[n]);

    std::vector<VIDEO_FRAME_INFO_S *> dummyFrames = {frame};
    run(dummyFrames);

    float *out_p = getOutputRawPtr<float>(OUTPUT_NAME_PROBABILITY);
    float *out_t = getOutputRawPtr<float>(OUTPUT_NAME_TRANSFORM);

    // TODO:
    //   return more corner points, use std::vector<CornerPts>
    CornerPts corner_pts;
    float score;
    bool detection_result = reconstruct(out_p, out_t, corner_pts, score, 0.9);

    // TODO:
    //   add more points to bpts for false positive
    if (detection_result) {
      // TODO: For backward compatibility, remove old field
      if (license_plate_meta) {
        license_plate_meta->info[n].bpts.size = 4;
        license_plate_meta->info[n].bpts.x = (float *)malloc(sizeof(float) * 4);
        license_plate_meta->info[n].bpts.y = (float *)malloc(sizeof(float) * 4);
        for (int m = 0; m < 4; m++) {
          license_plate_meta->info[n].bpts.x[m] =
              corner_pts(0, m) / rs_scale[n] + vehicle_meta->info[n].bbox.x1;
          license_plate_meta->info[n].bpts.y[m] =
              corner_pts(1, m) / rs_scale[n] + vehicle_meta->info[n].bbox.y1;
        }
      }
      /////////////////////////////////
      vehicle_meta->info[n].vehicle_properity =
          (cvai_vehicle_meta *)malloc(sizeof(cvai_vehicle_meta));
      for (int m = 0; m < 4; m++) {
        vehicle_meta->info[n].vehicle_properity->license_pts.x[m] =
            corner_pts(0, m) / rs_scale[n] + vehicle_meta->info[n].bbox.x1;
        vehicle_meta->info[n].vehicle_properity->license_pts.y[m] =
            corner_pts(1, m) / rs_scale[n] + vehicle_meta->info[n].bbox.y1;
      }

      cvai_bbox_t &bbox = vehicle_meta->info[n].vehicle_properity->license_bbox;
      cvai_4_pts_t &pts = vehicle_meta->info[n].vehicle_properity->license_pts;

      bbox.x1 = std::min({pts.x[0], pts.x[1], pts.x[2], pts.x[3]});
      bbox.x2 = std::max({pts.x[0], pts.x[1], pts.x[2], pts.x[3]});
      bbox.y1 = std::min({pts.y[0], pts.y[1], pts.y[2], pts.y[3]});
      bbox.y2 = std::max({pts.y[0], pts.y[1], pts.y[2], pts.y[3]});
      bbox.score = score;
    }
  }

  delete[] rs_scale;

  return CVI_SUCCESS;
}

void LicensePlateDetection::prepareInputTensor(cv::Mat &input_mat) {
  const TensorInfo &tinfo = getInputTensorInfo(0);
  int8_t *input_ptr = tinfo.get<int8_t>();

  cv::Mat tmpchannels[3];
  cv::split(input_mat, tmpchannels);

  for (int c = 0; c < 3; ++c) {
    tmpchannels[c].convertTo(tmpchannels[c], CV_8UC1);

    int size = tmpchannels[c].rows * tmpchannels[c].cols;
    for (int r = 0; r < tmpchannels[c].rows; ++r) {
      memcpy(input_ptr + size * c + tmpchannels[c].cols * r, tmpchannels[c].ptr(r, 0),
             tmpchannels[c].cols);
    }
  }
}

bool LicensePlateDetection::reconstruct(float *t_prob, float *t_trans, CornerPts &c_pts,
                                        float &ret_prob, float threshold_prob) {
#if DEBUG_LICENSE_PLATE_DETECTION
  printf("[%s:%d] reconstruct\n", __FILE__, __LINE__);
#endif
  Eigen::Matrix<float, 3, 4> anchors;
  // anchors = [[ -0.5,  0.5,  0.5, -0.5]
  //            [ -0.5, -0.5,  0.5,  0.5]
  //            [  1.0,  1.0,  1.0,  1.0]]
  anchors << -0.5, 0.5, 0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0;
  Eigen::Matrix<float, 2, 1> tensor_wh;
  tensor_wh << OUT_TENSOR_W, OUT_TENSOR_H;
  int tensor_size = OUT_TENSOR_H * OUT_TENSOR_W;
  std::vector<LicensePlateObjBBox> lp_cands;
  for (int i = 0; i < OUT_TENSOR_H; i++) {
    for (int j = 0; j < OUT_TENSOR_W; j++) {
      int ij = i * OUT_TENSOR_W + j;
      float prob_pos = t_prob[ij];
      float prob_neg = t_prob[tensor_size + ij];
      float exp_pos = std::exp(prob_pos);
      float exp_neg = std::exp(prob_neg);
      float softmax = exp_pos / (exp_pos + exp_neg);  // softmax
      if (softmax < threshold_prob) continue;
      Eigen::Matrix<float, 2, 1> center_pts;
      center_pts << (float)j + 0.5, (float)i + 0.5;
      Eigen::Matrix<float, 2, 3> affine;
      affine << t_trans[tensor_size * 0 + ij], t_trans[tensor_size * 1 + ij],
          t_trans[tensor_size * 2 + ij], t_trans[tensor_size * 3 + ij],
          t_trans[tensor_size * 4 + ij], t_trans[tensor_size * 5 + ij];
      affine(0, 0) = MAX(affine(0, 0), 0);
      affine(1, 1) = MAX(affine(1, 1), 0);

      Eigen::Matrix<float, 2, 4> affine_pts;
      affine_pts = SIDE * affine * anchors;
      affine_pts = affine_pts.colwise() + center_pts;
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
          affine_pts(i, j) = affine_pts(i, j) / tensor_wh(i, 0);
        }
      }
      // TODO:
      //   [fix]  affine_pts = affine_pts.colwise() / tensor_wh;

      lp_cands.push_back(LicensePlateObjBBox(0, affine_pts, softmax));
    }
  }
  std::vector<LicensePlateObjBBox> selected_LP;
  nms(lp_cands, selected_LP);

  if (selected_LP.size() > 0) {
    for (auto it = selected_LP.begin(); it != selected_LP.end(); it++) {
      Eigen::Matrix<float, 2, 4> corner_pts = it->getCornerPts();
      c_pts = corner_pts;
      ret_prob = it->prob();
      return true;
#if DEBUG_LICENSE_PLATE_DETECTION
      cv::Point2f src_points[4] = {
          cv::Point2f(corner_pts(0, 0), corner_pts(1, 0)),
          cv::Point2f(corner_pts(0, 1), corner_pts(1, 1)),
          cv::Point2f(corner_pts(0, 2), corner_pts(1, 2)),
          cv::Point2f(corner_pts(0, 3), corner_pts(1, 3)),
      };
      cv::Point2f dst_points[4] = {
          cv::Point2f(0, 0),
          cv::Point2f(LICENSE_PLATE_WIDTH, 0),
          cv::Point2f(LICENSE_PLATE_WIDTH, LICENSE_PLATE_HEIGHT),
          cv::Point2f(0, LICENSE_PLATE_HEIGHT),
      };
      cv::Mat M = cv::getPerspectiveTransform(src_points, dst_points);
      std::cout << "M: " << std::endl << M << std::endl;
#endif
    }
  }
  return false;
}

}  // namespace cviai