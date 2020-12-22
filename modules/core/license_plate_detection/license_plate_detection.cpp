#include "license_plate_detection.hpp"

#include "core/cviai_types_mem.h"
#include "core/cviai_types_mem_internal.h"
#include "core_utils.hpp"
#include "face_utils.hpp"

#include "cvi_sys.h"
#include "opencv2/opencv.hpp"

#define DEBUG_LICENSE_PLATE_DETECTION 0

#define OUTPUT_NAME_PROBABILITY "conv2d_25_dequant"
#define OUTPUT_NAME_TRANSFORM "conv2d_26_dequant"

#include <sstream>

namespace cviai {

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
#if DEBUG_LICENSE_PLATE_DETECTION
    std::cout << "cv_frame.cols = " << cv_frame.cols << std::endl;
    std::cout << "cv_frame.rows = " << cv_frame.rows << std::endl;
    std::cout << "roi.x = " << roi.x << std::endl;
    std::cout << "roi.y = " << roi.y << std::endl;
    std::cout << "roi.width  = " << roi.width << std::endl;
    std::cout << "roi.height = " << roi.height << std::endl;
#endif
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

    cv::Mat out_image = cv::Mat::zeros(cv::Size(VEHICLE_WIDTH, VEHICLE_HEIGHT), CV_32FC3);
    vehicle_image.copyTo(out_image(cv::Rect(0, 0, cols, rows)));
    // s_str.str("");
    // s_str << "tmp_vehicle_" << i << ".jpg";
    // cv::imwrite(s_str.str().c_str(), out_image);
    out_image = out_image / 255.0;

    vehicle_image_list.push_back(out_image);
  }

  return vehicle_image_list;
}

LicensePlateDetection::LicensePlateDetection() {
  mp_mi = std::make_unique<CvimodelInfo>();
  mp_mi->conf.input_mem_type = CVI_MEM_SYSTEM;
}

LicensePlateDetection::~LicensePlateDetection() {}

int LicensePlateDetection::inference(VIDEO_FRAME_INFO_S *frame, cvai_object_t *vehicle_meta,
                                     cvai_object_t *license_plate_meta) {
#if DEBUG_LICENSE_PLATE_DETECTION
  std::cout << "LicensePlateDetection::inference" << std::endl;
#endif
  if (vehicle_meta->size == 0) {
    std::cout << "vehicle count is zero" << std::endl;
    return CVI_SUCCESS;
  }
  float *rs_scale = new float[vehicle_meta->size];  // resize scale
  std::vector<cv::Mat> input_mats = crop_vehicle(frame, vehicle_meta, rs_scale);
  if (input_mats.size() != vehicle_meta->size) {
    std::cout << "input mat size != vehicle meta size" << std::endl;
    return CVI_FAILURE;
  }
  frame->stVFrame.pu8VirAddr[0] =
      (CVI_U8 *)CVI_SYS_MmapCache(frame->stVFrame.u64PhyAddr[0], frame->stVFrame.u32Length[0]);
  cv::Mat cv_frame(frame->stVFrame.u32Height, frame->stVFrame.u32Width, CV_8UC3,
                   frame->stVFrame.pu8VirAddr[0], frame->stVFrame.u32Stride[0]);
  // cv::imwrite("tmp_frame.jpg", cv_frame);
  std::stringstream s_str;

  CVI_TENSOR *input =
      CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, mp_mi->in.tensors, mp_mi->in.num);

  for (uint32_t n = 0; n < input_mats.size(); n++) {
    // if (n>0){
    //   continue;
    // }
    uint16_t *input_ptr = (uint16_t *)CVI_NN_TensorPtr(input);
    std::vector<cv::Mat> rgbChannels(3);
    cv::split(input_mats[n], rgbChannels);

    int rows = input_mats[n].rows;
    int cols = input_mats[n].cols;
    for (int c = 0; c < 3; c++) {
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          uint16_t bf16_input = 0;
          floatToBF16((float *)rgbChannels[2 - c].ptr(i, j), &bf16_input);
          memcpy(input_ptr + rows * cols * c + cols * i + j, &bf16_input, sizeof(uint16_t));
        }
      }
    }

    std::vector<VIDEO_FRAME_INFO_S *> frames = {frame};
    run(frames);

    CVI_TENSOR *out_probability =
        CVI_NN_GetTensorByName(OUTPUT_NAME_PROBABILITY, mp_mi->out.tensors, mp_mi->out.num);
    CVI_TENSOR *out_transform =
        CVI_NN_GetTensorByName(OUTPUT_NAME_TRANSFORM, mp_mi->out.tensors, mp_mi->out.num);

#if DEBUG_LICENSE_PLATE_DETECTION
    if (out_probability == nullptr || out_transform == nullptr) {
      std::cout << "out_probability or out_transform is nullptr" << std::endl;
    }
#endif

    float *out_p = (float *)CVI_NN_TensorPtr(out_probability);
    float *out_t = (float *)CVI_NN_TensorPtr(out_transform);

    // TODO:
    //   return more corner points, use std::vector<CornerPts>
    CornerPts corner_pts;
    bool detection_result = reconstruct(out_p, out_t, corner_pts, 0.9);

    // std::cout << "scale = " << rs_scale[n] << std::endl;
    // std::cout << corner_pts << std::endl;
    // std::cout << "bbox.x1 = " << vehicle_meta->info[n].bbox.x1 << std::endl
    //           << "bbox.y1 = " << vehicle_meta->info[n].bbox.y1 << std::endl;
    // cv::Point2f src_points[4] = {
    //     cv::Point2f(corner_pts(0,0) / rs_scale[n] + vehicle_meta->info[n].bbox.x1,
    //                 corner_pts(1,0) / rs_scale[n] + vehicle_meta->info[n].bbox.y1),
    //     cv::Point2f(corner_pts(0,1) / rs_scale[n] + vehicle_meta->info[n].bbox.x1,
    //                 corner_pts(1,1) / rs_scale[n] + vehicle_meta->info[n].bbox.y1),
    //     cv::Point2f(corner_pts(0,2) / rs_scale[n] + vehicle_meta->info[n].bbox.x1,
    //                 corner_pts(1,2) / rs_scale[n] + vehicle_meta->info[n].bbox.y1),
    //     cv::Point2f(corner_pts(0,3) / rs_scale[n] + vehicle_meta->info[n].bbox.x1,
    //                 corner_pts(1,3) / rs_scale[n] + vehicle_meta->info[n].bbox.y1),
    // };
    // std::cout << src_points << std::endl;

    // cv::Point2f dst_points[4] = {
    //     cv::Point2f(0, 0),
    //     cv::Point2f(LICENSE_PLATE_WIDTH, 0),
    //     cv::Point2f(LICENSE_PLATE_WIDTH, LICENSE_PLATE_HEIGHT),
    //     cv::Point2f(0, LICENSE_PLATE_HEIGHT),
    // };
    // cv::Mat M = cv::getPerspectiveTransform(src_points, dst_points);
    // cv::Mat sub_cvFrame;
    // cv::warpPerspective(cv_frame, sub_cvFrame, M,
    //                     cv::Size(LICENSE_PLATE_WIDTH, LICENSE_PLATE_HEIGHT), cv::INTER_LINEAR);
    // cv::Mat greyMat;
    // cv::cvtColor(sub_cvFrame, greyMat, cv::COLOR_RGB2GRAY); /* BGR or RGB ? */
    // cv::cvtColor(greyMat, sub_cvFrame, cv::COLOR_GRAY2RGB);
    // s_str.str("");
    // s_str << "tmp_license_plate_debug_" << n << ".jpg";
    // cv::imwrite(s_str.str().c_str(), greyMat);

#if DEBUG_LICENSE_PLATE_DETECTION
    if (detection_result) {
      std::cout << "Detect LP! corner pts:" << std::endl << corner_pts << std::endl;
    } else {
      std::cout << "Do not detect LP!!!" << std::endl;
    }
#endif

    // TODO:
    //   add more points to bpts for false positive
    if (detection_result) {
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
  }

  delete[] rs_scale;

  return CVI_SUCCESS;
}

bool LicensePlateDetection::reconstruct(float *t_prob, float *t_trans, CornerPts &c_pts,
                                        float threshold_prob) {
#if DEBUG_LICENSE_PLATE_DETECTION
  std::cout << "LicensePlateDetection::reconstruct" << std::endl;
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

      lp_cands.push_back(LicensePlateObjBBox(0, affine_pts, prob_pos));
    }
  }
  std::vector<LicensePlateObjBBox> selected_LP;
  nms(lp_cands, selected_LP);

  if (selected_LP.size() > 0) {
    for (auto it = selected_LP.begin(); it != selected_LP.end(); it++) {
      Eigen::Matrix<float, 2, 4> corner_pts = it->getCornerPts();
      c_pts = corner_pts;
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