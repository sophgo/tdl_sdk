#include "alphapose.hpp"

#include "core/core/cvtdl_errno.h"
#include "cvi_sys.h"

#ifdef ENABLE_CVI_TDL_CV_UTILS
#include "cv/imgproc.hpp"
#else
#include "opencv2/imgproc.hpp"
#endif

#define ALPHAPOSE_C 3
#define ALPHAPOSE_PTS_NUM 17
#define OUTPUT_NAME "output_Conv_dequant"

static cv::Point2f get3rdPoint(cv::Point2f a, cv::Point2f b) {
  cv::Point2f direct;
  direct.x = b.x - (a - b).y;
  direct.y = b.y + (a - b).x;

  return direct;
}

static std::vector<float> getDir(float src_w) {
  // float sn = sin(0);
  // float cs = cos(0);

  // vector<float> src_result(2, 0);
  // src_result[0] = -src_w * sn;
  // src_result[1] = src_w * cs;
  std::vector<float> src_result(2, 0);
  src_result[0] = 0;
  src_result[1] = src_w;

  return src_result;
}

static cv::Mat getAffineTransform(const std::vector<float> &center, const std::vector<float> &scale,
                                  const std::vector<float> &output_size, bool inv = false) {
  std::vector<float> shift(2, 0);
  float src_w = scale[0];
  int dst_h = output_size[0];
  int dst_w = output_size[1];

  std::vector<float> src_dir = getDir(src_w * -0.5);
  std::vector<float> dst_dir(2, 0);
  dst_dir[1] = dst_w * -0.5;

  cv::Point2f src[3];
  cv::Point2f dst[3];

  src[0] = cv::Point2f(center[0], center[1]);
  src[1] = cv::Point2f(center[0] + src_dir[0], center[1] + src_dir[1]);
  src[2] = get3rdPoint(src[0], src[1]);
  dst[0] = cv::Point2f(dst_w * 0.5, dst_h * 0.5);
  dst[1] = cv::Point2f(dst_w * 0.5 + dst_dir[0], dst_h * 0.5 + dst_dir[1]);
  dst[2] = get3rdPoint(dst[0], dst[1]);

#ifdef ENABLE_CVI_TDL_CV_UTILS
  if (inv)
    return cvitdl::getAffineTransform(dst, src);
  else
    return cvitdl::getAffineTransform(src, dst);
#else
  if (inv)
    return cv::getAffineTransform(dst, src);
  else
    return cv::getAffineTransform(src, dst);
#endif
}

static cvtdl_bbox_t centerScaleToBox(const std::vector<float> &center,
                                     const std::vector<float> &scale) {
  float w = scale[0] * 1.0;
  float h = scale[1] * 1.0;
  cvtdl_bbox_t bbox;

  bbox.x1 = center[0] - w * 0.5;
  bbox.y1 = center[1] - h * 0.5;
  bbox.x2 = bbox.x1 + w;
  bbox.y2 = bbox.y1 + h;

  return bbox;
}

static void boxToCenterScale(float x, float y, float w, float h, float aspect_ratio,
                             std::vector<float> &scale, std::vector<float> &center) {
  float pixel_std = 1;
  float scale_mult = 1.25;

  center[0] = x + w * 0.5;
  center[1] = y + h * 0.5;

  if (w > aspect_ratio * h) {
    h = w / aspect_ratio;
  } else if (w < aspect_ratio * h) {
    w = h * aspect_ratio;
  }

  scale[0] = w * 1.0 / pixel_std;
  scale[1] = h * 1.0 / pixel_std;
  if (center[0] != -1) {
    scale[0] = scale[0] * scale_mult;
    scale[1] = scale[1] * scale_mult;
  }
}

static void preprocess(const cvtdl_bbox_t &input_bbox, const cv::Mat &input_image,
                       cvtdl_bbox_t &align_bbox, cv::Mat &align_image, int pose_h = 256,
                       int pose_w = 192) {
  float _aspect_ratio = float(pose_w) / pose_h;

  float x = input_bbox.x1;
  float y = input_bbox.y1;
  float w = input_bbox.x2 - input_bbox.x1;
  float h = input_bbox.y2 - input_bbox.y1;

  std::vector<float> center(2, 0);
  std::vector<float> scale(2, 0);
  boxToCenterScale(x, y, w, h, _aspect_ratio, scale, center);

  cv::Mat trans = getAffineTransform(center, scale, {(float)pose_h, (float)pose_w});
#ifdef ENABLE_CVI_TDL_CV_UTILS
  cvitdl::warpAffine(input_image, align_image, trans, cv::Size(int(pose_w), int(pose_h)),
                     INTER_LINEAR);
#else
  cv::warpAffine(input_image, align_image, trans, cv::Size(int(pose_w), int(pose_h)),
                 cv::INTER_LINEAR);
#endif
  align_bbox = centerScaleToBox(center, scale);

  align_image.convertTo(align_image, CV_32FC3, 1.0 / 255);
  cv::Scalar mean = cv::Scalar(0.406, 0.457, 0.48);
  align_image -= mean;
}

static void getMaxPred(const cv::Mat &pose_pred, cvtdl_pose17_meta_t &dst_pose) {
  int inner_size = pose_pred.size[2] * pose_pred.size[3];
  float *ptr = (float *)pose_pred.data;
  for (int c = 0; c < ALPHAPOSE_PTS_NUM; ++c) {
    dst_pose.score[c] = 0;
    dst_pose.x[c] = 0;
    dst_pose.y[c] = 0;
    // for (int h = 0; h < pose_pred.size[2]; ++h) {
    //    for (int w = 0; w < pose_pred.size[3]; ++w) {
    //        float current_score = blob_to_val(pose_pred, 0, c, h, w);
    //        if (current_score > dst_pose.score[c]) {
    //            dst_pose.score[c] = current_score;
    //            dst_pose.x[c] = w;
    //            dst_pose.y[c] = h;
    //        }
    //    }
    //}
    int max_idx = 0;
    for (int i = 0; i < inner_size; ++i) {
      if (ptr[i] > dst_pose.score[c]) {
        dst_pose.score[c] = ptr[i];
        max_idx = i;
      }
    }
    dst_pose.x[c] = max_idx % pose_pred.size[3];
    dst_pose.y[c] = max_idx / pose_pred.size[3];
    ptr += inner_size;
  }
}

static void simplePostprocess(const std::vector<cv::Mat> &pose_pred_list,
                              const std::vector<cvtdl_bbox_t> &align_bbox_list,
                              std::vector<cvtdl_pose17_meta_t> &dst_pose_list) {
  for (uint32_t i = 0; i < pose_pred_list.size(); ++i) {
    float x = align_bbox_list[i].x1;
    float y = align_bbox_list[i].y1;
    float w = align_bbox_list[i].x2 - align_bbox_list[i].x1;
    float h = align_bbox_list[i].y2 - align_bbox_list[i].y1;
    std::vector<float> center = {x + w * 0.5f, y + h * 0.5f};
    std::vector<float> scale = {w, h};

    getMaxPred(pose_pred_list[i], dst_pose_list[i]);
    cv::Mat trans = getAffineTransform(
        center, scale, {(float)pose_pred_list[i].size[2], (float)pose_pred_list[i].size[3]}, true);
    for (int c = 0; c < ALPHAPOSE_PTS_NUM; ++c) {
      dst_pose_list[i].x[c] = trans.at<double>(0) * dst_pose_list[i].x[c] +
                              trans.at<double>(1) * dst_pose_list[i].y[c] + trans.at<double>(2);
      dst_pose_list[i].y[c] = trans.at<double>(3) * dst_pose_list[i].x[c] +
                              trans.at<double>(4) * dst_pose_list[i].y[c] + trans.at<double>(5);
    }
  }
}

namespace cvitdl {

AlphaPose::AlphaPose() : Core(CVI_MEM_SYSTEM) {}

AlphaPose::~AlphaPose() {}

int AlphaPose::setupInputPreprocess(std::vector<InputPreprecessSetup> *data) {
  if (data->size() != 1) {
    LOGE("Alpha pose only has 1 input.\n");
    return CVI_TDL_ERR_INVALID_ARGS;
  }

  return CVI_TDL_SUCCESS;
}

int AlphaPose::inference(VIDEO_FRAME_INFO_S *srcFrame, cvtdl_object_t *objects) {
  srcFrame->stVFrame.pu8VirAddr[0] = (CVI_U8 *)CVI_SYS_MmapCache(srcFrame->stVFrame.u64PhyAddr[0],
                                                                 srcFrame->stVFrame.u32Length[0]);
  cv::Mat img_rgb(srcFrame->stVFrame.u32Height, srcFrame->stVFrame.u32Width, CV_8UC3,
                  srcFrame->stVFrame.pu8VirAddr[0], srcFrame->stVFrame.u32Stride[0]);
  if (img_rgb.data == nullptr) {
    LOGE("src image is empty!\n");
    return CVI_TDL_ERR_INVALID_ARGS;
  }

  std::vector<cvtdl_bbox_t> align_bbox_list;
  std::vector<cv::Mat> pose_pred_list;

  for (uint32_t i = 0; i < objects->size; ++i) {
    cvtdl_bbox_t align_bbox;
    // human label is 0 in COCO dataset
    if (objects->info[i].classes != 0) {
      cv::Mat pose_pred({1, 17, 64, 48}, CV_32FC1, cv::Scalar(0));
      pose_pred_list.push_back(pose_pred);
      align_bbox_list.push_back(align_bbox);
      continue;
    }

    cvtdl_bbox_t predict_bbox = objects->info[i].bbox;
    prepareInputTensor(predict_bbox, img_rgb, align_bbox);

    std::vector<VIDEO_FRAME_INFO_S *> frames = {srcFrame};
    int ret = run(frames);
    if (ret != CVI_TDL_SUCCESS) {
      return ret;
    }

    size_t output_size = getOutputTensorElem(OUTPUT_NAME);
    float *output_data = getOutputRawPtr<float>(OUTPUT_NAME);

    cv::Mat pose_pred({1, 17, 64, 48}, CV_32FC1, cv::Scalar(0));
    memcpy(pose_pred.data, output_data, output_size * sizeof(float));
    pose_pred_list.push_back(pose_pred);
    align_bbox_list.push_back(align_bbox);
  }

  std::vector<cvtdl_pose17_meta_t> pose_list(objects->size);
  simplePostprocess(pose_pred_list, align_bbox_list, pose_list);

  // TODO - Use cvi_pose_list_meta_t directly in simplePostprocess
  for (uint32_t i = 0; i < pose_list.size(); ++i) {
    objects->info[i].pedestrian_properity =
        (cvtdl_pedestrian_meta *)malloc(sizeof(cvtdl_pedestrian_meta));
    objects->info[i].pedestrian_properity->pose_17 = pose_list[i];
  }

  CVI_SYS_Munmap((void *)srcFrame->stVFrame.pu8VirAddr[0], srcFrame->stVFrame.u32Length[0]);
  srcFrame->stVFrame.pu8VirAddr[0] = NULL;

  return CVI_TDL_SUCCESS;
}

void AlphaPose::prepareInputTensor(const cvtdl_bbox_t &bbox, cv::Mat img_rgb,
                                   cvtdl_bbox_t &align_bbox) {
  cv::Mat align_img;
  preprocess(bbox, img_rgb, align_bbox, align_img);

  cv::Mat input_channels[ALPHAPOSE_C];
  split(align_img, input_channels);

  const TensorInfo &tinfo = getInputTensorInfo(0);
  int8_t *input_ptr = tinfo.get<int8_t>();
  float quant_scale = getInputQuantScale(0);

  // memcpy by row, because opencv might add some datas at the end of each row
  for (int c = 0; c < ALPHAPOSE_C; ++c) {
    input_channels[c].convertTo(input_channels[c], CV_8SC1, quant_scale, 0);

    int size = input_channels[c].rows * input_channels[c].cols;
    for (int r = 0; r < input_channels[c].rows; ++r) {
      memcpy(input_ptr + size * c + input_channels[c].cols * r, input_channels[c].ptr(r, 0),
             input_channels[c].cols);
    }
  }
}

}  // namespace cvitdl
