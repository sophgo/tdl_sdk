#include "alphapose.hpp"

#include "cvi_sys.h"

#define ALPHAPOSE_C 3
#define ALPHAPOSE_PTS_NUM 17
#define OUTPUT_NAME "output_Conv_dequant"

static std::vector<std::pair<int, int>> l_pair = {{0, 1},   {0, 2},   {1, 3},   {2, 4},   {5, 6},
                                                  {5, 7},   {7, 9},   {6, 8},   {8, 10},  {17, 11},
                                                  {17, 12}, {11, 13}, {12, 14}, {13, 15}, {14, 16}};

static std::vector<cv::Scalar> p_color = {
    {0, 255, 255},  {0, 191, 255},  {0, 255, 102},  {0, 77, 255},   {0, 255, 0},    {77, 255, 255},
    {77, 255, 204}, {77, 204, 255}, {191, 255, 77}, {77, 191, 255}, {191, 255, 77}, {204, 77, 255},
    {77, 255, 204}, {191, 77, 255}, {77, 255, 191}, {127, 77, 255}, {77, 255, 127}, {0, 255, 255}};

static std::vector<cv::Scalar> line_color = {
    {0, 215, 255},   {0, 255, 204},  {0, 134, 255},  {0, 255, 50},  {77, 255, 222},
    {77, 196, 255},  {77, 135, 255}, {191, 255, 77}, {77, 255, 77}, {77, 222, 255},
    {255, 156, 127}, {0, 127, 255},  {255, 127, 77}, {0, 77, 255},  {255, 77, 36}};

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

  if (inv)
    return cv::getAffineTransform(dst, src);
  else
    return cv::getAffineTransform(src, dst);
}

static cvai_bbox_t centerScaleToBox(const std::vector<float> &center,
                                    const std::vector<float> &scale) {
  float w = scale[0] * 1.0;
  float h = scale[1] * 1.0;
  cvai_bbox_t bbox;

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

static void preprocess(const cvai_bbox_t &input_bbox, const cv::Mat &input_image,
                       cvai_bbox_t &align_bbox, cv::Mat &align_image, int pose_h = 256,
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
  cv::warpAffine(input_image, align_image, trans, cv::Size(int(pose_w), int(pose_h)),
                 cv::INTER_LINEAR);
  align_bbox = centerScaleToBox(center, scale);
  // cv::cvtColor(align_image, align_image, CV_BGR2RGB);
  align_image.convertTo(align_image, CV_32FC3, 1.0 / 255);
  cv::Scalar mean = cv::Scalar(0.406, 0.457, 0.48);
  align_image -= mean;
}

static void getMaxPred(const cv::Mat &pose_pred, cvai_pose17_meta_t &dst_pose) {
  int inner_size = pose_pred.size[2] * pose_pred.size[3];
  float *ptr = (float *)pose_pred.data;
  for (int c = 0; c < 17; ++c) {
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
                              const std::vector<cvai_bbox_t> &align_bbox_list,
                              std::vector<cvai_pose17_meta_t> &dst_pose_list) {
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
    for (int c = 0; c < 17; ++c) {
      dst_pose_list[i].x[c] = trans.at<double>(0) * dst_pose_list[i].x[c] +
                              trans.at<double>(1) * dst_pose_list[i].y[c] + trans.at<double>(2);
      dst_pose_list[i].y[c] = trans.at<double>(3) * dst_pose_list[i].x[c] +
                              trans.at<double>(4) * dst_pose_list[i].y[c] + trans.at<double>(5);
    }
  }
}

static cv::Mat draw_pose(cv::Mat &image, std::vector<cvai_pose17_meta_t> &pose_list) {
  cv::Mat img = image.clone();

  for (cvai_pose17_meta_t pose : pose_list) {
    std::vector<cv::Point2f> kp_preds(ALPHAPOSE_PTS_NUM);
    std::vector<float> kp_scores(ALPHAPOSE_PTS_NUM);

    for (int i = 0; i < ALPHAPOSE_PTS_NUM; ++i) {
      kp_preds[i].x = pose.x[i];
      kp_preds[i].y = pose.y[i];
      kp_scores[i] = pose.score[i];
    }

    cv::Point2f extra_pred;
    extra_pred.x = (kp_preds[5].x + kp_preds[6].x) / 2;
    extra_pred.y = (kp_preds[5].y + kp_preds[6].y) / 2;
    kp_preds.push_back(extra_pred);

    float extra_score = (kp_scores[5] + kp_scores[6]) / 2;
    kp_scores.push_back(extra_score);

    // Draw keypoints
    std::unordered_map<int, std::pair<int, int>> part_line;
    for (uint32_t n = 0; n < kp_scores.size(); n++) {
      if (kp_scores[n] <= 0.35) continue;

      int cor_x = kp_preds[n].x;
      int cor_y = kp_preds[n].y;
      part_line[n] = std::make_pair(cor_x, cor_y);

      cv::Mat bg;
      img.copyTo(bg);
      cv::circle(bg, cv::Size(cor_x, cor_y), 2, p_color[n], -1);
      float transparency = std::max(float(0.0), std::min(float(1.0), kp_scores[n]));
      cv::addWeighted(bg, transparency, img, 1 - transparency, 0, img);
    }

    // Draw limbs
    for (uint32_t i = 0; i < l_pair.size(); i++) {
      int start_p = l_pair[i].first;
      int end_p = l_pair[i].second;
      if (part_line.count(start_p) > 0 && part_line.count(end_p) > 0) {
        std::pair<int, int> start_xy = part_line[start_p];
        std::pair<int, int> end_xy = part_line[end_p];

        float mX = (start_xy.first + end_xy.first) / 2;
        float mY = (start_xy.second + end_xy.second) / 2;
        float length = sqrt(pow((start_xy.second - end_xy.second), 2) +
                            pow((start_xy.first - end_xy.first), 2));
        float angle =
            atan2(start_xy.second - end_xy.second, start_xy.first - end_xy.first) * 180.0 / M_PI;
        float stickwidth = (kp_scores[start_p] + kp_scores[end_p]) + 1;
        std::vector<cv::Point> polygon;
        cv::ellipse2Poly(cv::Point(int(mX), int(mY)), cv::Size(int(length / 2), stickwidth),
                         int(angle), 0, 360, 1, polygon);

        cv::Mat bg;
        img.copyTo(bg);
        cv::fillConvexPoly(bg, polygon, line_color[i]);
        float transparency = std::max(
            float(0.0), std::min(float(1.0), float(0.5) * (kp_scores[start_p] + kp_scores[end_p])));
        cv::addWeighted(bg, transparency, img, 1 - transparency, 0, img);
      }
    }
  }

  return img;
}

namespace cviai {

AlphaPose::AlphaPose() : Core(CVI_MEM_SYSTEM) {}

AlphaPose::~AlphaPose() {}

int AlphaPose::setupInputPreprocess(std::vector<InputPreprecessSetup> *data) {
  if (data->size() != 1) {
    LOGE("Alpha pose only has 1 input.\n");
    return CVI_FAILURE;
  }

  return CVI_SUCCESS;
}

int AlphaPose::inference(VIDEO_FRAME_INFO_S *srcFrame, cvai_object_t *objects) {
  srcFrame->stVFrame.pu8VirAddr[0] = (CVI_U8 *)CVI_SYS_MmapCache(srcFrame->stVFrame.u64PhyAddr[0],
                                                                 srcFrame->stVFrame.u32Length[0]);
  cv::Mat img_rgb(srcFrame->stVFrame.u32Height, srcFrame->stVFrame.u32Width, CV_8UC3,
                  srcFrame->stVFrame.pu8VirAddr[0], srcFrame->stVFrame.u32Stride[0]);
  if (img_rgb.data == nullptr) {
    LOGE("src image is empty!\n");
    return CVI_FAILURE;
  }

  std::vector<cvai_bbox_t> align_bbox_list;
  std::vector<cv::Mat> pose_pred_list;

  for (uint32_t i = 0; i < objects->size; ++i) {
    cvai_bbox_t align_bbox;
    // human label is 0 in COCO dataset
    if (objects->info[i].classes != 0) {
      cv::Mat pose_pred({1, 17, 64, 48}, CV_32FC1, cv::Scalar(0));
      pose_pred_list.push_back(pose_pred);
      align_bbox_list.push_back(align_bbox);
      continue;
    }

    cvai_bbox_t predict_bbox = objects->info[i].bbox;
    // predict_bbox = box_rescale(srcFrame->stVFrame.u32Width, srcFrame->stVFrame.u32Height,
    // objects->width, objects->height, objects->info[i].bbox,
    // meta_rescale_type_e::RESCALE_CENTER);

    prepareInputTensor(predict_bbox, img_rgb, align_bbox);

    std::vector<VIDEO_FRAME_INFO_S *> frames = {srcFrame};
    run(frames);

    size_t output_size = getOutputTensorElem(OUTPUT_NAME);
    float *output_data = getOutputRawPtr<float>(OUTPUT_NAME);

    cv::Mat pose_pred({1, 17, 64, 48}, CV_32FC1, cv::Scalar(0));
    memcpy(pose_pred.data, output_data, output_size * sizeof(float));
    pose_pred_list.push_back(pose_pred);
    align_bbox_list.push_back(align_bbox);
  }

  std::vector<cvai_pose17_meta_t> pose_list(objects->size);
  simplePostprocess(pose_pred_list, align_bbox_list, pose_list);

  cv::Mat draw_img = draw_pose(img_rgb, pose_list);
  // cv::cvtColor(draw_img, draw_img, CV_RGB2BGR);
  // cv::imwrite("/mnt/data/out.jpg", draw_img);

  // TODO - Use cvi_pose_list_meta_t directly in simplePostprocess
  for (uint32_t i = 0; i < pose_list.size(); ++i) {
    objects->info[i].pos_17 = pose_list[i];
  }

  return CVI_SUCCESS;
}

void AlphaPose::prepareInputTensor(const cvai_bbox_t &bbox, cv::Mat img_rgb,
                                   cvai_bbox_t &align_bbox) {
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

}  // namespace cviai
