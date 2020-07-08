#include "face_utils.hpp"

#include "opencv2/imgproc.hpp"

#include <algorithm>

using namespace std;

cv::Mat tformfwd(const cv::Mat &trans, const cv::Mat &uv) {
  cv::Mat uv_h = cv::Mat::ones(uv.rows, 3, CV_64FC1);
  uv.copyTo(uv_h(cv::Rect(0, 0, 2, uv.rows)));
  cv::Mat xv_h = uv_h * trans;
  return xv_h(cv::Rect(0, 0, 2, uv.rows));
}

static cv::Mat find_none_flectives_similarity(const cv::Mat &uv, const cv::Mat &xy) {
  cv::Mat A = cv::Mat::zeros(2 * xy.rows, 4, CV_64FC1);
  cv::Mat b = cv::Mat::zeros(2 * xy.rows, 1, CV_64FC1);
  cv::Mat x = cv::Mat::zeros(4, 1, CV_64FC1);

  xy(cv::Rect(0, 0, 1, xy.rows)).copyTo(A(cv::Rect(0, 0, 1, xy.rows)));  // x
  xy(cv::Rect(1, 0, 1, xy.rows)).copyTo(A(cv::Rect(1, 0, 1, xy.rows)));  // y
  A(cv::Rect(2, 0, 1, xy.rows)).setTo(1.);

  xy(cv::Rect(1, 0, 1, xy.rows)).copyTo(A(cv::Rect(0, xy.rows, 1, xy.rows)));    // y
  (xy(cv::Rect(0, 0, 1, xy.rows))).copyTo(A(cv::Rect(1, xy.rows, 1, xy.rows)));  //-x
  A(cv::Rect(1, xy.rows, 1, xy.rows)) *= -1;
  A(cv::Rect(3, xy.rows, 1, xy.rows)).setTo(1.);

  uv(cv::Rect(0, 0, 1, uv.rows)).copyTo(b(cv::Rect(0, 0, 1, uv.rows)));
  uv(cv::Rect(1, 0, 1, uv.rows)).copyTo(b(cv::Rect(0, uv.rows, 1, uv.rows)));

  cv::solve(A, b, x, cv::DECOMP_SVD);
  cv::Mat trans_inv = (cv::Mat_<double>(3, 3) << x.at<double>(0), -x.at<double>(1), 0,
                       x.at<double>(1), x.at<double>(0), 0, x.at<double>(2), x.at<double>(3), 1);
  cv::Mat trans = trans_inv.inv(cv::DECOMP_SVD);
  trans.at<double>(0, 2) = 0;
  trans.at<double>(1, 2) = 0;
  trans.at<double>(2, 2) = 1;

  return trans;
}

static cv::Mat find_similarity(const cv::Mat &uv, const cv::Mat &xy) {
  cv::Mat trans1 = find_none_flectives_similarity(uv, xy);
  cv::Mat xy_reflect = xy;
  xy_reflect(cv::Rect(0, 0, 1, xy.rows)) *= -1;
  cv::Mat trans2r = find_none_flectives_similarity(uv, xy_reflect);
  cv::Mat reflect = (cv::Mat_<double>(3, 3) << -1, 0, 0, 0, 1, 0, 0, 0, 1);

  cv::Mat trans2 = trans2r * reflect;
  cv::Mat xy1 = tformfwd(trans1, uv);

  double norm1 = cv::norm(xy1 - xy);

  cv::Mat xy2 = tformfwd(trans2, uv);
  double norm2 = cv::norm(xy2 - xy);

  cv::Mat trans;
  if (norm1 < norm2) {
    trans = trans1;
  } else {
    trans = trans2;
  }
  return trans;
}

static cv::Mat get_similarity_transform(const vector<cv::Point2f> &src_pts,
                                        const vector<cv::Point2f> &dest_pts, bool reflective) {
  cv::Mat src((int)src_pts.size(), 2, CV_32FC1, (void *)(&src_pts[0].x));
  src.convertTo(src, CV_64FC1);

  cv::Mat dst((int)dest_pts.size(), 2, CV_32FC1, (void *)(&dest_pts[0].x));
  dst.convertTo(dst, CV_64FC1);

  cv::Mat trans = reflective ? find_similarity(src, dst) : find_none_flectives_similarity(src, dst);
  return trans(cv::Rect(0, 0, 2, trans.rows)).t();
}

namespace cviai {

cvi_face_info_t bbox_rescale(VIDEO_FRAME_INFO_S *frame, cvi_face_t *face_meta, int face_idx) {
  float width = frame->stVFrame.u32Width;
  float height = frame->stVFrame.u32Height;
  float ratio_x, ratio_y, bbox_y_height, bbox_x_height, bbox_padding_top, bbox_padding_left;

  if (width >= height) {
    ratio_x = width / face_meta->width;
    bbox_y_height = face_meta->height * height / width;
    ratio_y = height / bbox_y_height;
    bbox_padding_top = (face_meta->height - bbox_y_height) / 2;
  } else {
    ratio_y = height / face_meta->height;
    bbox_x_height = face_meta->width * width / height;
    ratio_x = width / bbox_x_height;
    bbox_padding_left = (face_meta->width - bbox_x_height) / 2;
  }

  cvi_bbox_t bbox = face_meta->face_info[face_idx].bbox;
  cvi_face_info_t face_info;
  float x1, x2, y1, y2;

  if (width >= height) {
    x1 = bbox.x1 * ratio_x;
    x2 = bbox.x2 * ratio_x;
    y1 = (bbox.y1 - bbox_padding_top) * ratio_y;
    y2 = (bbox.y2 - bbox_padding_top) * ratio_y;

    for (int j = 0; j < 5; ++j) {
      face_info.face_pts.x[j] = face_meta->face_info[face_idx].face_pts.x[j] * ratio_x;
      face_info.face_pts.y[j] =
          (face_meta->face_info[face_idx].face_pts.y[j] - bbox_padding_top) * ratio_y;
    }
  } else {
    x1 = (bbox.x1 - bbox_padding_left) * ratio_x;
    x2 = (bbox.x2 - bbox_padding_left) * ratio_x;
    y1 = bbox.y1 * ratio_y;
    y2 = bbox.y2 * ratio_y;

    for (int j = 0; j < 5; ++j) {
      face_info.face_pts.x[j] =
          (face_meta->face_info[face_idx].face_pts.x[j] - bbox_padding_left) * ratio_x;
      face_info.face_pts.y[j] = face_meta->face_info[face_idx].face_pts.y[j] * ratio_y;
    }
  }

  face_info.bbox.x1 = max(min(x1, width - 1), (float)0);
  face_info.bbox.x2 = max(min(x2, width - 1), (float)0);
  face_info.bbox.y1 = max(min(y1, height - 1), (float)0);
  face_info.bbox.y2 = max(min(y2, height - 1), (float)0);

  return face_info;
}

int face_align(const cv::Mat &image, cv::Mat &aligned, const cvi_face_info_t &face_info, int width,
               int height) {
  assert(width == 96 || width == 112);
  assert(height == 112);
  if ((width != 96 && width != 112) || height != 112) {
    return -1;
  }

  int ref_width = width;
  int ref_height = height;

  vector<cv::Point2f> detect_points;
  for (int j = 0; j < 5; ++j) {
    cv::Point2f e;
    e.x = face_info.face_pts.x[j];
    e.y = face_info.face_pts.y[j];
    detect_points.emplace_back(e);
  }

  vector<cv::Point2f> reference_points;
  if (96 == width) {
    reference_points = {{30.29459953, 51.69630051},
                        {65.53179932, 51.50139999},
                        {48.02519989, 71.73660278},
                        {33.54930115, 92.3655014},
                        {62.72990036, 92.20410156}};
  } else {
    reference_points = {{38.29459953, 51.69630051},
                        {73.53179932, 51.50139999},
                        {56.02519989, 71.73660278},
                        {41.54930115, 92.3655014},
                        {70.72990036, 92.20410156}};
  }

  for (auto &e : reference_points) {
    e.x += (width - ref_width) / 2.0f;
    e.y += (height - ref_height) / 2.0f;
  }
  cv::Mat tfm = get_similarity_transform(detect_points, reference_points, true);
  cv::warpAffine(image, aligned, tfm, cv::Size(width, height), cv::INTER_NEAREST);

  return 0;
}

}  // namespace cviai