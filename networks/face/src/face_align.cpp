

#include "core/net_types.hpp"
#include "net_face.hpp"
#include "opencv2/imgproc.hpp"
#include "utils/function_tracer.h"

namespace qnn {
namespace vision {

using cv::Mat;
using std::string;
using std::vector;

Mat tformfwd(const Mat &trans, const Mat &uv);
Mat find_similarity(const Mat &uv, const Mat &xy);
Mat find_none_flectives_similarity(const Mat &uv, const Mat &xy);
Mat get_similarity_transform(const vector<cv::Point2f> &src, const vector<cv::Point2f> &dest,
                             bool reflective);

Mat tformfwd(const Mat &trans, const Mat &uv) {
    Mat uv_h = Mat::ones(uv.rows, 3, CV_64FC1);
    uv.copyTo(uv_h(cv::Rect(0, 0, 2, uv.rows)));
    Mat xv_h = uv_h * trans;
    return xv_h(cv::Rect(0, 0, 2, uv.rows));
}

Mat find_similarity(const Mat &uv, const Mat &xy) {
    Mat trans1 = find_none_flectives_similarity(uv, xy);
    Mat xy_reflect = xy;
    xy_reflect(cv::Rect(0, 0, 1, xy.rows)) *= -1;
    Mat trans2r = find_none_flectives_similarity(uv, xy_reflect);
    Mat reflect = (cv::Mat_<double>(3, 3) << -1, 0, 0, 0, 1, 0, 0, 0, 1);

    Mat trans2 = trans2r * reflect;
    Mat xy1 = tformfwd(trans1, uv);

    double norm1 = cv::norm(xy1 - xy);

    Mat xy2 = tformfwd(trans2, uv);
    double norm2 = cv::norm(xy2 - xy);

    Mat trans;
    if (norm1 < norm2) {
        trans = trans1;
    } else {
        trans = trans2;
    }
    return trans;
}

Mat find_none_flectives_similarity(const Mat &uv, const Mat &xy) {
    Mat A = Mat::zeros(2 * xy.rows, 4, CV_64FC1);
    Mat b = Mat::zeros(2 * xy.rows, 1, CV_64FC1);
    Mat x = Mat::zeros(4, 1, CV_64FC1);

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
    Mat trans_inv = (cv::Mat_<double>(3, 3) << x.at<double>(0), -x.at<double>(1), 0,
                     x.at<double>(1), x.at<double>(0), 0, x.at<double>(2), x.at<double>(3), 1);
    Mat trans = trans_inv.inv(cv::DECOMP_SVD);
    trans.at<double>(0, 2) = 0;
    trans.at<double>(1, 2) = 0;
    trans.at<double>(2, 2) = 1;

    return trans;
}

Mat get_similarity_transform(const vector<cv::Point2f> &src_pts,
                             const vector<cv::Point2f> &dest_pts, bool reflective) {
    Mat src((int)src_pts.size(), 2, CV_32FC1, (void *)(&src_pts[0].x));
    src.convertTo(src, CV_64FC1);

    Mat dst((int)dest_pts.size(), 2, CV_32FC1, (void *)(&dest_pts[0].x));
    dst.convertTo(dst, CV_64FC1);

    Mat trans = reflective ? find_similarity(src, dst) : find_none_flectives_similarity(src, dst);
    return trans(cv::Rect(0, 0, 2, trans.rows)).t();
}

net_err_t face_align(const Mat &image, Mat &aligned, const face_info_t &face_info, int width,
                     int height) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    assert(width == 96 || width == 112);
    assert(height == 112);
    if ((width != 96 && width != 112) || height != 112) {
        return RET_UNSUPPORTED_RESOLUTION;
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
    Mat tfm = get_similarity_transform(detect_points, reference_points, true);
    cv::warpAffine(image, aligned, tfm, cv::Size(width, height), cv::INTER_NEAREST);
    return RET_SUCCESS;
}

}  // namespace vision
}  // namespace qnn
