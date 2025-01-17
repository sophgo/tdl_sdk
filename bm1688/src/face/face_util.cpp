#include "face/face_util.hpp"

inline float calc_sqrt(const std::vector<float> &feature) {
  float a = 0;
  for (int i = 0; i < feature.size(); ++i) {
    a += feature[i] * feature[i];
  }
  return sqrt(a);
}

float calc_cosine(const std::vector<float> &feature1,
                  const std::vector<float> &feature2) {
  float a = 0;
  for (int i = 0; i < feature1.size(); ++i) {
    a += feature1[i] * feature2[i];
  }
  float b = calc_sqrt(feature1) * calc_sqrt(feature2);
  return a / b;
}

bool compareArea(const FaceRect &a, const FaceRect &b) {
  return (a.x2 - a.x1) * (a.y2 - a.y1) > (b.x2 - b.x1) * (b.y2 - b.y1);
}

bool compareScore(const FaceRect &a, const FaceRect &b) {
  return a.score > b.score;
}

cv::Mat tformfwd(const cv::Mat &trans, const cv::Mat &uv) {
  cv::Mat uv_h = cv::Mat::ones(uv.rows, 3, CV_64FC1);
  uv.copyTo(uv_h(cv::Rect(0, 0, 2, uv.rows)));
  cv::Mat xv_h = uv_h * trans.t();
  return xv_h(cv::Rect(0, 0, 2, uv.rows));
}

cv::Mat find_none_flectives_similarity(const cv::Mat &xy, const cv::Mat &uv) {
  cv::Mat A = cv::Mat::zeros(2 * xy.rows, 4, CV_64FC1);
  cv::Mat b = cv::Mat::zeros(2 * xy.rows, 1, CV_64FC1);
  cv::Mat x = cv::Mat::zeros(4, 1, CV_64FC1);

  xy(cv::Rect(0, 0, 1, xy.rows)).copyTo(A(cv::Rect(0, 0, 1, xy.rows))); // x
  xy(cv::Rect(1, 0, 1, xy.rows)).copyTo(A(cv::Rect(1, 0, 1, xy.rows))); // y
  A(cv::Rect(2, 0, 1, xy.rows)).setTo(1.);

  xy(cv::Rect(1, 0, 1, xy.rows))
      .copyTo(A(cv::Rect(0, xy.rows, 1, xy.rows))); // y
  xy(cv::Rect(0, 0, 1, xy.rows))
      .copyTo(A(cv::Rect(1, xy.rows, 1, xy.rows))); //-x
  A(cv::Rect(1, xy.rows, 1, xy.rows)) *= -1;
  A(cv::Rect(3, xy.rows, 1, xy.rows)).setTo(1.);

  uv(cv::Rect(0, 0, 1, uv.rows)).copyTo(b(cv::Rect(0, 0, 1, uv.rows)));
  uv(cv::Rect(1, 0, 1, uv.rows)).copyTo(b(cv::Rect(0, uv.rows, 1, uv.rows)));

  cv::solve(A, b, x, cv::DECOMP_SVD);
  cv::Mat trans = (cv::Mat_<double>(3, 3) << x.at<double>(0), x.at<double>(1),
                   x.at<double>(2), -x.at<double>(1), x.at<double>(0),
                   x.at<double>(3), 0, 0, 1);

  return trans;
}

cv::Mat find_similarity(const cv::Mat &xy, const cv::Mat &uv) {
  cv::Mat trans1 = find_none_flectives_similarity(xy, uv);
  cv::Mat xy_reflect = xy.clone();
  xy_reflect(cv::Rect(0, 0, 1, xy.rows)) *= -1;
  cv::Mat trans2r = find_none_flectives_similarity(xy_reflect, uv);
  cv::Mat reflect = (cv::Mat_<double>(3, 3) << -1, 0, 0, 0, 1, 0, 0, 0, 1);

  cv::Mat trans2 = trans2r * reflect;
  cv::Mat xy1 = tformfwd(trans1, xy);
  double norm1 = cv::norm(xy1 - uv);

  cv::Mat xy2 = tformfwd(trans2, xy);
  double norm2 = cv::norm(xy2 - uv);

  cv::Mat trans;
  if (norm1 < norm2) {
    trans = trans1;
  } else {
    trans = trans2;
  }
  return trans;
}

cv::Mat get_similarity_transform(const std::vector<cv::Point2f> &src_points,
                                 const std::vector<cv::Point2f> &dst_points,
                                 bool reflective = true) {
  cv::Mat trans;
  cv::Mat src((int)src_points.size(), 2, CV_32FC1, (void *)(&src_points[0].x));
  src.convertTo(src, CV_64FC1);

  cv::Mat dst((int)dst_points.size(), 2, CV_32FC1, (void *)(&dst_points[0].x));
  dst.convertTo(dst, CV_64FC1);

  if (reflective) {
    trans = find_similarity(src, dst);
  } else {
    trans = find_none_flectives_similarity(src, dst);
  }
  cv::Mat trans_cv2 = trans(cv::Rect(0, 0, trans.rows, 2));
  return trans_cv2;
}

cv::Mat align_face(const cv::Mat &src, const FacePts facePt, int width,
                   int height) {
  const int ReferenceWidth = 96;
  const int ReferenceHeight = 112;
  std::vector<cv::Point2f> detect_points;
  for (int j = 0; j < 5; ++j) {
    cv::Point2f e;
    e.x = facePt.x[j];
    e.y = facePt.y[j];
    detect_points.push_back(e);
  }
  std::vector<cv::Point2f> reference_points;
  reference_points.push_back(cv::Point2f(30.29459953, 51.69630051));
  reference_points.push_back(cv::Point2f(65.53179932, 51.50139999));
  reference_points.push_back(cv::Point2f(48.02519989, 71.73660278));
  reference_points.push_back(cv::Point2f(33.54930115, 92.36550140));
  reference_points.push_back(cv::Point2f(62.72990036, 92.20410156));
  for (int j = 0; j < 5; ++j) {
    reference_points[j].x += (width - ReferenceWidth) / 2.0f;
    reference_points[j].y += (height - ReferenceHeight) / 2.0f;
  }
  cv::Mat tfm = get_similarity_transform(
      detect_points, reference_points,
      false); // reflective affine transfrom makes no difference
  cv::Mat aligned_face;
  warpAffine(src, aligned_face, tfm, cv::Size(width, height));
  return aligned_face;
}

cv::Mat align_eye(const cv::Mat &src, const FaceRect rect, const FacePts facePt,
                  int width, int height) {
  const int ReferenceWidth = 96;
  const int ReferenceHeight = 112;
  std::vector<cv::Point2f> detect_points;
  for (int j = 0; j < 2; ++j) {
    cv::Point2f e;
    e.x = facePt.x[j];
    e.y = facePt.y[j];
    detect_points.push_back(e);
  }
  std::vector<cv::Point2f> reference_points;
  reference_points.push_back(cv::Point2f(30.29459953, 51.69630051));
  reference_points.push_back(cv::Point2f(65.53179932, 51.50139999));
  for (int j = 0; j < 2; ++j) {
    // reference_points[j].x += (width - ReferenceWidth) / 2.0f;
    // reference_points[j].y += (height - ReferenceHeight) / 2.0f;
    reference_points[j].x += 8;
    reference_points[j].y -= 6;
  }
  cv::Mat tfm = get_similarity_transform(
      detect_points, reference_points,
      false); // reflective affine transfrom makes no difference
  cv::Mat aligned_face;
  warpAffine(src, aligned_face, tfm, cv::Size(width, height));
  return aligned_face;
}

cv::Mat align_face_to_dest(const cv::Mat &src, cv::Mat &aligned,
                           const FacePts &facePt, int width, int height) {
  const int ReferenceWidth = 96;
  const int ReferenceHeight = 112;
  std::vector<cv::Point2f> detect_points;
  for (int j = 0; j < 5; ++j) {
    cv::Point2f e;
    e.x = facePt.x[j];
    e.y = facePt.y[j];
    detect_points.push_back(e);
  }
  std::vector<cv::Point2f> reference_points;
  reference_points.push_back(cv::Point2f(30.29459953, 51.69630051));
  reference_points.push_back(cv::Point2f(65.53179932, 51.50139999));
  reference_points.push_back(cv::Point2f(48.02519989, 71.73660278));
  reference_points.push_back(cv::Point2f(33.54930115, 92.36550140));
  reference_points.push_back(cv::Point2f(62.72990036, 92.20410156));
  for (int j = 0; j < 5; ++j) {
    reference_points[j].x += (width - ReferenceWidth) / 2.0f;
    reference_points[j].y += (height - ReferenceHeight) / 2.0f;
  }
  cv::Mat tfm = get_similarity_transform(
      detect_points, reference_points,
      false); // reflective affine transfrom makes no difference

  warpAffine(src, aligned, tfm, cv::Size(width, height));
  return tfm;
}

cv::Mat calc_transform_matrix(const FacePts &face_pt,
                              cv::Size dst_size /*= cv::Size(112,112)*/) {

  const int ReferenceWidth = 96;
  const int ReferenceHeight = 112;

  std::vector<cv::Point2f> detect_points;
  for (int j = 0; j < 5; ++j) {
    cv::Point2f e;
    e.x = face_pt.x[j];
    e.y = face_pt.y[j];
    detect_points.push_back(e);
  }
  std::vector<cv::Point2f> reference_points;
  reference_points.push_back(cv::Point2f(30.29459953, 51.69630051));
  reference_points.push_back(cv::Point2f(65.53179932, 51.50139999));
  reference_points.push_back(cv::Point2f(48.02519989, 71.73660278));
  reference_points.push_back(cv::Point2f(33.54930115, 92.36550140));
  reference_points.push_back(cv::Point2f(62.72990036, 92.20410156));
  for (int j = 0; j < 5; ++j) {
    reference_points[j].x += (dst_size.width - ReferenceWidth) / 2.0f;
    reference_points[j].y += (dst_size.height - ReferenceHeight) / 2.0f;
  }
  //cv::Mat tfm0 =
  //    similarTransform(cv::Mat(detect_points), cv::Mat(reference_points));
  //return tfm0;
  cv::Mat tfm =
      get_similarity_transform(detect_points, reference_points, false);
  return tfm;
}

// pitch,yaw,roll,degrees
std::vector<float> face_pose_estimate(const FacePts &facept, cv::Size imsize) {

  // 2D image points. If you change the image, you need to change vector
  std::vector<cv::Point2d> image_points;
  float offx = imsize.width/2 - facept.x[2];
  float offy = imsize.height/2 - facept.y[2];  
  for (int i = 0; i < facept.x.size(); i++)
    image_points.push_back(cv::Point2d(facept.x[i]+offx, facept.y[i]+offy));
  // std::cout << "image points size:" << image_points.size() << std::endl;
  // 3D model points.
  std::vector<cv::Point3d> model_points;
  model_points.push_back(
      cv::Point3d(-165.0f, -170.0f, -135.0f)); // Left eye left corner
  model_points.push_back(
      cv::Point3d(165.0f, -170.0f, -135.0f)); // Right eye right corner
  model_points.push_back(cv::Point3d(0.0f, 0.0f, 0.0f)); // Nose tip

  model_points.push_back(
      cv::Point3d(-150.0f, 150.0f, -125.0f)); // left mouth corner
  model_points.push_back(
      cv::Point3d(150.0f, 150.0f, -125.0f)); // right mouth corner

  // Camera internals
  double focal_length =
      imsize.width / 2 /
      tan((60 / 2) * (M_PI / 180)); // Approximate focal length.

  cv::Point2d center = cv::Point2d(imsize.width / 2, imsize.height / 2);
  cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x,
                           0, focal_length, center.y, 0, 0, 1);

  cv::Mat dist_coeffs = cv::Mat::zeros(
      4, 1, cv::DataType<double>::type); // Assuming no lens distortion

  // std::cout << "Camera Matrix " << std::endl << camera_matrix << std::endl;
  // Output rotation and translation
  cv::Mat rotation_vector; // Rotation in axis-angle form
  cv::Mat translation_vector;
  cv::Mat rvec_matrix;
  cv::Mat proj_matrix;
  cv::Mat eulerAngles;
  cv::Mat cameraMatrix;
  cv::Mat rotMatrix;
  cv::Mat transVect;
  eulerAngles = cv::Mat(3, 1, CV_64FC1);

  // Solve for pose
  bool isok = cv::solvePnP(model_points, image_points, camera_matrix,
                           dist_coeffs, rotation_vector, translation_vector,
                           false, cv::SOLVEPNP_UPNP);
  if (!isok)
    assert(0);
  // std::cout << "pnp done,rotvec:\n" << rotation_vector << std::endl;
  // cout << nose_end_point2D << endl;
  cv::Rodrigues(rotation_vector, rvec_matrix, cv::noArray());
  // std::cout << "Rodrigues Vector" << std::endl << rvec_matrix << std::endl;

  cv::hconcat(rvec_matrix, translation_vector, proj_matrix);
  // std::cout << "proj_matrix Vector" << std::endl << proj_matrix << std::endl;

  cv::decomposeProjectionMatrix(proj_matrix, cameraMatrix, rotMatrix, transVect,
                                cv::noArray(), cv::noArray(), cv::noArray(),
                                eulerAngles);
  // std::cout << "eulerAngles Vector" << std::endl << eulerAngles << std::endl;

  std::vector<float> res;
  for (int i = 0; i < 3; i++)
    res.push_back(eulerAngles.at<double>(i));
  double *prot = rotation_vector.ptr<double>(0);
  float score = 1 - (fabs(prot[0]) + fabs(prot[1]) + fabs(prot[2])) * 1.25;
  res.push_back(rvec_matrix.ptr<double>(2)[2]);
  // res.push_back(score);
  
  return res;
}

cv::Mat meanAxis0(const cv::Mat &src) {
  int num = src.rows;
  int dim = src.cols;

  // x1 y1
  // x2 y2

  cv::Mat output(1, dim, CV_32F);
  for (int i = 0; i < dim; i++) {
    float sum = 0;
    for (int j = 0; j < num; j++) {
      sum += src.at<float>(j, i);
    }
    output.at<float>(0, i) = sum / num;
  }

  return output;
}

cv::Mat elementwiseMinus(const cv::Mat &A, const cv::Mat &B) {
  cv::Mat output(A.rows, A.cols, A.type());

  assert(B.cols == A.cols);
  if (B.cols == A.cols) {
    for (int i = 0; i < A.rows; i++) {
      for (int j = 0; j < B.cols; j++) {
        output.at<float>(i, j) = A.at<float>(i, j) - B.at<float>(0, j);
      }
    }
  }
  return output;
}

cv::Mat varAxis0(const cv::Mat &src) {
  cv::Mat temp_ = elementwiseMinus(src, meanAxis0(src));
  cv::multiply(temp_, temp_, temp_);
  return meanAxis0(temp_);
}

int MatrixRank(cv::Mat M) {
  cv::Mat w, u, vt;
  cv::SVD::compute(M, w, u, vt);
  cv::Mat1b nonZeroSingularValues = w > 0.0001;
  int rank = countNonZero(nonZeroSingularValues);
  return rank;
}

//    References
//    ----------
//    .. [1] "Least-squares estimation of transformation parameters between two
//    point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
//
//    """
//
//    Anthor:Jack Yu
cv::Mat similarTransform(cv::Mat src, cv::Mat dst) {
  int num = src.rows;
  int dim = src.cols;
  cv::Mat src_mean = meanAxis0(src);
  cv::Mat dst_mean = meanAxis0(dst);
  cv::Mat src_demean = elementwiseMinus(src, src_mean);
  cv::Mat dst_demean = elementwiseMinus(dst, dst_mean);
  cv::Mat A = (dst_demean.t() * src_demean) / static_cast<float>(num);
  cv::Mat d(dim, 1, CV_32F);
  d.setTo(1.0f);
  if (cv::determinant(A) < 0) {
    d.at<float>(dim - 1, 0) = -1;
  }
  cv::Mat T = cv::Mat::eye(dim + 1, dim + 1, CV_32F);
  cv::Mat U, S, V;
  cv::SVD::compute(A, S, U, V);

  // the SVD function in opencv differ from scipy .

  int rank = MatrixRank(A);
  if (rank == 0) {
    assert(rank == 0);

  } else if (rank == dim - 1) {
    if (cv::determinant(U) * cv::determinant(V) > 0) {
      T.rowRange(0, dim).colRange(0, dim) = U * V;
    } else {
      //            s = d[dim - 1]
      //            d[dim - 1] = -1
      //            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
      //            d[dim - 1] = s
      int s = d.at<float>(dim - 1, 0) = -1;
      d.at<float>(dim - 1, 0) = -1;

      T.rowRange(0, dim).colRange(0, dim) = U * V;
      cv::Mat diag_ = cv::Mat::diag(d);
      cv::Mat twp = diag_ * V; // np.dot(np.diag(d), V.T)
      cv::Mat B = cv::Mat::zeros(3, 3, CV_8UC1);
      cv::Mat C = B.diag(0);
      T.rowRange(0, dim).colRange(0, dim) = U * twp;
      d.at<float>(dim - 1, 0) = s;
    }
  } else {
    cv::Mat diag_ = cv::Mat::diag(d);
    cv::Mat twp = diag_ * V.t(); // np.dot(np.diag(d), V.T)
    cv::Mat res = U * twp;       // U
    T.rowRange(0, dim).colRange(0, dim) = -U.t() * twp;
  }
  cv::Mat var_ = varAxis0(src_demean);
  float val = cv::sum(var_).val[0];
  cv::Mat res;
  cv::multiply(d, S, res);
  float scale = 1.0 / val * cv::sum(res).val[0];
  T.rowRange(0, dim).colRange(0, dim) =
      -T.rowRange(0, dim).colRange(0, dim).t();
  cv::Mat temp1 = T.rowRange(0, dim).colRange(0, dim); // T[:dim, :dim]
  cv::Mat temp2 = src_mean.t();                        // src_mean.T
  cv::Mat temp3 = temp1 * temp2; // np.dot(T[:dim, :dim], src_mean.T)
  cv::Mat temp4 = scale * temp3;
  T.rowRange(0, dim).colRange(dim, dim + 1) = -(temp4 - dst_mean.t());
  T.rowRange(0, dim).colRange(0, dim) *= scale;
  return T;
}
