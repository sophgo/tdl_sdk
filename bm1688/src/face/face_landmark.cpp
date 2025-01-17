#include "face/face_landmark.hpp"

#include <log/Logger.hpp>
using nncompact::Net;
using nncompact::Tensor;

#define ONET

bmStatus_t FaceLandmark::setup() {
  setup_net(net_param_);
  tmp_resized_ = cv::Mat::zeros(input_geometry_, CV_8UC3);
  tmp_transposed_ = cv::Mat::zeros(input_geometry_, CV_8UC3);
  create_bgr_channels(tmp_bgr_planar_);

  LOG(INFO) << "landmark init done";
  return BM_COMMON_SUCCESS;
}

bmStatus_t FaceLandmark::detect(const std::vector<cv::Mat> &images,
                                const float threshold,
                                std::vector<FacePts> &facePts) {
  auto left_size = images.size();
  auto img_iter = images.cbegin();

#ifdef TIME_PRINT
  timer_.store_timestamp("total detection");
#endif
  while (left_size > 0) {
    int batch_size = get_fit_n(left_size);

#ifdef TIME_PRINT
    timer_.store_timestamp("preprocess");
#endif

    BM_CHECK_STATUS(preprocess(img_iter, batch_size));

#ifdef TIME_PRINT
    timer_.store_timestamp("preprocess");
    timer_.store_timestamp("forward");
#endif

    BM_CHECK_STATUS(forward());

#ifdef TIME_PRINT
    timer_.store_timestamp("forward");
    timer_.store_timestamp("postprocess");
#endif
    std::vector<cv::Size> frame_sizes;
    for (int i = 0; i < batch_size; i++) {
      frame_sizes.push_back((img_iter + i)->size());
    }
    BM_CHECK_STATUS(postprocess(frame_sizes, threshold, facePts));
    img_iter += batch_size;
    left_size -= batch_size;

#ifdef TIME_PRINT
    timer_.store_timestamp("postprocess");
#endif
  }

#ifdef TIME_PRINT
  timer_.store_timestamp("total detection");
  timer_.show();
  timer_.clear();
#endif

  return BM_COMMON_SUCCESS;
}

void FaceLandmark::preprocess(const cv::Mat &img, cv::Mat &tmp_resized,
                              cv::Mat &tmp_transposed,
                              std::vector<cv::Mat> &tmp_bgr,
                              std::vector<cv::Mat> &bgr) {
  create_bgr_channels(bgr);
  cv::resize(img, tmp_resized, input_geometry_, 0, 0, cv::INTER_NEAREST);

  cv::transpose(tmp_resized, tmp_transposed);
  bgr_split_scale1(tmp_transposed, temp_bgr_, bgr, means_, scales_);
}

bmStatus_t FaceLandmark::detect_direct(
    const std::vector<std::vector<cv::Mat>> &frame_bgrs, float threshold,
    const std::vector<cv::Size> &frame_sizes, std::vector<FacePts> &facePt) {
  int process_idx = 0;
  while (process_idx < frame_bgrs.size()) {
    int left_size = frame_bgrs.size() - process_idx;
    int n = get_fit_n(left_size);
    set_input_n(n);

    std::shared_ptr<nncompact::Tensor> input_tensor =
        net_->get_input_tensor(input_layer_);
    std::vector<cv::Size> batch_frame_size;
    for (int i = process_idx; i < process_idx + n; i++) {
      const std::vector<cv::Mat> &bgr = frame_bgrs[i];
      for (int j = 0; j < bgr.size(); j++) {
        input_tensor->from_mat(bgr[j], i - process_idx, j);
      }
      batch_frame_size.push_back(frame_sizes[i]);
    }
    net_->update_input_tensors();
    BM_CHECK_STATUS(forward());
    std::vector<FacePts> batch_res;
    BM_CHECK_STATUS(postprocess(batch_frame_size, threshold, batch_res));
    facePt.insert(facePt.end(), batch_res.begin(), batch_res.end());

    process_idx += n;
  }
}

bmStatus_t FaceLandmark::preprocess(
    std::vector<cv::Mat>::const_iterator &img_iter, int batch_size) {
  set_input_n(batch_size);

  // create_bgr_channels(bgr);
  std::shared_ptr<nncompact::Tensor> input_tensor =
      net_->get_input_tensor(input_layer_);
  for (int i = 0; i < batch_size; i++) {
    preprocess(*(img_iter + i), tmp_resized_, tmp_transposed_, temp_bgr_,
               tmp_bgr_planar_);
    for (int j = 0; j < tmp_bgr_planar_.size(); j++)
      input_tensor->from_mat(tmp_bgr_planar_[j], i, j);
  };
  // input_tensor->flush();
  net_->update_input_tensors();

  return BM_COMMON_SUCCESS;
}

bmStatus_t FaceLandmark::postprocess(const std::vector<cv::Size> &img_size,
                                     const float threshold,
                                     std::vector<FacePts> &facePts) {
  int batch_n = get_input_n();

  net_->update_output_tensors();
  nncompact::Tensor *prob_tensor =
      net_->get_output_tensor(net_->output_names_[2]).get();
  nncompact::Tensor *points_tensor =
      net_->get_output_tensor(net_->output_names_[1]).get();
  const bm_net_info_t *net_info = (const bm_net_info_t *)net_->get_net_info();

  int prob_idx = get_output_index(net_->output_names_[2]);
  int point_idx = get_output_index(net_->output_names_[1]);

  if (prob_idx == -1 || point_idx == -1) {
    LOG(FATAL) << "probidx:" << prob_idx << ",name:" << net_->output_names_[2]
               << ",pointidx:" << point_idx
               << ",name:" << net_->output_names_[1];
  } else {
    LOG(INFO) << "probidx:" << prob_idx << ",name:" << net_->output_names_[2]
              << ",pointidx:" << point_idx
              << ",name:" << net_->output_names_[1];
  }
  float *prob_out = prob_tensor->get_data();
  float *point_out = points_tensor->get_data();
  //  std::cout << "begin postprocess" << std::endl;
  // output_dtypes:0 conv6-3 1 conv6-2 2 prob
  for (int i = 0; i < batch_n; i++) {
    int width = img_size[i].width;
    int height = img_size[i].height;
    // batch dimension is [n, c, "h", w]
    const float *prob_data =
        prob_out + i * 2;  // prob_tensor->width(); //prob float
    const void *points_data;

    float prob_scale = 1;
    float point_scale = 1;
    if (net_info->output_dtypes[point_idx] == BM_INT8) {
      points_data = (int8_t *)(point_out) +
                    i * 10;  // points_tensor->width(); //point int8
      point_scale = net_info->output_scales[point_idx];
    } else if (net_info->output_dtypes[point_idx] == BM_FLOAT32) {
      points_data = point_out + i * 10;  // i * points_tensor->width();

    } else if (net_info->output_dtypes[point_idx] == BM_UINT8) {
      points_data = (uint8_t *)point_out + i * 10;
      point_scale = net_info->output_scales[point_idx];
    }

    FacePts facePt;
    if (net_info->output_dtypes[prob_idx] == BM_INT8) {
      int8_t *prob_data = (int8_t *)(prob_out) + i * 2;
      prob_scale = net_info->output_scales[prob_idx];
      facePt.score = prob_data[1] * prob_scale;
    } else if (net_info->output_dtypes[prob_idx] == BM_FLOAT32) {
      float *prob_data = prob_out + i * 2;
      facePt.score = prob_data[1] * prob_scale;
    } else if (net_info->output_dtypes[prob_idx] == BM_UINT8) {
      uint8_t *prob_data = (uint8_t *)(prob_out) + i * 2;
      prob_scale = net_info->output_scales[prob_idx];
      facePt.score = prob_data[1] * prob_scale;
    }

    const int8_t *points_data_char = (const int8_t *)points_data;
    const uint8_t *points_data_uchar = (const uint8_t *)points_data;
    const float *points_data_float = (const float *)points_data;

    //    std::cout << "get real points" << std::endl;
    LOG(INFO) << "batch " << i << ",score:" << facePt.score
              << ",probscale:" << prob_scale << ",pointscale:" << point_scale;

    if (facePt.score > threshold) {
      for (int j = 0; j < 5; ++j) {
#ifdef ONET
        // std::cout << "get points " << j << std::endl;
        if (net_info->output_dtypes[point_idx] == BM_INT8) {
          facePt.x.push_back(width * points_data_char[j] * point_scale - 1);
          facePt.y.push_back(height * points_data_char[j + 5] * point_scale -
                             1);
        } else if (net_info->output_dtypes[point_idx] == BM_UINT8) {
          facePt.x.push_back(width * points_data_uchar[j] * point_scale - 1);
          facePt.y.push_back(height * points_data_uchar[j + 5] * point_scale -
                             1);
        } else if (net_info->output_dtypes[point_idx] == BM_FLOAT32) {
          facePt.x.push_back(width * points_data_float[j] * point_scale - 1);
          facePt.y.push_back(height * points_data_float[j + 5] * point_scale -
                             1);
        }
        LOG(INFO) << "pt:" << j << ",x:" << facePt.x[j] << ",y:" << facePt.y[j];
#else
        if (net_info->output_dtypes[0] == BM_INT8) {
          facePt.x.push_back(width * points_data_char[2 * j] * point_scale - 1);
          facePt.y.push_back(
              height * points_data_char[2 * j + 1] * point_scale - 1);
        } else if (net_info->output_dtypes[0] == BM_FLOAT32) {
          facePt.x.push_back(width * points_data_float[2 * j] * point_scale -
                             1);
          facePt.y.push_back(
              height * points_data_float[2 * j + 1] * point_scale - 1);
        }
#endif
      }
    }
    facePts.push_back(facePt);
  }
  //  std::cout << "postprocess over" << std::endl;
  return BM_COMMON_SUCCESS;
}

// bmStatus_t FaceLandmark::postprocess(const std::vector<cv::Size> &img_size,
//                                      const float threshold,
//                                      std::vector<FacePts> &facePts) {
//   int batch_n = get_input_n();

//   net_->update_output_tensors();
//   nncompact::Tensor *prob_tensor =
//       net_->get_output_tensor(net_->output_names_[0]).get();
//       // net_->get_output_tensor(output_layers_[0]).get();
//   nncompact::Tensor *points_tensor =
//       net_->get_output_tensor(net_->output_names_[1]).get();
//       // net_->get_output_tensor(output_layers_[1]).get();
//   const bm_net_info_t *net_info = (const bm_net_info_t
//   *)net_->get_net_info(); float *prob_out = prob_tensor->get_data(); float
//   *point_out = points_tensor->get_data();
//   //  std::cout << "begin postprocess" << std::endl;
//   // output_dtypes:0 conv6-3 1 conv6-2 2 prob
//   for (int i = 0; i < batch_n; i++) {
//     int width = img_size[i].width;
//     int height = img_size[i].height;
//     // batch dimension is [n, c, "h", w]
//     const float *prob_data =
//         prob_out + i * 2; // prob_tensor->width(); //prob float
//     const void *points_data;
//     if (net_info->output_dtypes[0] == BM_INT8) {
//       points_data = (int8_t *)(point_out) +
//                     i * 10; // points_tensor->width(); //point int8
//     } else if (net_info->output_dtypes[0] == BM_FLOAT32) {
//       points_data = point_out + i * 10; // i * points_tensor->width();
//     } else if (net_info->output_dtypes[0] == BM_UINT8) {
//       points_data = (int8_t *)(point_out) +
//                     i * 10;
//     }
//     // const char *points_data =
//     //    (char*)(points_tensor->get_data()) + i * points_tensor->width();
//     //    //point int8
//     const int8_t *points_data_char = (const int8_t *)points_data;
//     const float *points_data_float = (const float *)points_data;
//     FacePts facePt;
//     facePt.score = prob_data[1];
//     //    std::cout << "get real points" << std::endl;
//     LOG(INFO) << "batch " << i << ",score:" << facePt.score;
//     float out_scale;
//     if (net_info->output_dtypes[0] == BM_INT8) {
//       out_scale = net_info->output_scales[0];
//     } else if (net_info->output_dtypes[0] == BM_FLOAT32) {
//       out_scale = 1;
//     } else {
//       LOG(FATAL) << "not implement yet";
//     }
//     if (facePt.score > threshold) {
//       for (int j = 0; j < 5; ++j) {
// #ifdef ONET
//         //        std::cout << "get points " << j << std::endl;
//         if (net_info->output_dtypes[0] == BM_INT8) {
//           facePt.x.push_back(width * points_data_char[j] * out_scale - 1);
//           facePt.y.push_back(height * points_data_char[j + 5] * out_scale -
//           1);
//         } else if (net_info->output_dtypes[0] == BM_FLOAT32) {
//           facePt.x.push_back(width * points_data_float[j] * out_scale - 1);
//           facePt.y.push_back(height * points_data_float[j + 5] * out_scale -
//           1);
//         }
// #else
//         if (net_info->output_dtypes[0] == BM_INT8) {
//           facePt.x.push_back(width * points_data_char[2 * j] * out_scale -
//           1); facePt.y.push_back(height * points_data_char[2 * j + 1] *
//           out_scale -
//                              1);
//         } else if (net_info->output_dtypes[0] == BM_FLOAT32) {
//           facePt.x.push_back(width * points_data_float[2 * j] * out_scale -
//           1); facePt.y.push_back(height * points_data_float[2 * j + 1] *
//           out_scale -
//                              1);
//         }
// #endif
//       }
//     }
//     facePts.push_back(facePt);
//   }
//   //  std::cout << "postprocess over" << std::endl;
//   return BM_COMMON_SUCCESS;
// }

///////////////////// BMMark Implementation //////////////////////////////
BMMark::BMMark(const stNetParam &param) {
  net_param_ = param;
  LOG(INFO) << "[BMMark] Initialize done!\n";
}

bmStatus_t BMMark::setup() {
  setup_net(net_param_);
  LOG(INFO) << "[BMMark] Setup done!\n";
  return BM_COMMON_SUCCESS;
}

bmStatus_t BMMark::preprocess(std::vector<cv::Mat>::const_iterator &img_iter,
                              int batch_size,
                              std::vector<IMGTransParam> &trans_params) {
  set_input_n(batch_size);
  trans_params.clear();
  cv::Mat resized;
  int hsize, wsize;
  for (int i = 0; i < batch_size; i++) {
    std::vector<cv::Mat> input_channels;
    wrap_input_layer(i, input_channels);

    auto h = (img_iter + i)->rows;
    auto w = (img_iter + i)->cols;
    if (h > w) {
      hsize = size;
      wsize = int(w * hsize / h);
    } else {
      wsize = size;
      hsize = int(h * wsize / w);
    }
    cv::resize(*(img_iter + i), resized, cv::Size(wsize, hsize));
    auto fx = static_cast<float>(w) / static_cast<float>(wsize);
    auto fy = static_cast<float>(h) / static_cast<float>(hsize);
    cv::Mat pad_img = cv::Mat::zeros(size, size, CV_8UC3);
    int ox = 0;
    int oy = 0;
    if (h > w) {
      ox = int(0.5 * size - 0.5 * wsize);
      oy = 0;
      // x, y, w, h
      resized.copyTo(pad_img(cv::Rect(ox, 0, wsize, hsize)));
    } else {
      oy = int(0.5 * size - 0.5 * hsize);
      ox = 0;
      resized.copyTo(pad_img(cv::Rect(0, oy, wsize, hsize)));
    }
    trans_params.emplace_back(w, h, fx, fy, ox, oy);

    cv::split(pad_img, temp_bgr_);  // cv::split is faster than vpp
    for (int k = 0; k < temp_bgr_.size(); k++) {
      temp_bgr_[k].convertTo(input_channels[k], input_channels[k].type(), 1.0,
                             0);
    }
  }

  net_->update_input_tensors();
  LOG(INFO) << "[BMMark] Pre-process done!\n";
  return BM_COMMON_SUCCESS;
}

bmStatus_t BMMark::postprocess(
    const std::vector<IMGTransParam>::const_iterator &iter,
    const float &threshold, std::vector<FacePts> &facePts) {
  int batch_n = get_input_n();
  net_->update_output_tensors();
  nncompact::Tensor *output_tensor =
      net_->get_output_tensor(output_layer_).get();
  const float *data = output_tensor->get_data();
  // (n, 1, 1, 12)
  std::vector<int> output_shape = output_tensor->get_shape();
  auto p = data;
  for (int i = 0; i < batch_n; i++) {
    FacePts pts;
    pts.score = p[1];
    if (p[1] > threshold) {
      for (int k = 1; k < 6; k++) {
        float x = p[2 * k] * static_cast<float>(size);
        x = (x - (iter + i)->ox) * (iter + i)->fx;
        float y = p[2 * k + 1] * static_cast<float>(size);
        y = (y - (iter + i)->oy) * (iter + i)->fy;
        pts.x.push_back(x);
        pts.y.push_back(y);
      }
    }
    facePts.push_back(pts);
    p = p + output_shape[3];
  }
  LOG(INFO) << "[BMMark] Post-process done!\n";
  return BM_COMMON_SUCCESS;
}

bmStatus_t BMMark::detect(const std::vector<cv::Mat> &imgs,
                          const float &threshold, std::vector<FacePts> &pts) {
  // check inputs
  if (imgs.empty() || !pts.empty()) {
    return BM_COMMON_INVALID_ARGS;
  }
  if (threshold < 0 || threshold > 1.0) {
    return BM_COMMON_OUT_OF_BOUND;
  }
  auto left_size = imgs.size();
  auto img_iter = imgs.cbegin();
#ifdef TIME_PRINT
  timer_.store_timestamp("total landmark detection");
#endif
  std::vector<IMGTransParam> trans_params;
  while (left_size > 0) {
    int batch_size = get_fit_n(left_size);
#ifdef TIME_PRINT
    timer_.store_timestamp("preprocess");
#endif
    BM_CHECK_STATUS(preprocess(img_iter, batch_size, trans_params));
#ifdef TIME_PRINT
    timer_.store_timestamp("preprocess");
    timer_.store_timestamp("forward");
#endif
    BM_CHECK_STATUS(forward());
#ifdef TIME_PRINT
    timer_.store_timestamp("forward");
    timer_.store_timestamp("postprocess");
#endif
    BM_CHECK_STATUS(postprocess(trans_params.cbegin(), threshold, pts));
#ifdef TIME_PRINT
    timer_.store_timestamp("postprocess");
#endif
    img_iter += batch_size;
    left_size -= batch_size;
  }
#ifdef TIME_PRINT
  timer_.store_timestamp("total landmark detection");
  timer_.show();
  timer_.clear();
#endif

  return BM_COMMON_SUCCESS;
}
bmStatus_t BMMark::preprocess(const cv::Mat &img, cv::Mat &tmp_resized,
                              cv::Mat &tmp_transposed,
                              std::vector<cv::Mat> &tmp_bgr,
                              std::vector<cv::Mat> &bgr) {
  create_bgr_channels(bgr);
  cv::resize(img, tmp_resized, input_geometry_, 0, 0, cv::INTER_NEAREST);
  bgr_split_scale1(tmp_resized, tmp_bgr, bgr, means_, scales_);
}

bmStatus_t BMMark::detect_direct(
    const std::vector<std::vector<cv::Mat>> &frame_bgrs, float threshold,
    const std::vector<cv::Size> &frame_sizes, std::vector<FacePts> &facePt) {
  std::vector<IMGTransParam> trans_paras;
  float w = input_geometry_.width;
  float h = input_geometry_.height;
  for (int i = 0; i < frame_bgrs.size(); i++) {
    IMGTransParam para;
    memset(&para, 0, sizeof(para));
    para.src_h = frame_sizes[i].height;
    para.src_w = frame_sizes[i].width;
    para.fx = frame_sizes[i].width / w;
    para.fy = frame_sizes[i].height / h;
    trans_paras.push_back(para);
  }
  return detect_direct(frame_bgrs, trans_paras, threshold, facePt);
}

bmStatus_t BMMark::preprocess(const cv::Mat &img, std::vector<cv::Mat> &dst,
                              IMGTransParam &trans_param) {
  create_bgr_channels(dst);
  cv::Mat resized;
  int hsize, wsize;

  auto h = img.rows;
  auto w = img.cols;
  if (h > w) {
    hsize = size;
    wsize = int(w * hsize / h);
  } else {
    wsize = size;
    hsize = int(h * wsize / w);
  }
  cv::resize(img, resized, cv::Size(wsize, hsize));
  trans_param.fx = static_cast<float>(w) / static_cast<float>(wsize);
  trans_param.fy = static_cast<float>(h) / static_cast<float>(hsize);
  cv::Mat pad_img = cv::Mat::zeros(size, size, CV_8UC3);
  int ox = 0;
  int oy = 0;
  if (h > w) {
    ox = int(0.5 * size - 0.5 * wsize);
    oy = 0;
    // x, y, w, h
    resized.copyTo(pad_img(cv::Rect(ox, 0, wsize, hsize)));
  } else {
    oy = int(0.5 * size - 0.5 * hsize);
    ox = 0;
    resized.copyTo(pad_img(cv::Rect(0, oy, wsize, hsize)));
  }
  trans_param.src_w = w;
  trans_param.src_h = h;
  trans_param.ox = ox;
  trans_param.oy = oy;

  cv::split(pad_img, temp_bgr_);
  for (int k = 0; k < temp_bgr_.size(); k++) {
    temp_bgr_[k].convertTo(dst[k], dst[k].type(), 1.0, 0);
  }

  return BM_COMMON_SUCCESS;
}

bmStatus_t BMMark::detect_direct(const std::vector<std::vector<cv::Mat>> &rois,
                                 const std::vector<IMGTransParam> &trans_params,
                                 const float &threshold,
                                 std::vector<FacePts> &facePt) {
  int index = 0;
  while (index < rois.size()) {
    int left_size = rois.size() - index;
    int batch = get_fit_n(left_size);
    set_input_n(batch);
    auto input_tensor = net_->get_input_tensor(input_layer_);
    for (int i = index; i < index + batch; i++) {
      const std::vector<cv::Mat> &bgr = rois[i];
      for (int j = 0; j < bgr.size(); j++) {
        input_tensor->from_mat(bgr[j], i - index, j);
      }
    }
    net_->update_input_tensors();
    BM_CHECK_STATUS(forward());
    std::vector<FacePts> batch_out;
    // slice
    BM_CHECK_STATUS(
        postprocess(trans_params.cbegin() + index, threshold, batch_out));
    facePt.insert(facePt.end(), batch_out.begin(), batch_out.end());
    index += batch;
  }
  return BM_COMMON_SUCCESS;
}
