#include "detection/obj_landmark.hpp"

#include <log/Logger.hpp>

bmStatus_t ObjectLandmark::setup() {
  setup_net(net_param_);
  LOG(INFO) << "landmark init done";
  return BM_COMMON_SUCCESS;
}

bmStatus_t ObjectLandmark::detect(const std::vector<cv::Mat> &images,
                                  const float threshold,
                                  std::vector<stObjPts> &facePts) {
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

bmStatus_t ObjectLandmark::detect_direct(
    const std::vector<std::vector<cv::Mat>> &frame_bgrs, float threshold,
    const std::vector<cv::Size> &frame_sizes, std::vector<stObjPts> &obj_pts) {
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
    std::vector<stObjPts> batch_res;
    BM_CHECK_STATUS(postprocess(batch_frame_size, threshold, batch_res));
    obj_pts.insert(obj_pts.end(), batch_res.begin(), batch_res.end());

    process_idx += n;
  }
}

bmStatus_t ObjectLandmark::postprocess(const std::vector<cv::Size> &img_size,
                                       const float threshold,
                                       std::vector<stObjPts> &facePts) {
  int batch_n = get_input_n();

  net_->update_output_tensors();
  nncompact::Tensor *prob_tensor =
      net_->get_output_tensor(output_layers_[0]).get();
  nncompact::Tensor *points_tensor =
      net_->get_output_tensor(output_layers_[1]).get();
  const bm_net_info_t *net_info = (const bm_net_info_t *)net_->get_net_info();
  float *prob_out = prob_tensor->get_data();
  float *point_out = points_tensor->get_data();
  int num_pts = points_tensor->width() / 2;
  for (int i = 0; i < batch_n; i++) {
    int width = img_size[i].width;
    int height = img_size[i].height;
    // batch dimension is [n, c, "h", w]
    const float *prob_data =
        prob_out + i * 2;  // prob_tensor->width(); //prob float
    const void *points_data;
    if (net_info->output_dtypes[0] == BM_INT8) {
      points_data =
          (int8_t *)(point_out) +
          i * points_tensor
                  ->batch_num_elems();  // points_tensor->width(); //point int8
    } else if (net_info->output_dtypes[0] == BM_FLOAT32) {
      points_data =
          point_out +
          i * points_tensor->batch_num_elems();  // i * points_tensor->width();
    }
    const int8_t *points_data_char = (const int8_t *)points_data;
    const float *points_data_float = (const float *)points_data;
    stObjPts obj_pts;
    obj_pts.score = prob_data[1];
    //    std::cout << "get real points" << std::endl;
    LOG(INFO) << "batch " << i << ",score:" << obj_pts.score;
    float out_scale;
    if (net_info->output_dtypes[0] == BM_INT8) {
      out_scale = net_info->output_scales[0];
    } else if (net_info->output_dtypes[0] == BM_FLOAT32) {
      out_scale = 1;
    } else {
      LOG(FATAL) << "not implement yet";
    }
    if (obj_pts.score > threshold) {
      for (int j = 0; j < num_pts; ++j) {
        if (net_info->output_dtypes[0] == BM_INT8) {
          obj_pts.x.push_back(width * points_data_char[2 * j] * out_scale - 1);
          obj_pts.y.push_back(height * points_data_char[2 * j + 1] * out_scale -
                              1);
        } else if (net_info->output_dtypes[0] == BM_FLOAT32) {
          obj_pts.x.push_back(width * points_data_float[2 * j] * out_scale - 1);
          obj_pts.y.push_back(
              height * points_data_float[2 * j + 1] * out_scale - 1);
        }
      }
    }
    facePts.push_back(obj_pts);
  }
  //  std::cout << "postprocess over" << std::endl;
  return BM_COMMON_SUCCESS;
}
