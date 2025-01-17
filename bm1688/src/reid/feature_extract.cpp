#include "reid/feature_extract.hpp"

#include <log/Logger.hpp>
bmStatus_t FeatureExtract::setup() {
  setup_net(net_param_);
  return BM_COMMON_SUCCESS;
}

bmStatus_t FeatureExtract::extract(const std::vector<cv::Mat> &images,
                                   std::vector<std::vector<float>> &features) {
  int left_size = images.size();
  auto img_iter = images.cbegin();

#ifdef TIME_PRINT
  timer_.store_timestamp("total extraction");
#endif
  while (left_size > 0) {
    int batch_size = get_fit_n(left_size);

#ifdef TIME_PRINT
    timer_.store_timestamp("preprocess");
#endif
    LOG(INFO) << images[0].size() << "," << img_iter->size();
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

    postprocess(features);
    img_iter += batch_size;
    left_size -= batch_size;

#ifdef TIME_PRINT
    timer_.store_timestamp("postprocess");
#endif
  }

#ifdef TIME_PRINT
  timer_.store_timestamp("total extraction");
  timer_.show();
  timer_.clear();
#endif

  return BM_COMMON_SUCCESS;
}

bmStatus_t FeatureExtract::extract_direct(
    const std::vector<std::vector<cv::Mat>> &frame_bgrs,
    std::vector<std::vector<float>> &features) {
  int left_size = frame_bgrs.size();
  int process_idx = 0;
  std::shared_ptr<nncompact::Tensor> input_tensor =
      net_->get_input_tensor(input_layer_);
  while (left_size > 0) {
    int n = get_fit_n(left_size);
    set_input_n(n);
    for (int i = process_idx; i < process_idx + n; i++) {
      const std::vector<cv::Mat> &bgr = frame_bgrs[i];
      for (int j = 0; j < bgr.size(); j++) {
        input_tensor->from_mat(bgr[j], i - process_idx, j);
      }
    }
    // input_tensor->flush();//flush is not necessary,the data has been copied
    // to device memory
    net_->update_input_tensors();
    BM_CHECK_STATUS(forward());
    std::vector<std::vector<float>> batch_res;
    BM_CHECK_STATUS(postprocess(batch_res));
    features.insert(features.end(), batch_res.begin(), batch_res.end());
    left_size -= n;
    process_idx += n;
  }
}

bmStatus_t FeatureExtract::postprocess(
    std::vector<std::vector<float>> &features) {
  int batch_n = get_input_n();
  net_->update_output_tensors();
  nncompact::Tensor *output_tensor =
      net_->get_output_tensor(output_layer_).get();
  const bm_net_info_t *net_info = (const bm_net_info_t *)net_->get_net_info();
  // const float* data = output_tensor->get_data();

  float *data_float = (float *)output_tensor->get_data();
  const int8_t *data_char = (int8_t *)data_float;
  std::vector<int> output_shape = output_tensor->get_shape();
  int batch_elem_num = output_shape[3] * output_shape[2] * output_shape[1];
  for (int i = 0; i < batch_n; i++) {
    // batch dimension is [n, c, "h", w]
    LOG(INFO) << "outshape:" << output_shape[0] << " " << output_shape[1] << " "
              << output_shape[2] << " " << output_shape[3];
    const int8_t *begin_char = data_char + i * batch_elem_num;
    const float *begin_float = data_float + i * batch_elem_num;
    if (net_info->output_dtypes[0] == BM_INT8) {
      std::vector<float> temp;
      for (int j = 0; j < batch_elem_num; j++) {
        temp.push_back(
            begin_char[j]);  // do not need to scale,would be normalized outside
      }
      features.push_back(temp);
    } else if (net_info->output_dtypes[0] == BM_UINT8) {
      std::vector<float> temp;
      const unsigned char *uint_ptr = (const unsigned char *)(begin_char);
      for (int j = 0; j < batch_elem_num; j++) {
        temp.push_back(
            uint_ptr[j]);  // do not need to scale,would be normalized outside
      }
      features.push_back(temp);

    } else if (net_info->output_dtypes[0] == BM_FLOAT32) {
      const float *end_float = begin_float + batch_elem_num;
      features.emplace_back(begin_float, end_float);
    } else {
      LOG(FATAL) << "not implement yet";
    }
  }

  return BM_COMMON_SUCCESS;
}
