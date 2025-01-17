#include "classify/carplate_ocr.hpp"

#include <log/Logger.hpp>

const std::string PROVINCE_STR[] = {
    "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏",
    "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼",
    "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"};
const std::string CITY_STR[] = {"A", "B", "C", "D", "E", "F", "G", "H",
                                "J", "K", "L", "M", "N", "P", "Q", "R",
                                "S", "T", "U", "V", "W", "X", "Y", "Z"};
const std::string NUM_STR[] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                               "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
                               "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
                               "W", "X", "Y", "Z", "I", "O", "-"};
bmStatus_t CarplateOCR::setup() {
  setup_net(net_param_);
  LOG(INFO) << "landmark init done";
  return BM_COMMON_SUCCESS;
}

bmStatus_t CarplateOCR::detect(const std::vector<cv::Mat> &images,
                               const float threshold,
                               std::vector<stCarplate> &carplates) {
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
    BM_CHECK_STATUS(postprocess(threshold, carplates));
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

bmStatus_t CarplateOCR::detect_direct(
    const std::vector<std::vector<cv::Mat>> &frame_bgrs, float threshold,
    std::vector<stCarplate> &carplates) {
  int process_idx = 0;
  while (process_idx < frame_bgrs.size()) {
    int left_size = frame_bgrs.size() - process_idx;
    int n = get_fit_n(left_size);
    set_input_n(n);

    std::shared_ptr<nncompact::Tensor> input_tensor =
        net_->get_input_tensor(input_layer_);

    for (int i = process_idx; i < process_idx + n; i++) {
      const std::vector<cv::Mat> &bgr = frame_bgrs[i];
      for (int j = 0; j < bgr.size(); j++) {
        input_tensor->from_mat(bgr[j], i - process_idx, j);
      }
    }
    net_->update_input_tensors();
    BM_CHECK_STATUS(forward());
    std::vector<stCarplate> batch_res;
    BM_CHECK_STATUS(postprocess(threshold, batch_res));
    carplates.insert(carplates.end(), batch_res.begin(), batch_res.end());

    process_idx += n;
  }
}

bmStatus_t CarplateOCR::postprocess(const float threshold,
                                    std::vector<stCarplate> &carplates) {
  int batch_n = get_input_n();
  net_->update_output_tensors();
  nncompact::Tensor *province_tensor =
      net_->get_output_tensor(output_layers_[0]).get();
  nncompact::Tensor *city_tensor =
      net_->get_output_tensor(output_layers_[1]).get();
  nncompact::Tensor *num_tensor =
      net_->get_output_tensor(output_layers_[2]).get();

  nncompact::Tensor *plate_type_tensor =
      net_->get_output_tensor(output_layers_[3]).get();
  const bm_net_info_t *net_info = (const bm_net_info_t *)net_->get_net_info();
  int province_batch_offset = province_tensor->batch_num_elems();
  int city_batch_offset = city_tensor->batch_num_elems();
  int num_batch_offset = num_tensor->batch_num_elems();
  int plate_type_batch_offset = plate_type_tensor->batch_num_elems();
  for (int i = 0; i < batch_n; i++) {
    const float *province_prob =
        province_tensor->get_data() + i * province_batch_offset;
    const float *city_prob = city_tensor->get_data() + i * city_batch_offset;
    const float *num_prob = num_tensor->get_data() + i * num_batch_offset;
    const float *plate_type_prob =
        plate_type_tensor->get_data() + i * plate_type_batch_offset;
    stCarplate carplate;
    int max_province_idx = std::distance(
        province_prob,
        std::max_element(province_prob, province_prob + province_batch_offset));
    int max_city_idx = std::distance(
        city_prob, std::max_element(city_prob, city_prob + city_batch_offset));
    carplate.plate_type = std::distance(
        plate_type_prob,
        std::max_element(plate_type_prob,
                         plate_type_prob + plate_type_batch_offset));

    carplate.str_labels += PROVINCE_STR[max_province_idx];
    carplate.str_labels += CITY_STR[max_city_idx];
    carplate.scores.push_back(province_prob[max_province_idx]);
    carplate.scores.push_back(city_prob[max_city_idx]);
    LOG(INFO) << "score:" << province_prob[max_province_idx] << ","
              << city_prob[max_city_idx];
    for (int t = 0; t < num_tensor->height(); t++) {
      const float *num_prob_t = num_prob + t * num_tensor->width();
      int max_num_idx = std::distance(
          num_prob_t,
          (std::max_element(num_prob_t, num_prob_t + num_tensor->width())));
      if (t == num_tensor->height() - 1 && NUM_STR[max_num_idx] == "-")
        continue;  // skip it
      carplate.scores.push_back(num_prob_t[max_num_idx]);
      carplate.str_labels += NUM_STR[max_num_idx];
    }
    LOG(INFO) << "province_idx:" << max_province_idx
              << ",batch_offset:" << province_tensor->batch_num_elems()
              << ",max_city_idx:" << max_city_idx
              << ",batch_offset:" << city_tensor->batch_num_elems()
              << ",label:" << carplate.str_labels;
    carplate.allok = true;
    for (int j = 0; j < carplate.scores.size(); j++) {
      if (carplate.scores[j] < threshold) {
        carplate.allok = false;
        break;
      }
    }
    carplates.push_back(carplate);
  }
  //  std::cout << "postprocess over" << std::endl;
  return BM_COMMON_SUCCESS;
}
