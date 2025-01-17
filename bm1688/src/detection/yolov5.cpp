#include "detection/yolov5.hpp"

#include <log/Logger.hpp>

bmStatus_t YOLOV5::setup() {
  setup_net(net_param_);
  nms_threshold_ = 0.5f;
  return BM_COMMON_SUCCESS;
}

bmStatus_t YOLOV5::detect(const std::vector<cv::Mat> &images,
                          const float threshold,
                          std::vector<std::vector<ObjectBox>> &results) {
  auto left_size = images.size();
  auto img_iter = images.cbegin();
  LOG(INFO) << "retina begin";
#ifdef TIME_PRINT
  timer_.store_timestamp("total detection");
#endif
  while (left_size > 0) {
    int batch_size = get_fit_n(left_size);

#ifdef TIME_PRINT
    timer_.store_timestamp("preprocess");
#endif
    LOG(INFO) << "preprocess,batch:" << batch_size;
    BM_CHECK_STATUS(preprocess_opencv(img_iter, batch_size));

#ifdef TIME_PRINT
    timer_.store_timestamp("preprocess");
    timer_.store_timestamp("forward");
#endif
    // std::cout<<"forwardstart"<<std::endl;
    BM_CHECK_STATUS(forward());

#ifdef TIME_PRINT
    timer_.store_timestamp("forward");
    timer_.store_timestamp("postprocess");
#endif
    std::vector<cv::Size> frame_sizes;
    for (int i = 0; i < batch_size; i++) {
      frame_sizes.push_back((img_iter + i)->size());
    }
    // std::cout<<"forward done"<<std::endl;
    if (output_scales_.size() == 1) {
      BM_CHECK_STATUS(
          postprocess(frame_sizes, threshold, batch_rescale_params_, results));
    } else if (output_scales_.size() == 3) {
      BM_CHECK_STATUS(postprocess_3out(frame_sizes, threshold,
                                       batch_rescale_params_, results));
    } else {
      LOG(FATAL) << "output size error,numout:" << output_scales_.size();
    }

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

bmStatus_t YOLOV5::detect_direct(
    const std::vector<std::vector<cv::Mat>> &frame_bgrs, float threshold,
    const std::vector<std::vector<float>> &frame_rescale_params,
    const std::vector<cv::Size> &frame_sizes,
    std::vector<std::vector<ObjectBox>> &results) {
  int left_size = frame_bgrs.size();
  int process_idx = 0;

  while (left_size > 0) {
    int n = get_fit_n(left_size);
    set_input_n(n);

    std::shared_ptr<nncompact::Tensor> input_tensor =
        net_->get_input_tensor(input_layer_);
    std::vector<std::vector<float>> batch_rescale_params;
    std::vector<cv::Size> batch_frame_sizes;

    for (int i = process_idx; i < process_idx + n; i++) {
      const std::vector<cv::Mat> &bgr = frame_bgrs[i];
      for (int j = 0; j < bgr.size(); j++) {
        input_tensor->from_mat(bgr[j], i - process_idx, j);
      }
      batch_rescale_params.push_back(frame_rescale_params[i]);
      batch_frame_sizes.push_back(frame_sizes[i]);
    }

    net_->update_input_tensors();
    BM_CHECK_STATUS(forward());

    std::vector<std::vector<ObjectBox>> batch_res;
    if (output_scales_.size() == 1) {
      BM_CHECK_STATUS(postprocess(batch_frame_sizes, threshold,
                                  batch_rescale_params, batch_res));
    } else if (output_scales_.size() == 3) {
      BM_CHECK_STATUS(postprocess_3out(batch_frame_sizes, threshold,
                                       batch_rescale_params, batch_res));
    } else {
      LOG(FATAL) << "output size error,numout:" << output_scales_.size();
    }

    results.insert(results.end(), batch_res.begin(), batch_res.end());
    left_size -= n;
    process_idx += n;
  }
  return BM_COMMON_SUCCESS;
}
bmStatus_t YOLOV5::postprocess(
    const std::vector<cv::Size> &frame_sizes, const float threshold,
    const std::vector<std::vector<float>> &frame_scale_params,
    std::vector<std::vector<ObjectBox>> &results) {
  int batch_n = get_input_n();
  double tick0 = cv::getTickCount();
  net_->update_output_tensors();
  double tick1 = cv::getTickCount();

  nncompact::Tensor *det_output_tensor =
      net_->get_output_tensor(output_layer_).get();
  std::vector<int> shape = det_output_tensor->get_shape();
  float out_scale = output_scales_[output_layer_];
  LOG(INFO) << "detshape,batch:" << batch_n << "output:" << shape[0] << ","
            << shape[1] << "," << shape[2] << "," << shape[3]
            << ",out_scale:" << out_scale;
  const float *ptr_float = det_output_tensor->get_data();
  const int8_t *ptr_int8 = (const int8_t *)ptr_float;
  int num_anchors = shape[2];
  int num_elemets = shape[3];

  // element data:x,y,w,h,obj,cls1,cls2,..,clsn
  for (int b = 0; b < batch_n; b++) {
    results.emplace_back();
    auto &last_result = results.back();
    std::vector<ObjectBox> total_box;
    const std::vector<float> &rescale_param = frame_scale_params[b];
    float im_scale_w = rescale_param[0];
    float im_scale_h = rescale_param[1];
    float pad_x = rescale_param[2];
    float pad_y = rescale_param[3];
    // TODO:int8 output should multiply outscale
    for (int i = 0; i < num_anchors; ++i) {
      if ((is_int8_model_ && ptr_int8[4] * out_scale < threshold) ||
          (!is_int8_model_ && ptr_float[4] < threshold)) {
        ptr_int8 += num_elemets;
        ptr_float += num_elemets;
        continue;
      }

      int label = -1;
      float score = 0;
      float obj_score;
      std::stringstream ss;
      ss << "batch:" << b << ",ind:" << i << ",";
      for (int k = 5; k < num_elemets; k++) {
        if (is_int8_model_) {
          ss << ptr_int8[k] << ",";
          if (ptr_int8[k] > score) {
            score = ptr_int8[k];
            label = k;
          }
          score = score * out_scale;
          obj_score = ptr_int8[4] * out_scale;
        } else {
          ss << ptr_float[k] << ",";
          if (ptr_float[k] > score) {
            score = ptr_float[k];
            label = k;
          }
          obj_score = ptr_float[4];
        }
      }
      if (score * obj_score > threshold && label != -1) {
        ObjectBox obj;
        if (is_int8_model_) {
          obj.x1 = ptr_int8[0] * out_scale;
          obj.y1 = ptr_int8[1] * out_scale;
          obj.x2 = ptr_int8[2] * out_scale;
          obj.y2 = ptr_int8[3] * out_scale;
        } else {
          obj.x1 = ptr_float[0];
          obj.y1 = ptr_float[1];
          obj.x2 = ptr_float[2];
          obj.y2 = ptr_float[3];
        }
        obj.x1 -= obj.x2 * 0.5;
        obj.y1 -= obj.y2 * 0.5;
        obj.x2 = obj.x1 + obj.x2;
        obj.y2 = obj.y1 + obj.y2;

        obj.label = label - 5;
        obj.score = score * obj_score;

        ss << ",bbox:" << obj.x1 << "," << obj.y1 << "," << obj.x2 << ","
           << obj.y2 << ",label" << obj.label;
        LOG(INFO) << ss.str();

        total_box.push_back(obj);
      }
      ptr_int8 += num_elemets;
      ptr_float += num_elemets;
    }
    LOG(INFO) << "to do nms:";
    if (label_nms_threshold_.size() > 0) {
      nms_obj_with_type(total_box, label_nms_threshold_);
    } else {
      nms_obj(total_box, nms_thresh_);
    }

    // nms_obj(total_box, 0.8,true);
    for (int j = 0; j < total_box.size(); ++j) {
      LOG(INFO) << "bbox:" << total_box[j].x1 << "," << total_box[j].y1 << ","
                << total_box[j].x2 << "," << total_box[j].y2;
      last_result.push_back(total_box[j]);
    }
  }
  return BM_COMMON_SUCCESS;
}

bmStatus_t YOLOV5::postprocess_3out(
    const std::vector<cv::Size> &frame_sizes, const float threshold,
    const std::vector<std::vector<float>> &frame_scale_params,
    std::vector<std::vector<ObjectBox>> &results) {
  int batch_n = get_input_n();
  double tick0 = cv::getTickCount();
  net_->update_output_tensors();
  double tick1 = cv::getTickCount();
  int num_anchors = 0;
  int num_batch = 0;
  std::map<std::string, nncompact::Tensor *> out_tensors;
  std::map<std::string, float> out_scales;
  std::map<std::string, int> out_elems;
  std::map<std::string, const float *> out_floats;
  std::map<std::string, const int8_t *> out_int8s;
  std::map<std::string, bool> out_isint8;
  // parse output layer with shape
  for (auto &iter : output_scales_) {
    std::string str_out = iter.first;
    nncompact::Tensor *out = net_->get_output_tensor(str_out).get();
    std::vector<int> shape = out->get_shape();
    if (num_anchors == 0) {
      num_anchors = shape[2];
      num_batch = shape[0];
    } else if (num_anchors != shape[2]) {
      LOG(FATAL) << "anchorsize not ok,prev:" << num_anchors
                 << ",current:" << shape[2];
    }
    int num_elem = shape[3];
    std::string strout;
    if (num_elem == 1) {
      strout = "obj";
    } else if (num_elem == 4) {
      strout = "box";
    } else if (out_tensors.count("cls") == 0) {
      strout = "cls";
    } else {
      LOG(FATAL) << "shape not expected";
    }
    out_elems[strout] = num_elem;
    out_tensors[strout] = out;
    out_scales[strout] = iter.second;
    out_floats[strout] = out->get_data();
    out_int8s[strout] = (const int8_t *)out_floats[strout];
    out_isint8[strout] = iter.second != 1;

    LOG(INFO) << "out:" << strout << ",detshape,batch:" << batch_n
              << "output:" << shape[0] << "," << shape[1] << "," << shape[2]
              << "," << shape[3] << ",out_scale:" << out_scales[strout]
              << ",isint8:" << out_isint8[strout] << ",num_elem:" << num_elem;
  }

  const int8_t *int8_ptrs[3] = {out_int8s["obj"], out_int8s["cls"],
                                out_int8s["box"]};
  const float *float_ptrs[3] = {out_floats["obj"], out_floats["cls"],
                                out_floats["box"]};
  int num_elemes[3] = {out_elems["obj"], out_elems["cls"], out_elems["box"]};
  bool is_int8_list[3] = {out_isint8["obj"], out_isint8["cls"],
                          out_isint8["box"]};
  float out_scales_list[3] = {out_scales["obj"], out_scales["cls"],
                              out_scales["box"]};
  // element data:x,y,w,h,obj,cls1,cls2,..,clsn
  double tick2 = cv::getTickCount();
  for (int b = 0; b < batch_n; b++) {
    results.emplace_back();
    auto &last_result = results.back();
    std::vector<ObjectBox> total_box;
    float width = frame_sizes[b].width;
    float height = frame_sizes[b].height;
    const std::vector<float> &rescale_param = frame_scale_params[b];
    float im_scale_w = rescale_param[0];
    float im_scale_h = rescale_param[1];
    float pad_x = rescale_param[2];
    float pad_y = rescale_param[3];
    LOG(INFO) << "padx:" << pad_x << ",pady:" << pad_y << ",imsw:" << im_scale_w
              << ",imsh:" << im_scale_h;
    bool is_obj_int8 = is_int8_list[0];
    float obj_thresh = threshold / out_scales_list[0];
    float cls_thresh = threshold / out_scales_list[1];
    // TODO:int8 output should multiply outscale
    for (int i = 0; i < num_anchors; ++i) {
      float obj_score = 0;
      if ((is_obj_int8 && int8_ptrs[0][0] < obj_thresh) ||
          (!is_obj_int8 && float_ptrs[0][0] < obj_thresh)) {
        for (int k = 0; k < out_tensors.size(); k++) {
          int8_ptrs[k] += num_elemes[k];
          float_ptrs[k] += num_elemes[k];
        }
        continue;
      }
      if (is_int8_list[0]) {
        obj_score = int8_ptrs[0][0] * out_scales_list[0];
      } else {
        obj_score = float_ptrs[0][0];
      }

      int label = -1;
      float score = 0;

      std::stringstream ss;
      // ss << "batch:" << b << ",ind:" << i << "," <<",objscore:"<<obj_score;;
      for (int k = 0; k < num_elemes[1]; k++) {
        if (is_int8_list[1]) {
          // ss << out_int8s["cls"][k] << ",";
          if (int8_ptrs[1][k] < cls_thresh) continue;
          if (int8_ptrs[1][k] > score) {
            score = int8_ptrs[1][k];
            label = k;
          }
        } else {
          // ss << out_floats["cls"][k] << ",";
          if (float_ptrs[1][k] > score) {
            score = float_ptrs[1][k];
            label = k;
          }
        }
      }
      score = score * out_scales_list[1];
      if (score * obj_score > threshold) {
        ObjectBox obj;
        if (is_int8_list[2]) {
          obj.x1 = int8_ptrs[2][0] * out_scales_list[2];
          obj.y1 = int8_ptrs[2][1] * out_scales_list[2];
          obj.x2 = int8_ptrs[2][2] * out_scales_list[2];
          obj.y2 = int8_ptrs[2][3] * out_scales_list[2];
        } else {
          obj.x1 = float_ptrs[2][0];
          obj.y1 = float_ptrs[2][1];
          obj.x2 = float_ptrs[2][2];
          obj.y2 = float_ptrs[2][3];
        }
        obj.x1 -= obj.x2 * 0.5;
        obj.y1 -= obj.y2 * 0.5;
        obj.x2 = obj.x1 + obj.x2;
        obj.y2 = obj.y1 + obj.y2;

        obj.x1 = (obj.x1 - pad_x) * im_scale_w;
        obj.y1 = (obj.y1 - pad_y) * im_scale_h;
        obj.x2 = (obj.x2 - pad_x) * im_scale_w;
        obj.y2 = (obj.y2 - pad_y) * im_scale_h;

        obj.label = label;
        obj.score = score * obj_score;

        // ss << ",bbox:" << obj.x1 << "," << obj.y1 << "," << obj.x2 << ","
        //    << obj.y2 << ",label" << obj.label;
        //  LOG(INFO) << ss.str();

        total_box.push_back(obj);
      }
      for (int k = 0; k < out_tensors.size(); k++) {
        int8_ptrs[k] += num_elemes[k];
        float_ptrs[k] += num_elemes[k];
      }
    }

    if (label_nms_threshold_.size() > 0) {
      nms_obj_with_type(total_box, label_nms_threshold_);
    } else {
      nms_obj(total_box, nms_thresh_);
    }

    // nms_obj(total_box, 0.8,true);
    // for (int j = 0; j < total_box.size(); ++j) {
    //    LOG(INFO) << "bbox:" << total_box[j].x1 << "," << total_box[j].y1 <<
    //    ","
    //              << total_box[j].x2 << "," <<
    //              total_box[j].y2<<",label:"<<total_box[j].label;
    // //   last_result.push_back(total_box[j]);
    // }
    LOG(INFO) << "obj size:" << total_box.size();
    last_result.insert(last_result.end(), total_box.begin(), total_box.end());
  }
  double tick3 = cv::getTickCount();
  double freq = cv::getTickFrequency() / 1000;
  LOG(INFO) << "memts:" << (tick1 - tick0) / freq
            << ",decodets:" << (tick3 - tick2) / freq;
  return BM_COMMON_SUCCESS;
}
