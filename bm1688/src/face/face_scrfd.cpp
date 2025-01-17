#include "face/face_scrfd.hpp"

#include <log/Logger.hpp>
FaceSCRFD::FaceSCRFD(const stNetParam &param) { net_param_ = param; }

bmStatus_t FaceSCRFD::setup() {
  setup_net(net_param_);
  resize_width = 768;
  resize_height = 432;
  // input_layer_ = param_.input_layer();

  // NetParameter net_param = param_.net_param();
  // BM168XNetParameter *net_168x_param = net_param.mutable_bm168xnet_param();
  // net_168x_param->set_input_mem_type(BM168XNetParameter_MEM_TYPE_DEVICE);
  // setup_net(net_param);
  // set_input_n(1);
  // resize_width = param_.width();
  // resize_height = param_.height();

  return BM_COMMON_SUCCESS;
}
inline bool compareBBox(const FaceRect &a, const FaceRect &b) {
  return a.score > b.score;
}
void FaceSCRFD::init_anchor() {
  std::vector<anchor_cfg> cfg;
  anchor_cfg tmp;

  m_feat_stride_fpn = {8, 16, 32};

  tmp.SCALES = {1, 2};
  tmp.BASE_SIZE = 16;
  tmp.RATIOS = {1.0};
  tmp.ALLOWED_BORDER = 9999;
  tmp.STRIDE = 8;
  cfg.push_back(tmp);

  tmp.SCALES = {4, 8};
  tmp.BASE_SIZE = 16;
  tmp.RATIOS = {1.0};
  tmp.ALLOWED_BORDER = 9999;
  tmp.STRIDE = 16;
  cfg.push_back(tmp);

  tmp.SCALES = {16, 32};
  tmp.BASE_SIZE = 16;
  tmp.RATIOS = {1.0};
  tmp.ALLOWED_BORDER = 9999;
  tmp.STRIDE = 32;
  cfg.push_back(tmp);
  // std::cout << "start to parse node\n";
  std::map<std::string, std::vector<anchor_box>> anchors_fpn_map;
  LOG(INFO) << "inputname:" << net_->input_names_[0];
  nncompact::Tensor *p_input =
      net_->get_input_tensor(net_->input_names_[0]).get();

  for (size_t i = 0; i < cfg.size(); i++) {
    std::vector<std::vector<float>> base_anchors = generate_mmdet_base_anchors(
        cfg[i].BASE_SIZE, 0, cfg[i].RATIOS, cfg[i].SCALES);
    int stride = cfg[i].STRIDE;
    int input_w = p_input->width();
    int input_h = p_input->height();
    int feat_w = int(input_w / float(stride) + 0.5);
    int feat_h = int(input_h / float(stride) + 0.5);
    fpn_anchors_[stride] =
        generate_mmdet_grid_anchors(feat_w, feat_h, stride, base_anchors);

    int num_feat_branch = 0;
    int num_anchors = int(cfg[i].SCALES.size() * cfg[i].RATIOS.size());
    fpn_grid_anchor_num_[stride] = num_anchors;

    for (size_t j = 0; j < net_->output_names_.size(); j++) {
      nncompact::Tensor *p_outj =
          net_->get_output_tensor(net_->output_names_[j]).get();
      LOG(INFO) << "out:" << net_->output_names_[j] << ",featw:" << feat_w
                << ",outh:" << p_outj->height();
      if (p_outj->height() == feat_h && p_outj->width() == feat_w) {
        // std::cout << "fpnnode,stride:" << stride << ",w:" << feat_w <<
        // ",feath:" << feat_h
        //           << std::endl;
        if (p_outj->channels() == num_anchors * 1) {
          fpn_out_nodes_[stride]["score"] = net_->output_names_[j];
          num_feat_branch++;
        } else if (p_outj->channels() == num_anchors * 4) {
          fpn_out_nodes_[stride]["bbox"] = net_->output_names_[j];
          num_feat_branch++;
        }
      }
    }
    // std::cout << "numfeat:" << num_feat_branch << std::endl;
    for (auto &kv : fpn_out_nodes_[stride]) {
      std::cout << kv.first << ":" << kv.second << std::endl;
    }
    if (num_feat_branch != 2) {
      LOG(FATAL) << "output nodenum error,got:" << num_feat_branch
                 << ",expected:" << cfg.size() << ",fpn:" << i;
    }
  }
  for (size_t j = 0; j < net_->output_names_.size(); j++) {
    name_id_map_[net_->output_names_[j]] = j;
  }
}

bmStatus_t FaceSCRFD::postprocess(
    const std::vector<cv::Size> &frame_sizes, const float threshold,
    const std::vector<std::vector<float>> &frame_scale_params,
    std::vector<std::vector<FaceRect>> &results) {
  int anchor_num = 2;
  int batch_n = get_input_n();
  net_->update_output_tensors();
  nncompact::Tensor *p_input =
      net_->get_input_tensor(net_->input_names_[0]).get();
  // p_input->dump_to_file("input.bin");
  const bm_net_info_t *net_info = (const bm_net_info_t *)net_->get_net_info();

  for (size_t batch = 0; batch < batch_n; ++batch) {
    const std::vector<float> &rescale_param = frame_scale_params[batch];
    float im_scale_w = rescale_param[0];
    float im_scale_h = rescale_param[1];
    float pad_x = rescale_param[2];
    float pad_y = rescale_param[3];
    float width = frame_sizes[batch].width;
    float height = frame_sizes[batch].height;

    results.emplace_back();
    auto &last_result = results.back();
    std::vector<FaceRect> one_frame_result;
    // Verify that the type of output layer is supported, only BM_INT8 and
    // BM_FLOAT32 are supported currently
    //  for(int layer = 0; layer < 9; ++layer){
    //    if(net_info->output_dtypes[layer] != BM_INT8 &&
    //    net_info->output_dtypes[layer] != BM_FLOAT32) {
    //      LOG(FATAL) << "not implement yet";
    //    }
    //  }
    // every output group including {score，bbox，landmark}, 3 group totally
    for (int i = 0; i < 3; ++i) {
      float score_scale = net_info->output_scales[i];
      float bbox_scale = net_info->output_scales[i + 3];
      float pts_scale = net_info->output_scales[i + 6];

      LOG(INFO) << "score_scale:" << score_scale << ",bboxscale:" << bbox_scale
                << ",ptscale:" << pts_scale;
      // calculate base anchor and grid anchor
      std::vector<std::vector<int>> scales = {{1, 2}, {4, 8}, {16, 32}};
      int base_size = 16;
      std::vector<float> ratios = {1.0};
      std::vector<int> stride = {8, 16, 32};
      base_anchors =
          generate_mmdet_base_anchors(base_size, 0, ratios, scales[i]);
      int feat_w = int(resize_width / float(stride[i]) + 0.5);
      int feat_h = int(resize_height / float(stride[i]) + 0.5);
      // LOG(INFO)<<"i:" << i <<" base_anchors.size: "<<base_anchors.size() <<
      // ",feat_w:" << feat_w << ",feat_h:" << feat_h << ",stride[i]" <<
      // stride[i];
      grid_anchors =
          generate_mmdet_grid_anchors(feat_w, feat_h, stride[i], base_anchors);
      nncompact::Tensor *score_tensor =
          net_->get_output_tensor(output_layers_[i]).get();
      nncompact::Tensor *bbox_tensor =
          net_->get_output_tensor(output_layers_[i + 3]).get();
      nncompact::Tensor *landmark_tensor =
          net_->get_output_tensor(output_layers_[i + 6]).get();
      // LOG(INFO)<<"output_layers_[i]:"<<output_layers_[i];

      // std::string strt =
      // std::string("score_")+std::to_string(score_tensor->width())+std::string(".bin");
      // score_tensor->dump_to_file(strt);
      // int num_det = score_tensor->height();
      int count = (int)score_tensor->height() * score_tensor->width();
      float *p_score = score_tensor->get_data();
      float *p_bbox = bbox_tensor->get_data();
      int8_t *p_score_int8 = (int8_t *)p_score;
      int8_t *p_bbox_int8 = (int8_t *)p_bbox;
      // LOG(INFO)<<"outscore,c:"<<score_tensor->channels()<<",h:"<<score_tensor->height()<<",w:"<<score_tensor->width();

      float *p_score_fp_b = p_score + batch * score_tensor->batch_num_elems();
      float *p_bbox_fp_b = p_bbox + batch * bbox_tensor->batch_num_elems();
      float *landmark_out = landmark_tensor->get_data() +
                            batch * landmark_tensor->batch_num_elems();
      int8_t *p_score_i8_b =
          p_score_int8 + batch * score_tensor->batch_num_elems();
      int8_t *p_bbox_i8_b =
          p_bbox_int8 + batch * bbox_tensor->batch_num_elems();

      bool is_score_i8 = score_scale != 1.0;
      bool is_bbox_i8 = bbox_scale != 1.0;
      // LOG(INFO)<<"is_bbox_i8:"<<is_score_i8;
      // LOG(INFO)<<"count:"<<count<<";count:height:"<<score_tensor->height()<<";count:width:"<<score_tensor->width();

      for (size_t k = 0; k < score_tensor->height(); k++) {
        if (p_score_fp_b[k] > 0.2) std::cout << p_score_fp_b[k] << " ";
      }
      for (size_t num = 0; num < anchor_num; num++) {
        for (size_t j = 0; j < count; j++) {
          // decode score
          float score = 0;
          if (is_score_i8) {
            score = p_score_i8_b[j + count * num] * score_scale;
          } else {
            score = p_score_fp_b[j + count * num];
          }

          if (score > threshold) {
            LOG(INFO) << "score > threshold:" << score
                      << ";id:" << j + count * num
                      << ";threshold:" << threshold;
            FaceRect bbox;
            bbox.score = score;
            std::vector<float> &grid = grid_anchors[j + count * num];
            float grid_cx = (grid[0] + grid[2]) / 2;
            float grid_cy = (grid[1] + grid[3]) / 2;
            float ox1 = 0, ox2 = 0, oy1 = 0, oy2 = 0;
            if (is_bbox_i8) {
              ox1 = p_bbox_i8_b[j + count * (0 + num * 4)] * bbox_scale;
              oy1 = p_bbox_i8_b[j + count * (1 + num * 4)] * bbox_scale;
              ox2 = p_bbox_i8_b[j + count * (2 + num * 4)] * bbox_scale;
              oy2 = p_bbox_i8_b[j + count * (3 + num * 4)] * bbox_scale;
            } else {
              ox1 = p_bbox_fp_b[j + count * (0 + num * 4)];
              oy1 = p_bbox_fp_b[j + count * (1 + num * 4)];
              ox2 = p_bbox_fp_b[j + count * (2 + num * 4)];
              oy2 = p_bbox_fp_b[j + count * (3 + num * 4)];
            }

            bbox.x1 = grid_cx - ox1 * stride[i];
            bbox.y1 = grid_cy - oy1 * stride[i];
            bbox.x2 = grid_cx + ox2 * stride[i];
            bbox.y2 = grid_cy + oy2 * stride[i];

            bbox.x1 = clip_val((bbox.x1 - pad_x) * im_scale_w, 0.0, width);
            bbox.y1 = clip_val((bbox.y1 - pad_y) * im_scale_h, 0.0, height);
            bbox.x2 = clip_val((bbox.x2 - pad_x) * im_scale_w, 0.0, width);
            bbox.y2 = clip_val((bbox.y2 - pad_y) * im_scale_h, 0.0, height);
            // decode landmark
            //  for (size_t k = 0; k < 5; k++) {
            //    float tmp_facepts_x = landmark_out[j + count * (num * 10 + k *
            //    2)]*pts_scale* float(stride[i]) + grid_cx - pad_x; float
            //    tmp_facepts_y = landmark_out[j + count * (num * 10 + k * 2 +
            //    1)]*pts_scale* float(stride[i]) + grid_cy - pad_y;

            //   bbox.facepts.x.push_back(clip_val(tmp_facepts_x* im_scale_w, 0,
            //   width)); bbox.facepts.y.push_back(clip_val(tmp_facepts_y*
            //   im_scale_h, 0, height));
            // }
            one_frame_result.push_back(bbox);
          }
        }
      }
    }
    nms(one_frame_result, 0.4);
    // LOG(INFO)<<"after nms:";
    for (int j = 0; j < one_frame_result.size(); ++j) {
      // LOG(INFO) << "bbox:" << one_frame_result[j].x1 << "," <<
      // one_frame_result[j].y1 << ","
      //           << one_frame_result[j].x2 << "," <<
      //           one_frame_result[j].y2<<",score:"<<one_frame_result[j].score;
      last_result.push_back(one_frame_result[j]);
    }

    LOG(INFO) << "Number of detections: " << last_result.size();
  }
  return BM_COMMON_SUCCESS;
}

bmStatus_t FaceSCRFD::detect(const std::vector<cv::Mat> &images,
                             const float threshold,
                             std::vector<std::vector<FaceRect>> &results) {
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
    BM_CHECK_STATUS(
        postprocess(frame_sizes, threshold, batch_rescale_params_, results));
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

bmStatus_t FaceSCRFD::nms(std::vector<FaceRect> &input_boxes,
                          float NMS_THRESH) {
  std::sort(input_boxes.begin(), input_boxes.end(), compareBBox);
  LOG(INFO) << "input_boxes.size():" << input_boxes.size();
  std::vector<float> vArea(input_boxes.size());
  LOG(INFO) << "vArea.size():" << vArea.size();
  for (int i = 0; i < int(input_boxes.size()); ++i) {
    vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1) *
               (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
  }
  for (int i = 0; i < int(input_boxes.size()); ++i) {
    // LOG(INFO)<<"in for2, i:" << i;
    for (int j = i + 1; j < int(input_boxes.size());) {
      // LOG(INFO)<<"in for2, j:" << j;
      float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
      float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
      float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
      float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
      float w = std::max(float(0), xx2 - xx1 + 1);
      float h = std::max(float(0), yy2 - yy1 + 1);
      float inter = w * h;
      float ovr = inter / (vArea[i] + vArea[j] - inter);
      // LOG(INFO) << "(xx1,yy1,xx2,yy2):("<<xx1<<","<<yy1<<","<<xx2<<","<<yy2<<
      // ")"; LOG(INFO) << "w:" << w << ",h:" << h << ",inter:" << inter <<
      // ",ovr:" <<ovr;
      if (ovr >= NMS_THRESH) {
        // LOG(INFO)<<"j:" << j
        // <<",input_boxes.size():"<<input_boxes.size()<<",vArea.size():"<<vArea.size();
        input_boxes.erase(input_boxes.begin() + j);
        vArea.erase(vArea.begin() + j);
        // LOG(INFO)<<"j:" << j
        // <<",input_boxes.size():"<<input_boxes.size()<<",vArea.size():"<<vArea.size();
      } else {
        j++;
      }
    }
  }
  return BM_COMMON_SUCCESS;
}
