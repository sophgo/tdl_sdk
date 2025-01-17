#include "face/face_spoof.hpp"
#include <log/Logger.hpp>
using nncompact::Net;
using nncompact::Tensor;

bmStatus_t FaceSpoof::setup() {
  setup_net(net_param_);
  tmp_resized_img_ = cv::Mat::zeros(input_geometry_, CV_8UC3);
  LOG(INFO) << "input geometry size:" << input_geometry_;
  return BM_COMMON_SUCCESS;
}
bmStatus_t FaceSpoof::classify(const std::vector<BGR_IR_MAT> &bgr_ir_images,
                               std::vector<bool> &is_real_faces) {
  auto img_iter = bgr_ir_images.cbegin();
  auto left_size = bgr_ir_images.size();
  is_real_faces.clear();
#ifdef TIME_PRINT
  timer_.store_timestamp("total classification");
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

    BM_CHECK_STATUS(postprocess(is_real_faces));
    img_iter += batch_size;
    left_size -= batch_size;

#ifdef TIME_PRINT
    timer_.store_timestamp("postprocess");
#endif
  }
#ifdef TIME_PRINT
  timer_.store_timestamp("total classification");
  timer_.show();
  timer_.clear();
#endif
  return BM_COMMON_SUCCESS;
}

bmStatus_t
FaceSpoof::preprocess(std::vector<BGR_IR_MAT>::const_iterator &img_iter,
                      int batch_size) {

  set_input_n(batch_size);

  if(scales_.size() == 0){
    scales_.push_back(1.0/255);
  }
  for (int i = 0; i < batch_size; i++) {
    std::vector<cv::Mat> input_channels;
    wrap_input_layer(i, input_channels);
    if (input_channels.size() != 6) {
      LOG(FATAL) << "bgr_ir pairs should have 6 channels,but got:"
                 << input_channels.size();
    }
    const BGR_IR_MAT &bgr_ir = *(img_iter + i);
    if (bgr_ir.size() != 2) {
      LOG(FATAL) << "bgr_ir pairs should have two images,but got:"
                 << bgr_ir.size();
    }

    for (int j = 0; j < bgr_ir.size(); j++) {
      cv::Mat resized = tmp_resized_img_;
      auto &img = bgr_ir[j];
      if (img.cols == input_geometry_.width &&
          img.rows == input_geometry_.height) {
        resized = img;
        LOG(INFO) << "no resize,geometry:" << input_geometry_;
      } else
        cv::resize(img, resized, input_geometry_, 0, 0, cv::INTER_NEAREST);
      std::vector<cv::Mat> channels(input_channels.begin() + j * 3,
                                    input_channels.begin() + (j + 1) * 3);
      bgr_split_scale1(resized, temp_bgr_, channels, means_,scales_,use_rgb_);
    }
  }
  std::shared_ptr<nncompact::Tensor> input_tensor =
      net_->get_input_tensor(input_layer_);
  // input_tensor->dump_to_file("bgr_ir.bin");
  net_->update_input_tensors();

  return BM_COMMON_SUCCESS;
}

bmStatus_t FaceSpoof::postprocess(std::vector<bool> &real_faces) {
  int batch_n = get_input_n();
  net_->update_output_tensors();
  LOG(INFO) << "outlayer:" << output_layer_ << ",batch:" << batch_n;
  nncompact::Tensor *output_tensor =
      net_->get_output_tensor(output_layer_).get();

  std::vector<int> shape = output_tensor->get_shape();
  LOG(INFO) << ",shapesize:" << shape.size();
  std::stringstream ss;
  ss << "batch:" << batch_n << "output:" << shape[0] << "," << shape[1] << ","
     << shape[2] << "," << shape[3] << ",";
  LOG(INFO) << ss.str();
  const float *p_res = output_tensor->get_data();
  for (int i = 0; i < batch_n; i++) {
    LOG(INFO) << p_res[i];
    bool is_real = p_res[i] > thresh_;
    real_faces.push_back(is_real);
  }

  return BM_COMMON_SUCCESS;
}
