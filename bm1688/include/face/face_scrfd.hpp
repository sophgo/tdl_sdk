#ifndef FACE_SCRFD_HPP_
#define FACE_SCRFD_HPP_
#include "common/obj_det_utils.hpp"
#include "face/face_common.hpp"
#include "framework/base_model.hpp"
#include "netcompact/tensor.hpp"

class FaceSCRFD : public BaseModel {
 public:
  explicit FaceSCRFD(const stNetParam &param);

  ~FaceSCRFD() {}

  bmStatus_t setup();
  void init_anchor();
  bmStatus_t detect(const std::vector<cv::Mat> &images, const float threshold,
                    std::vector<std::vector<FaceRect>> &results);

 private:
  bmStatus_t postprocess(
      const std::vector<cv::Size> &frame_sizes, const float threshold,
      const std::vector<std::vector<float>> &frame_scale_params,
      std::vector<std::vector<FaceRect>> &results);

  // scale, scale, pad_x, pad_y
  bmStatus_t nms(std::vector<FaceRect> &nmsProposals, float nms_threshold_);
  std::shared_ptr<nncompact::Tensor> resized_img_buffer_;

  // ResizeParameter resize_param_;
  // FaceCSSDParameter param_;
  std::vector<cv::Mat> tmp_bgr_planar_;
  // std::vector<std::string> output_layers_=
  // {"26","27","28","29","30","31","32","33","34"}; std::vector<std::string>
  // output_layers_=
  // {"448_Transpose_top.Sigmoid_top.Reshape_f32","489_Transpose_top.Sigmoid_top.Reshape_f32","530_Transpose_top.Sigmoid_top.Reshape_f32",
  //                                           "bbox_8_Reshape_f32","bbox_16_Reshape_f32","bbox_32_Reshape_f32","kps_8_Reshape_f32","kps_16_Reshape_f32","kps_32_Reshape_f32"};
  // std::vector<std::string> output_layers_=
  // {"448_Transpose_top.Sigmoid_top.Reshape","489_Transpose_top.Sigmoid_top.Reshape","530_Transpose_top.Sigmoid_top.Reshape",
  //                                           "bbox_8_Reshape","bbox_16_Reshape","bbox_32_Reshape","kps_8_Reshape","kps_16_Reshape","kps_32_Reshape"};
  std::vector<std::string> output_layers_ = {
      "score_8_Sigmoid_f32", "score_16_Sigmoid_f32", "score_32_Sigmoid_f32",
      "bbox_8_Conv_f32",     "bbox_16_Conv_f32",     "bbox_32_Conv_f32",
      "kps_8_Conv_f32",      "kps_16_Conv_f32",      "kps_32_Conv_f32"};
  std::vector<std::vector<float>> base_anchors;
  std::vector<std::vector<float>> grid_anchors;
  // Image preprocessing size
  int resize_width;
  int resize_height;

  std::vector<int> m_feat_stride_fpn;
  // std::map<std::string, std::vector<anchor_box>> m_anchors;
  // std::map<std::string, int> m_num_anchors;
  std::map<int, std::vector<std::vector<float>>> fpn_anchors_;
  std::map<int, std::map<std::string, std::string>>
      fpn_out_nodes_;  //{stride:{"box":"xxxx","score":"xxx","landmark":"xxxx"}}
  std::map<int, int> fpn_grid_anchor_num_;

  std::map<std::string, int> name_id_map_;
};

#endif
