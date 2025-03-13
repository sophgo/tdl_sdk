#include "face_detection/scrfd.hpp"

#include "utils/detection_helper.hpp"
#include "utils/tdl_log.hpp"

SCRFD::SCRFD() {
  std::vector<float> means = {127.5, 127.5, 127.5};
  std::vector<float> scales = {1.0 / 128, 1.0 / 128, 1.0 / 128};

  for (int i = 0; i < 3; i++) {
    net_param_.pre_params.scale[i] = scales[i];
    net_param_.pre_params.mean[i] = means[i] * scales[i];
  }
  net_param_.pre_params.dstImageFormat = ImageFormat::RGB_PLANAR;
  net_param_.pre_params.keepAspectRatio = true;
}
SCRFD::~SCRFD() {}
int32_t SCRFD::onModelOpened() {
  struct anchor_cfg {
    std::vector<int> SCALES;
    int BASE_SIZE;
    std::vector<float> RATIOS;
    int ALLOWED_BORDER;
    int STRIDE;
  };
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
  LOGI("start to parse node");

  std::string input_tensor_name = net_->getInputNames()[0];
  TensorInfo input_tensor = net_->getTensorInfo(input_tensor_name);

  for (size_t i = 0; i < cfg.size(); i++) {
    std::vector<std::vector<float>> base_anchors =
        DetectionHelper::generateMmdetBaseAnchors(cfg[i].BASE_SIZE, 0,
                                                  cfg[i].RATIOS, cfg[i].SCALES);
    int stride = cfg[i].STRIDE;
    int input_w = input_tensor.shape[3];
    int input_h = input_tensor.shape[2];
    int feat_w = ceil(input_w / float(stride));
    int feat_h = ceil(input_h / float(stride));
    fpn_anchors_[stride] = DetectionHelper::generateMmdetGridAnchors(
        feat_w, feat_h, stride, base_anchors);

    int num_feat_branch = 0;
    int num_anchors = int(cfg[i].SCALES.size() * cfg[i].RATIOS.size());
    fpn_grid_anchor_num_[stride] = num_anchors;
    std::vector<std::string> output_tensor_names = net_->getOutputNames();
    for (size_t j = 0; j < output_tensor_names.size(); j++) {
      TensorInfo oj = net_->getTensorInfo(output_tensor_names[j]);
      LOGI("stride:%d,w:%d,feath:%d,node:%s,sw:%d,sh:%d,c:%d", stride, feat_w,
           feat_h, output_tensor_names[j].c_str(), oj.shape[3], oj.shape[2],
           oj.shape[1]);
      // std::cout << "stride:" << stride << ",w:" << feat_w << ",feath:" <<
      // feat_h
      //           << ",node:" << getOutputTensorInfo(j).tensor_name << ",sw:"
      //           << oj.dim[3]
      //           << ",sh:" << oj.dim[2] << ",c:" << oj.dim[1] << std::endl;
      if (oj.shape[2] == feat_h && oj.shape[3] == feat_w) {
        LOGI("fpnnode,stride:%d,w:%d,feath:%d", stride, feat_w, feat_h);
        if (oj.shape[1] == num_anchors * 1) {
          fpn_out_nodes_[stride]["score"] = output_tensor_names[j];
          num_feat_branch++;
        } else if (oj.shape[1] == num_anchors * 4) {
          fpn_out_nodes_[stride]["bbox"] = output_tensor_names[j];
          num_feat_branch++;
        } else if (oj.shape[1] == num_anchors * 10) {
          fpn_out_nodes_[stride]["landmark"] = output_tensor_names[j];
          num_feat_branch++;
        }
      }
    }
    // std::cout << "numfeat:" << num_feat_branch << std::endl;
    for (auto &kv : fpn_out_nodes_[stride]) {
      LOGI("%s:%s", kv.first.c_str(), kv.second.c_str());
    }
    if (num_feat_branch != int(cfg.size())) {
      LOGE("output nodenum error,got:%d,expected:%d at branch:%d",
           num_feat_branch, int(cfg.size()), int(i));
    }
  }
  return 0;
}

int32_t SCRFD::outputParse(
    const std::vector<std::shared_ptr<BaseImage>> &images,
    std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) {
  std::string input_tensor_name = net_->getInputNames()[0];
  TensorInfo input_tensor = net_->getTensorInfo(input_tensor_name);
  uint32_t input_width = input_tensor.shape[3];
  uint32_t input_height = input_tensor.shape[2];

  LOGI("outputParse,batch size:%d,input shape:%d,%d,%d,%d", images.size(),
       input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2],
       input_tensor.shape[3]);

  const int FACE_LANDMARKS_NUM = 5;

  for (uint32_t b = 0; b < (uint32_t)input_tensor.shape[0]; b++) {
    uint32_t image_width = images[b]->getWidth();
    uint32_t image_height = images[b]->getHeight();
    float image_width_f = float(image_width);
    float image_height_f = float(image_height);
    std::vector<ObjectBoxLandmarkInfo> vec_bbox;
    std::vector<ObjectBoxLandmarkInfo> vec_bbox_nms;
    std::vector<float> &rescale_params = batch_rescale_params_[b];
    float scalex = rescale_params[0];
    float scaley = rescale_params[1];
    float pad_x = rescale_params[2];
    float pad_y = rescale_params[3];

    LOGI("stride size:%d", m_feat_stride_fpn.size());
    for (size_t i = 0; i < m_feat_stride_fpn.size(); i++) {
      int stride = m_feat_stride_fpn[i];
      std::string score_str = fpn_out_nodes_[stride]["score"];
      TensorInfo score_tensor = net_->getTensorInfo(score_str);
      // std::cout << "stride:" << m_feat_stride_fpn[i] << "\n";
      // print_dim(score_shape, "score");
      size_t score_size =
          score_tensor.shape[1] * score_tensor.shape[2] * score_tensor.shape[3];
      float *score_blob = (float *)score_tensor.sys_mem + (b * score_size);
      LOGI(
          "scoretensor:%s,shape:%d,%d,%d,%d,dev_addr:%d,dtype:%d,val0:%f,"
          "threshold:%f",
          score_str.c_str(), score_tensor.shape[0], score_tensor.shape[1],
          score_tensor.shape[2], score_tensor.shape[3], score_tensor.phy_addr,
          score_tensor.data_type, score_blob[0], model_threshold_);
      std::string bbox_str = fpn_out_nodes_[stride]["bbox"];
      TensorInfo bbox_tensor = net_->getTensorInfo(bbox_str);
      // print_dim(blob_shape, "bbox");
      size_t blob_size =
          bbox_tensor.shape[1] * bbox_tensor.shape[2] * bbox_tensor.shape[3];
      float *bbox_blob = (float *)bbox_tensor.sys_mem + (b * blob_size);

      std::string landmark_str = fpn_out_nodes_[stride]["landmark"];
      TensorInfo landmark_tensor = net_->getTensorInfo(landmark_str);
      // print_dim(blob_shape, "bbox");
      size_t landmark_size = landmark_tensor.shape[1] *
                             landmark_tensor.shape[2] *
                             landmark_tensor.shape[3];
      float *landmark_blob =
          (float *)landmark_tensor.sys_mem + (b * landmark_size);
      int width = bbox_tensor.shape[3];
      int height = bbox_tensor.shape[2];
      size_t count = width * height;
      size_t num_anchor = fpn_grid_anchor_num_[stride];
      // std::cout << "numanchor:" << num_anchor << ",count:" << count << "\n";
      std::vector<std::vector<float>> &fpn_grids = fpn_anchors_[stride];
      for (size_t num = 0; num < num_anchor; num++) {  // anchor index
        for (size_t j = 0; j < count; j++) {           // j:grid index
          float conf = score_blob[j + count * num];
          if (conf <= model_threshold_) {
            continue;
          }

          // TODO:could be optimized
          std::vector<float> &grid = fpn_grids[j + count * num];
          float grid_cx = (grid[0] + grid[2]) / 2;
          float grid_cy = (grid[1] + grid[3]) / 2;

          float box_score = conf;

          float box_x1 =
              grid_cx - bbox_blob[j + count * (0 + num * 4)] * stride;
          float box_y1 =
              grid_cy - bbox_blob[j + count * (1 + num * 4)] * stride;
          float box_x2 =
              grid_cx + bbox_blob[j + count * (2 + num * 4)] * stride;
          float box_y2 =
              grid_cy + bbox_blob[j + count * (3 + num * 4)] * stride;

          box_x1 = std::clamp((box_x1 - pad_x) / scalex, 0.0f, image_width_f);
          box_y1 = std::clamp((box_y1 - pad_y) / scaley, 0.0f, image_height_f);
          box_x2 = std::clamp((box_x2 - pad_x) / scalex, 0.0f, image_width_f);
          box_y2 = std::clamp((box_y2 - pad_y) / scaley, 0.0f, image_height_f);

          if (box_x1 >= box_x2 || box_y1 >= box_y2) {
            LOGI(
                "bbox "
                "invalid,x1:%f,y1:%f,x2:%f,y2:%f,stride:%d,grid_cx:%f,grid_cy:%"
                "f\n",
                box_x1, box_y1, box_x2, box_y2, stride, grid_cx, grid_cy);
            continue;
          }

          std::vector<float> landmarks_x;
          std::vector<float> landmarks_y;
          std::vector<float> landmarks_score;
          for (size_t k = 0; k < FACE_LANDMARKS_NUM; k++) {
            float landmark_x =
                landmark_blob[j + count * (num * 10 + k * 2)] * stride +
                grid_cx;
            float landmark_y =
                landmark_blob[j + count * (num * 10 + k * 2 + 1)] * stride +
                grid_cy;
            landmark_x =
                std::clamp((landmark_x - pad_x) / scalex, 0.0f, image_width_f);
            landmark_y =
                std::clamp((landmark_y - pad_y) / scaley, 0.0f, image_height_f);
            landmarks_x.push_back(landmark_x);
            landmarks_y.push_back(landmark_y);
            landmarks_score.push_back(0);
          }
          ObjectBoxLandmarkInfo box;
          box.class_id = 0;
          box.object_type = OBJECT_TYPE_FACE;
          box.score = box_score;
          box.x1 = box_x1;
          box.y1 = box_y1;
          box.x2 = box_x2;
          box.y2 = box_y2;
          box.landmarks_x = landmarks_x;
          box.landmarks_y = landmarks_y;
          box.landmarks_score = landmarks_score;
          vec_bbox.push_back(box);
        }
      }
    }
    // DO nms on output result

    DetectionHelper::nmsObjects(vec_bbox, iou_threshold_);
    // Init face meta

    std::shared_ptr<ModelBoxLandmarkInfo> facemeta =
        std::make_shared<ModelBoxLandmarkInfo>();
    facemeta->image_width = image_width;
    facemeta->image_height = image_height;
    facemeta->box_landmarks = vec_bbox;

    out_datas.push_back(facemeta);
  }
  LOGI("srcfd outputParse done");
  return 0;
}
