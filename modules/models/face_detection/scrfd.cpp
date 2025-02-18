#include "face_detection/scrfd.hpp"

#include "core/cvi_tdl_types_mem_internal.h"
#include "core/face/cvtdl_face_types.h"
#include "cvi_tdl_log.hpp"
#include "utils/detection_helper.hpp"

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
    std::vector<void *> &out_datas) {
  std::string input_tensor_name = net_->getInputNames()[0];
  TensorInfo input_tensor = net_->getTensorInfo(input_tensor_name);
  uint32_t input_width = input_tensor.shape[3];
  uint32_t input_height = input_tensor.shape[2];

  LOGI("outputParse,batch size:%d,input shape:%d,%d,%d,%d", images.size(),
       input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2],
       input_tensor.shape[3]);

  for (uint32_t b = 0; b < (uint32_t)input_tensor.shape[0]; b++) {
    uint32_t image_width = images[b]->getWidth();
    uint32_t image_height = images[b]->getHeight();
    float image_width_f = float(image_width);
    float image_height_f = float(image_height);
    std::vector<cvtdl_face_info_t> vec_bbox;
    std::vector<cvtdl_face_info_t> vec_bbox_nms;
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

          cvtdl_face_info_t box;
          memset(&box, 0, sizeof(box));
          box.pts.size = 5;
          box.pts.x = (float *)malloc(sizeof(float) * box.pts.size);
          box.pts.y = (float *)malloc(sizeof(float) * box.pts.size);
          box.bbox.score = conf;
          box.hardhat_score = 0;

          // cv::Vec4f regress;
          // bbox_blob:b x (num_anchors*num_elem) x h x w
          box.bbox.x1 = grid_cx - bbox_blob[j + count * (0 + num * 4)] * stride;
          box.bbox.y1 = grid_cy - bbox_blob[j + count * (1 + num * 4)] * stride;
          box.bbox.x2 = grid_cx + bbox_blob[j + count * (2 + num * 4)] * stride;
          box.bbox.y2 = grid_cy + bbox_blob[j + count * (3 + num * 4)] * stride;

          box.bbox.x1 =
              std::clamp((box.bbox.x1 - pad_x) / scalex, 0.0f, image_width_f);
          box.bbox.y1 =
              std::clamp((box.bbox.y1 - pad_y) / scaley, 0.0f, image_height_f);
          box.bbox.x2 =
              std::clamp((box.bbox.x2 - pad_x) / scalex, 0.0f, image_width_f);
          box.bbox.y2 =
              std::clamp((box.bbox.y2 - pad_y) / scaley, 0.0f, image_height_f);

          if (box.bbox.x1 >= box.bbox.x2 || box.bbox.y1 >= box.bbox.y2) {
            LOGI(
                "bbox "
                "invalid,x1:%f,y1:%f,x2:%f,y2:%f,stride:%d,grid_cx:%f,grid_cy:%"
                "f\n",
                box.bbox.x1, box.bbox.y1, box.bbox.x2, box.bbox.y2, stride,
                grid_cx, grid_cy);
            continue;
          }

          for (size_t k = 0; k < box.pts.size; k++) {
            box.pts.x[k] =
                landmark_blob[j + count * (num * 10 + k * 2)] * stride +
                grid_cx;
            box.pts.y[k] =
                landmark_blob[j + count * (num * 10 + k * 2 + 1)] * stride +
                grid_cy;
            box.pts.x[k] = std::clamp((box.pts.x[k] - pad_x) * scalex, 0.0f,
                                      image_width_f);
            box.pts.y[k] = std::clamp((box.pts.y[k] - pad_y) * scaley, 0.0f,
                                      image_height_f);
          }

          vec_bbox.push_back(box);
        }
      }
    }
    // DO nms on output result

    DetectionHelper::nmsFaces(vec_bbox, 0.4);
    // Init face meta
    cvtdl_face_t *facemeta = new cvtdl_face_t();
    facemeta->width = image_width;
    facemeta->height = image_height;
    LOGI("vec_bbox_nms size:%d", vec_bbox.size());
    if (vec_bbox.size() == 0) {
      facemeta->size = vec_bbox_nms.size();
      facemeta->info = NULL;
      out_datas.push_back((void *)facemeta);
      continue;
    }
    CVI_TDL_MemAllocInit(vec_bbox.size(), 5, facemeta);

    LOGI("rescale_params:%f,%f,%f,%f", scalex, scaley, pad_x, pad_y);
    std::stringstream ss;
    for (uint32_t i = 0; i < facemeta->size; ++i) {
      cvtdl_face_info_t info = vec_bbox[i];
      facemeta->info[i].bbox.x1 = info.bbox.x1;
      facemeta->info[i].bbox.x2 = info.bbox.x2;
      facemeta->info[i].bbox.y1 = info.bbox.y1;
      facemeta->info[i].bbox.y2 = info.bbox.y2;
      facemeta->info[i].bbox.score = info.bbox.score;
      facemeta->info[i].hardhat_score = info.hardhat_score;
      for (int j = 0; j < 5; ++j) {
        facemeta->info[i].pts.x[j] = info.pts.x[j];
        facemeta->info[i].pts.y[j] = info.pts.y[j];
      }

      ss << "bbox:" << facemeta->info[i].bbox.x1 << ","
         << facemeta->info[i].bbox.y1 << "," << facemeta->info[i].bbox.x2 << ","
         << facemeta->info[i].bbox.y2 << "," << facemeta->info[i].bbox.score
         << ",imgwidth:" << image_width << ",imgheight:" << image_height
         << std::endl;
    }
    LOGI("facemeta:%s", ss.str().c_str());

    out_datas.push_back((void *)facemeta);
  }
  return 0;
}
