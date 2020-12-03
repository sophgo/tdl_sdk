#include "retina_face.hpp"
#include "retina_face_utils.hpp"

#include "core/cviai_types_mem.h"
#include "core/cviai_types_mem_internal.h"
#include "face_utils.hpp"

#define NAME_BBOX "face_rpn_bbox_pred_"
#define NAME_SCORE "face_rpn_cls_prob_reshape_"
#define NAME_LANDMARK "face_rpn_landmark_pred_"
#define FACE_POINTS_SIZE 5

namespace cviai {

RetinaFace::RetinaFace() {
  mp_mi = std::make_unique<CvimodelInfo>();
  mp_mi->conf.input_mem_type = CVI_MEM_DEVICE;
}

RetinaFace::~RetinaFace() {}

int RetinaFace::initAfterModelOpened(std::vector<initSetup> *data) {
  std::vector<anchor_cfg> cfg;
  anchor_cfg tmp;
  tmp.SCALES = {32, 16};
  tmp.BASE_SIZE = 16;
  tmp.RATIOS = {1.0};
  tmp.ALLOWED_BORDER = 9999;
  tmp.STRIDE = 32;
  cfg.push_back(tmp);

  tmp.SCALES = {8, 4};
  tmp.BASE_SIZE = 16;
  tmp.RATIOS = {1.0};
  tmp.ALLOWED_BORDER = 9999;
  tmp.STRIDE = 16;
  cfg.push_back(tmp);

  tmp.SCALES = {2, 1};
  tmp.BASE_SIZE = 16;
  tmp.RATIOS = {1.0};
  tmp.ALLOWED_BORDER = 9999;
  tmp.STRIDE = 8;
  cfg.push_back(tmp);

  std::vector<std::vector<anchor_box>> anchors_fpn = generate_anchors_fpn(false, cfg);
  std::map<std::string, std::vector<anchor_box>> anchors_fpn_map;
  for (size_t i = 0; i < anchors_fpn.size(); i++) {
    std::string key = "stride" + std::to_string(m_feat_stride_fpn[i]) + "_dequant";
    anchors_fpn_map[key] = anchors_fpn[i];
    m_num_anchors[key] = anchors_fpn[i].size();
  }

  for (size_t i = 0; i < m_feat_stride_fpn.size(); i++) {
    std::string key = "stride" + std::to_string(m_feat_stride_fpn[i]) + "_dequant";
    std::string landmark_str = NAME_LANDMARK + key;
    CVI_TENSOR *out =
        CVI_NN_GetTensorByName(landmark_str.c_str(), mp_mi->out.tensors, mp_mi->out.num);
    CVI_SHAPE landmark_shape = CVI_NN_TensorShape(out);
    int stride = m_feat_stride_fpn[i];

    m_anchors[landmark_str] =
        anchors_plane(landmark_shape.dim[2], landmark_shape.dim[3], stride, anchors_fpn_map[key]);
  }

  if (data->size() != 1) {
    LOGE("Retina face only has 1 input.\n");
    return CVI_FAILURE;
  }
  for (int i = 0; i < 3; i++) {
    (*data)[0].factor[i] = 1;
  }
  (*data)[0].use_quantize_scale = true;
  m_export_chn_attr = true;
  return CVI_SUCCESS;
}

int RetinaFace::inference(VIDEO_FRAME_INFO_S *srcFrame, cvai_face_t *meta) {
  std::vector<VIDEO_FRAME_INFO_S *> frames;
  CVI_TENSOR *input =
      CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, mp_mi->in.tensors, mp_mi->in.num);
  for (uint32_t b = 0; b < (uint32_t)input->shape.dim[0]; b++) {
    frames.push_back(&srcFrame[b]);
  }
  int ret = run(frames);
  int image_width = input->shape.dim[3];
  int image_height = input->shape.dim[2];
  outputParser(image_width, image_height, srcFrame->stVFrame.u32Width, srcFrame->stVFrame.u32Height,
               meta);
  return ret;
}

void RetinaFace::outputParser(int image_width, int image_height, int frame_width, int frame_height,
                              cvai_face_t *meta) {
  CVI_TENSOR *input =
      CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, mp_mi->in.tensors, mp_mi->in.num);
  for (uint32_t b = 0; b < (uint32_t)input->shape.dim[0]; b++) {
    std::vector<cvai_face_info_t> vec_bbox;
    std::vector<cvai_face_info_t> vec_bbox_nms;
    for (size_t i = 0; i < m_feat_stride_fpn.size(); i++) {
      std::string key = "stride" + std::to_string(m_feat_stride_fpn[i]) + "_dequant";

      std::string score_str = NAME_SCORE + key;
      CVI_TENSOR *out =
          CVI_NN_GetTensorByName(score_str.c_str(), mp_mi->out.tensors, mp_mi->out.num);
      CVI_SHAPE score_shape = CVI_NN_TensorShape(out);
      size_t score_size = score_shape.dim[1] * score_shape.dim[2] * score_shape.dim[3];
      float *score_blob = (float *)CVI_NN_TensorPtr(out) + (b * score_size);
      score_blob += score_size / 2;

      std::string bbox_str = NAME_BBOX + key;
      out = CVI_NN_GetTensorByName(bbox_str.c_str(), mp_mi->out.tensors, mp_mi->out.num);
      CVI_SHAPE blob_shape = CVI_NN_TensorShape(out);
      size_t blob_size = blob_shape.dim[1] * blob_shape.dim[2] * blob_shape.dim[3];
      float *bbox_blob = (float *)CVI_NN_TensorPtr(out) + (b * blob_size);

      std::string landmark_str = NAME_LANDMARK + key;
      out = CVI_NN_GetTensorByName(landmark_str.c_str(), mp_mi->out.tensors, mp_mi->out.num);
      CVI_SHAPE landmark_shape = CVI_NN_TensorShape(out);
      size_t landmark_size = landmark_shape.dim[1] * landmark_shape.dim[2] * landmark_shape.dim[3];
      float *landmark_blob = (float *)CVI_NN_TensorPtr(out) + (b * landmark_size);

      int width = landmark_shape.dim[3];
      int height = landmark_shape.dim[2];
      size_t count = width * height;
      size_t num_anchor = m_num_anchors[key];

      std::vector<anchor_box> anchors = m_anchors[landmark_str];
      for (size_t num = 0; num < num_anchor; num++) {
        for (size_t j = 0; j < count; j++) {
          float conf = score_blob[j + count * num];
          if (conf <= m_model_threshold) {
            continue;
          }
          cvai_face_info_t box;
          memset(&box, 0, sizeof(box));
          box.pts.size = 5;
          box.pts.x = (float *)malloc(sizeof(float) * box.pts.size);
          box.pts.y = (float *)malloc(sizeof(float) * box.pts.size);
          box.bbox.score = conf;

          cv::Vec4f regress;
          float dx = bbox_blob[j + count * (0 + num * 4)];
          float dy = bbox_blob[j + count * (1 + num * 4)];
          float dw = bbox_blob[j + count * (2 + num * 4)];
          float dh = bbox_blob[j + count * (3 + num * 4)];
          regress = cv::Vec4f(dx, dy, dw, dh);
          bbox_pred(anchors[j + count * num], regress, box.bbox);

          for (size_t k = 0; k < box.pts.size; k++) {
            box.pts.x[k] = landmark_blob[j + count * (num * 10 + k * 2)];
            box.pts.y[k] = landmark_blob[j + count * (num * 10 + k * 2 + 1)];
          }
          landmark_pred(anchors[j + count * num], box.pts);
          vec_bbox.push_back(box);
        }
      }
    }
    // DO nms on output result
    vec_bbox_nms.clear();
    NonMaximumSuppression(vec_bbox, vec_bbox_nms, 0.4, 'u');
    // Init face meta
    cvai_face_t *facemeta = &meta[b];
    facemeta->width = image_width;
    facemeta->height = image_height;
    if (vec_bbox_nms.size() == 0) {
      facemeta->size = vec_bbox_nms.size();
      facemeta->info = NULL;
      return;
    }
    CVI_AI_MemAllocInit(vec_bbox_nms.size(), FACE_POINTS_SIZE, facemeta);
    if (m_skip_vpss_preprocess) {
      for (uint32_t i = 0; i < facemeta->size; ++i) {
        clip_boxes(image_width, image_height, vec_bbox_nms[i].bbox);
        facemeta->info[i].bbox.x1 = vec_bbox_nms[i].bbox.x1;
        facemeta->info[i].bbox.x2 = vec_bbox_nms[i].bbox.x2;
        facemeta->info[i].bbox.y1 = vec_bbox_nms[i].bbox.y1;
        facemeta->info[i].bbox.y2 = vec_bbox_nms[i].bbox.y2;
        facemeta->info[i].bbox.score = vec_bbox_nms[i].bbox.score;

        for (int j = 0; j < FACE_POINTS_SIZE; ++j) {
          facemeta->info[i].pts.x[j] = vec_bbox_nms[i].pts.x[j];
          facemeta->info[i].pts.y[j] = vec_bbox_nms[i].pts.y[j];
        }
      }
    } else {
      // Recover coordinate if internal vpss engine is used.
      facemeta->width = frame_width;
      facemeta->height = frame_height;
      facemeta->rescale_type = m_vpss_config[0].rescale_type;
      for (uint32_t i = 0; i < facemeta->size; ++i) {
        clip_boxes(image_width, image_height, vec_bbox_nms[i].bbox);
        cvai_face_info_t info =
            info_rescale_c(image_width, image_height, frame_width, frame_height, vec_bbox_nms[i]);
        facemeta->info[i].bbox.x1 = info.bbox.x1;
        facemeta->info[i].bbox.x2 = info.bbox.x2;
        facemeta->info[i].bbox.y1 = info.bbox.y1;
        facemeta->info[i].bbox.y2 = info.bbox.y2;
        facemeta->info[i].bbox.score = info.bbox.score;
        for (int j = 0; j < FACE_POINTS_SIZE; ++j) {
          facemeta->info[i].pts.x[j] = info.pts.x[j];
          facemeta->info[i].pts.y[j] = info.pts.y[j];
        }
        CVI_AI_FreeCpp(&info);
      }
    }
    // Clear original bbox. bbox_nms does not need to free since it points to bbox.
    for (size_t i = 0; i < vec_bbox.size(); ++i) {
      CVI_AI_FreeCpp(&vec_bbox[i].pts);
    }
  }
}

}  // namespace cviai
